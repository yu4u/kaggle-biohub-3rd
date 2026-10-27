"""
TimmUniversalEncoder provides a unified feature extraction interface built on the
`timm` library, supporting both traditional-style (e.g., ResNet) and transformer-style
models (e.g., Swin Transformer, ConvNeXt).

This encoder produces consistent multi-level feature maps for semantic segmentation tasks.
It allows configuring the number of feature extraction stages (`depth`) and adjusting
`output_stride` when supported.

Key Features:
- Flexible model selection using `timm.create_model`.
- Unified multi-level output across different model hierarchies.
- Automatic alignment for inconsistent feature scales:
  - Transformer-style models (start at 1/4 scale): Insert dummy features for 1/2 scale.
  - VGG-style models (include scale-1 features): Align outputs for compatibility.
- Easy access to feature scale information via the `reduction` property.

Feature Scale Differences:
- Traditional-style models (e.g., ResNet): Scales at 1/2, 1/4, 1/8, 1/16, 1/32.
- Transformer-style models (e.g., Swin Transformer): Start at 1/4 scale, skip 1/2 scale.
- VGG-style models: Include scale-1 features (input resolution).

Notes:
- `output_stride` is unsupported in some models, especially transformer-based architectures.
- Special handling for models like TResNet and DLA to ensure correct feature indexing.
- VGG-style models use `_is_vgg_style` to align scale-1 features with standard outputs.
"""

from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from timm.models import EfficientNetFeatures


def _widen_stem(model, in_planes):
    """Widen the stem conv to in_planes copies of its input, extra planes zeroed.

    timm's adapt_input_conv SUMS the RGB filters at in_chans=1 but TILES them
    and rescales by 3/in_chans for anything else, so asking timm for 2 channels
    does not hand back the 1-channel stem plus a fresh plane: measured corr
    0.795 and 0.604x its magnitude on tf_efficientnetv2_s. Creating the model
    at the baseline width and widening here keeps plane 0 bit-identical to the
    1-plane run and starts the extra planes at zero, so an added-plane A/B does
    not silently double as an initialization A/B. Zero weights still train --
    only a zero INPUT would leave them dead.
    """
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    w = conv.weight.data
    pad = torch.zeros(w.shape[0], w.shape[1] * (in_planes - 1), *w.shape[2:],
                      dtype=w.dtype, device=w.device)
    conv.weight = nn.Parameter(torch.cat([w, pad], dim=1))
    conv.in_channels = w.shape[1] * in_planes


class SlidingSlicer(nn.Module):
    def __init__(self, slice_size=3, stride=1, trainable=False):
        super(SlidingSlicer, self).__init__()

        # Create convolution layer to simulate the sliding slice operation
        self.conv = nn.Conv3d(1, slice_size, kernel_size=(slice_size, 1, 1), stride=(stride, 1, 1),
                              bias=False, padding=(slice_size // 2, 0, 0))

        # Set weights to simulate identity operation and bias to 0
        with torch.no_grad():
            self.conv.weight.data.fill_(0)
            for i in range(slice_size):
                self.conv.weight.data[i, 0, i] = 1

        # trainable=True (dl_413): keep the identity init but let the z-window
        # mixing weights learn with the rest of the trunk
        if not trainable:
            for param in self.conv.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.conv(x)
        return out


class TemporalShift(nn.Module):
    def __init__(self, fold_div=8):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x):
        b, c, d, h, w = x.shape
        fold = c // self.fold_div
        if fold == 0 or d == 1:
            return x

        out = torch.empty_like(x)

        out[:, :fold, :-1] = x[:, :fold, 1:]
        out[:, :fold, -1] = 0

        out[:, fold:2 * fold, 1:] = x[:, fold:2 * fold, :-1]
        out[:, fold:2 * fold, 0] = 0

        out[:, 2 * fold:] = x[:, 2 * fold:]
        return out


class ZMixBlock(nn.Module):
    """Zero-init residual (3,1,1) conv: mixes information along z at stages
    where z-pooling is skipped (iso_z). Identity at initialization, so
    pretrained 2D behavior is preserved at the start of training."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class IntermediateBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
        )
        nn.init.zeros_(self.f[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class TimmUniversalEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    various model architectures, including traditional-style and transformer-style models.

    Features:
        - Supports configurable depth and output stride.
        - Ensures consistent multi-level feature extraction across diverse models.
        - Compatible with convolutional and transformer-like backbones.
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        in_planes: int = 1,
        depth: int = 5,
        output_stride: int = 32,
        use_tsm: bool = False,
        use_intermediate_conv: bool = False,
        iso_z: bool = False,
        iso_z_zmix: bool = True,
        slicer_trainable: bool = False,
        grad_ckpt: bool = False,
        stem_init: str = "zero",
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the encoder.

        Args:
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
                Doubles as the sliding z-slice window: > 1 stacks that many
                ADJACENT z-slices as 2D channels (SlidingSlicer, 1 plane in).
            in_planes (int): Number of real feature channels per voxel (dl_156
                feeds 2: normalized intensity + local contrast). Multiplies
                in_channels for the timm stem. Only one of the two may exceed
                1 -- SlidingSlicer is a Conv3d(1, ...) and cannot slice a
                multi-plane input.
            depth (int): Number of feature stages to extract (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
        # At the moment we do not support models with more than 5 stages,
        # but can be reconfigured in the future.
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__()
        self.name = name
        assert in_planes == 1 or in_channels == 1, (
            f"in_planes={in_planes} with in_channels={in_channels}: "
            "the z-slice window cannot stack multi-plane input")

        # Default model configuration for feature extraction
        common_kwargs = dict(
            # stem_init="timm": timm's adapt_input_conv builds the in_planes
            # stem from the pretrained RGB filters (tiled, x3/in_planes)
            in_chans=in_channels * (in_planes if stem_init == "timm" else 1),
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # Ｎot all models support output stride argument, drop it by default
        if output_stride == 32:
            common_kwargs.pop("output_stride")

        # Load a temporary model to analyze its feature hierarchy
        try:
            with torch.device("meta"):
                tmp_model = timm.create_model(name, features_only=True)
        except Exception:
            tmp_model = timm.create_model(name, features_only=True)

        # Check if model output is in channel-last format (NHWC)
        self._is_channel_last = getattr(tmp_model, "output_fmt", None) == "NHWC"

        # Determine the model's downsampling pattern and set hierarchy flags
        encoder_stage = len(tmp_model.feature_info.reduction())
        reduction_scales = list(tmp_model.feature_info.reduction())

        if reduction_scales == [2 ** (i + 2) for i in range(encoder_stage)]:
            # Transformer-style downsampling: scales (4, 8, 16, 32)
            self._is_transformer_style = True
            self._is_vgg_style = False
        elif reduction_scales == [2 ** (i + 1) for i in range(encoder_stage)]:
            # Traditional-style downsampling: scales (2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = False
        elif reduction_scales == [2**i for i in range(encoder_stage)]:
            # Vgg-style models including scale 1: scales (1, 2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = True
        else:
            raise ValueError("Unsupported model downsampling pattern.")

        if iso_z and self._is_vgg_style:
            raise NotImplementedError("iso_z is not supported for vgg-style encoders")

        if self._is_transformer_style:
            # Transformer-like models (start at scale 4)
            if "tresnet" in name:
                # 'tresnet' models start feature extraction at stage 1,
                # so out_indices=(1, 2, 3, 4) for depth=5.
                common_kwargs["out_indices"] = tuple(range(1, depth))
            else:
                # Most transformer-like models use out_indices=(0, 1, 2, 3) for depth=5.
                common_kwargs["out_indices"] = tuple(range(depth - 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )
            # Add a dummy output channel (0) to align with traditional encoder structures.
            self._out_channels = (
                [in_planes * in_channels] + [0] + self.model.feature_info.channels()
            )
            self._strides = [4, 2, 2, 2] + [1] * (depth - 4)
        else:
            if "dla" in name:
                # For 'dla' models, out_indices starts at 0 and matches the input size.
                common_kwargs["out_indices"] = tuple(range(1, depth + 1))
            if self._is_vgg_style:
                common_kwargs["out_indices"] = tuple(range(depth + 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )

            if self._is_vgg_style:
                self._out_channels = self.model.feature_info.channels()
            else:
                self._out_channels = ([in_planes * in_channels]
                                      + self.model.feature_info.channels())

            self._strides = [2, 2, 2, 2, 2] + [1] * (depth - 5)

        assert stem_init in ("zero", "timm"), stem_init
        if in_planes > 1 and stem_init == "zero":
            _widen_stem(self.model, in_planes)

        self._in_channels = in_channels
        self._depth = depth
        self._output_stride = output_stride
        self.use_tsm = use_tsm
        # dl_492: activation checkpointing of every encoder block during
        # training (numerically identical, ~30% slower). Lets tf_efficientnet_b7
        # keep the recipe's bs8 on 32 GB (31.7 GB OOM without it). Only the
        # EfficientNet path re-runs blocks by hand, so only it honours this.
        self.grad_ckpt = grad_ckpt
        self.use_intermediate_conv = use_intermediate_conv
        self.in_channels = in_channels

        # iso_z: the input z spacing is ~4x coarser than y/x, so skip z-pooling
        # while the YX grid is still finer than 1/4 (the grid becomes isotropic
        # at 1/4); pool z together with YX from there on. A ZMixBlock provides
        # z information exchange at the stages whose z-pool was removed;
        # iso_z_zmix=False keeps the pool schedule change only (no extra convs,
        # z slices stay independent until the first joint pool).
        self.z_mix = nn.ModuleDict()
        if iso_z:
            channels = self.model.feature_info.channels()
            yx = 4 if self._is_transformer_style else 2  # YX scale of stage 0
            for i in range(len(channels)):
                if yx <= 4 and self._strides[i] > 1:
                    self._strides[i] = 1
                    if iso_z_zmix:
                        self.z_mix[str(i)] = ZMixBlock(channels[i])
                yx *= 2

        if use_tsm:
            self.tsm = TemporalShift()

        if use_intermediate_conv:
            self.intermediate_convs = nn.ModuleList()
            for out_ch in self._out_channels[1:]:
                self.intermediate_convs.append(
                    IntermediateBlock3D(out_ch, out_ch)
                )

        if in_channels > 1:
            self.sliding_slicer = SlidingSlicer(slice_size=in_channels, stride=1,
                                                trainable=slicer_trainable)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

        Returns:
            list[torch.Tensor]: List of feature maps at different scales.
        """
        if self.in_channels > 1:
            x = self.sliding_slicer(x)

        if isinstance(self.model, EfficientNetFeatures):
            return self.forward_effnet(x)

        B, _, D, H, W = x.shape
        ori_x = x
        features = []
        current_stage = 0
        x = rearrange(x, 'b c d h w -> (b d) c h w')

        for i, (name, module) in enumerate(self.model.items()):
            # dl_493: same training-only activation checkpointing as forward_effnet.
            # timm stages checkpoint per block through their own flag (a whole-stage
            # checkpoint still materialises every block of the stage on recompute:
            # caformer_m36 stage 1 alone OOMs a 24 GB 3090 at bs8); other modules
            # (stem) are checkpointed whole.
            ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
            if hasattr(module, "set_grad_checkpointing"):
                module.set_grad_checkpointing(ckpt)
                x = module(x)
            elif ckpt:
                x = torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
            else:
                x = module(x)

            if name in self.model.return_layers:
                if self._is_channel_last:
                    x = rearrange(x, '(b d) h w c -> b c d h w', b=B)
                else:
                    x = rearrange(x, '(b d) c h w -> b c d h w', b=B)

                stride = self._strides[current_stage]

                if stride > 1:
                    x = F.avg_pool3d(x, kernel_size=(stride, 1, 1), stride=(stride, 1, 1), padding=0)

                # TSM is independent of the z-pool schedule; keep it running at
                # stages whose z-pool iso_z removed (stride == 1).
                if self.use_tsm:
                    x = self.tsm(x)

                if str(current_stage) in self.z_mix:
                    x = self.z_mix[str(current_stage)](x)

                if self.use_intermediate_conv:
                    x = self.intermediate_convs[current_stage](x)

                features.append(x)

                if self._is_channel_last:
                    x = rearrange(x, 'b c d h w -> (b d) h w c')
                else:
                    x = rearrange(x, 'b c d h w -> (b d) c h w')

                current_stage += 1

        # Add dummy feature for scale 1/2 if missing (transformer-style models)
        if self._is_transformer_style:
            # z at the 1/2 YX scale: unpooled when iso_z removed the stage-0
            # z-pool, otherwise pooled by 2. The decoder is shape-driven, so a
            # stale D // 2 here would squeeze z back down at the finest stage.
            dummy_d = D if self._strides[0] == 1 else D // 2
            dummy = torch.empty([B, 0, dummy_d, H // 2, W // 2], dtype=x.dtype, device=x.device)
            features = [dummy] + features

        # Add input tensor as scale 1 feature if `self._is_vgg_style` is False
        if not self._is_vgg_style:
            features = [ori_x] + features

        return features

    def forward_effnet(self, x) -> list[torch.Tensor]:
        B, _, D, H, W = x.shape
        ori_x = x
        features = []
        current_stage = 0
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        x = self.model.conv_stem(x)
        x = self.model.bn1(x)

        if 0 in self.model._stage_out_idx:
            # features.append(x)  # add stem out
            raise ValueError("Unsupported model downsampling pattern.")

        for i, b in enumerate(self.model.blocks):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)

            if i + 1 in self.model._stage_out_idx:
                x = rearrange(x, '(b d) c h w -> b c d h w', b=B)
                stride = self._strides[current_stage]

                if stride > 1:
                    x = F.avg_pool3d(x, kernel_size=(stride, 1, 1), stride=(stride, 1, 1), padding=0)

                if self.use_tsm:
                    x = self.tsm(x)

                if str(current_stage) in self.z_mix:
                    x = self.z_mix[str(current_stage)](x)

                if self.use_intermediate_conv:
                    x = self.intermediate_convs[current_stage](x)

                features.append(x)
                x = rearrange(x, 'b c d h w -> (b d) c h w')
                current_stage += 1

        features = [ori_x] + features

        return features

    @property
    def out_channels(self) -> list[int]:
        """
        Returns the number of output channels for each feature stage.

        Returns:
            list[int]: A list of channel dimensions at each scale.
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """
        Returns the effective output stride based on the model depth.

        Returns:
            int: The effective output stride.
        """
        return int(min(self._output_stride, 2**self._depth))

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if not key.startswith("model."):
                    new_key = "model." + key
                if "gernet" in self.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)


def _merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries, ensuring no duplicate keys exist.

    Args:
        a (dict): Base dictionary.
        b (dict): Additional parameters to merge.

    Returns:
        dict: A merged dictionary.
    """
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b
