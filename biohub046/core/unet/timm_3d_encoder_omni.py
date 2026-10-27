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
from einops import rearrange, repeat
from timm.models import EfficientNetFeatures


class SlidingSlicer(nn.Module):
    def __init__(self, slice_size=3, stride=1):
        super(SlidingSlicer, self).__init__()

        # Create convolution layer to simulate the sliding slice operation
        self.conv = nn.Conv3d(1, slice_size, kernel_size=(slice_size, 1, 1), stride=(stride, 1, 1),
                              bias=False, padding=(slice_size // 2, 0, 0))

        # Set weights to simulate identity operation and bias to 0
        with torch.no_grad():
            self.conv.weight.data.fill_(0)
            for i in range(slice_size):
                self.conv.weight.data[i, 0, i] = 1

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


class TimmUniversalEncoderOmni(nn.Module):
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
        depth: int = 5,
        output_stride: int = 32,
        use_tsm: bool = False,
        use_intermediate_conv: bool = False,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the encoder.

        Args:
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional arguments passed to `timm.create_model`.

        Note:
            Creates two separate models: one for z-axis (model_z) and one for xy-axes (model_xy).
        """
        # At the moment we do not support models with more than 5 stages,
        # but can be reconfigured in the future.
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__()
        self.name = name

        # Default model configuration for feature extraction
        common_kwargs = dict(
            in_chans=in_channels,
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

        if self._is_transformer_style:
            # Transformer-like models (start at scale 4)
            if "tresnet" in name:
                # 'tresnet' models start feature extraction at stage 1,
                # so out_indices=(1, 2, 3, 4) for depth=5.
                common_kwargs["out_indices"] = tuple(range(1, depth))
            else:
                # Most transformer-like models use out_indices=(0, 1, 2, 3) for depth=5.
                common_kwargs["out_indices"] = tuple(range(depth - 1))

            # Create separate models for z-axis and xy-axes
            self.model_z = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs.copy(), kwargs)
            )
            self.model_xy = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs.copy(), kwargs)
            )
            # Add a dummy output channel (0) to align with traditional encoder structures.
            self._out_channels = (
                [in_channels] + [0] + self.model_z.feature_info.channels()
            )
            self._strides = [4, 2, 2, 2] + [1] * (depth - 4)
        else:
            if "dla" in name:
                # For 'dla' models, out_indices starts at 0 and matches the input size.
                common_kwargs["out_indices"] = tuple(range(1, depth + 1))
            if self._is_vgg_style:
                common_kwargs["out_indices"] = tuple(range(depth + 1))

            # Create separate models for z-axis and xy-axes
            self.model_z = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs.copy(), kwargs)
            )
            self.model_xy = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs.copy(), kwargs)
            )

            if self._is_vgg_style:
                self._out_channels = self.model_z.feature_info.channels()
            else:
                self._out_channels = [in_channels] + self.model_z.feature_info.channels()

            self._strides = [2, 2, 2, 2, 2] + [1] * (depth - 5)

        self._in_channels = in_channels
        self._depth = depth
        self._output_stride = output_stride
        self.use_tsm = use_tsm
        self.use_intermediate_conv = use_intermediate_conv
        self.in_channels = in_channels

        if use_tsm:
            self.tsm = TemporalShift()

        if use_intermediate_conv:
            self.intermediate_convs = nn.ModuleList()
            for out_ch in self._out_channels[1:]:
                self.intermediate_convs.append(
                    IntermediateBlock3D(out_ch, out_ch)
                )

        if in_channels > 1:
            self.sliding_slicer = SlidingSlicer(slice_size=in_channels, stride=1)

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

        # EfficientNet is not supported with separate z/xy models
        if isinstance(self.model_z, EfficientNetFeatures):
            raise NotImplementedError("EfficientNet is not supported with separate z/xy model architecture")

        B, _, D, H, W = x.shape
        ori_x = x
        features = []
        current_stage = 0
        # x1: z-axis (depth), x2: y-axis (height), x3: x-axis (width)
        x1 = rearrange(x, 'b c d h w -> (b d) c h w')
        x2 = rearrange(x, 'b c d h w -> (b h) c d w')
        x3 = rearrange(x, 'b c d h w -> (b w) c h d')

        # Iterate over both models in parallel (z-axis and xy-axes)
        for (name_z, module_z), (name_xy, module_xy) in zip(
            self.model_z.items(), self.model_xy.items()
        ):
            x1 = module_z(x1)   # z-axis uses model_z
            x2 = module_xy(x2)  # y-axis uses model_xy
            x3 = module_xy(x3)  # x-axis uses model_xy

            if name_z in self.model_z.return_layers:
                if self._is_channel_last:
                    x1 = rearrange(x1, '(b d) h w c -> b c d h w', b=B)
                    x2 = rearrange(x2, '(b h) d w c -> b c d h w', b=B)
                    x3 = rearrange(x3, '(b w) h d c -> b c d h w', b=B)
                else:
                    x1 = rearrange(x1, '(b d) c h w -> b c d h w', b=B)
                    x2 = rearrange(x2, '(b h) c d w -> b c d h w', b=B)
                    x3 = rearrange(x3, '(b w) c h d -> b c d h w', b=B)

                stride = self._strides[current_stage]

                if stride > 1:
                    x1 = F.avg_pool3d(x1, kernel_size=(stride, 1, 1), stride=(stride, 1, 1), padding=0)
                    x2 = F.avg_pool3d(x2, kernel_size=(1, stride, 1), stride=(1, stride, 1), padding=0)
                    x3 = F.avg_pool3d(x3, kernel_size=(1, 1, stride), stride=(1, 1, stride), padding=0)

                x = (x1 + x2 + x3) / 3.0
                features.append(x)

                if self._is_channel_last:
                    x1 = rearrange(x1, 'b c d h w -> (b d) h w c')
                    x2 = rearrange(x2, 'b c d h w -> (b h) d w c')
                    x3 = rearrange(x3, 'b c d h w -> (b w) h d c')
                else:
                    x1 = rearrange(x1, 'b c d h w -> (b d) c h w')
                    x2 = rearrange(x2, 'b c d h w -> (b h) c d w')
                    x3 = rearrange(x3, 'b c d h w -> (b w) c h d')

                current_stage += 1

        # Add dummy feature for scale 1/2 if missing (transformer-style models)
        if self._is_transformer_style:
            dummy = torch.empty([B, 0, D // 2, H // 2, W // 2], dtype=x.dtype, device=x.device)
            features = [dummy] + features

        # Add input tensor as scale 1 feature if `self._is_vgg_style` is False
        if not self._is_vgg_style:
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
                # Handle both model_z and model_xy prefixes
                if not key.startswith("model_z.") and not key.startswith("model_xy."):
                    # Legacy: duplicate to both models
                    if key.startswith("model."):
                        base_key = key[6:]  # Remove "model."
                    else:
                        base_key = key
                    if "gernet" in self.name:
                        base_key = base_key.replace(".stages.", ".stages_")
                    state_dict["model_z." + base_key] = state_dict[key].clone()
                    state_dict["model_xy." + base_key] = state_dict.pop(key)
                elif "gernet" in self.name:
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
