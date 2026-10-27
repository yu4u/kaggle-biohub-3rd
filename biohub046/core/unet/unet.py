from typing import Any, Optional, Union, Callable, Sequence
import torch
import torch.nn as nn

from .timm_3d_encoder import TimmUniversalEncoder
from .timm_3d_encoder_omni import TimmUniversalEncoderOmni
from .decoder import UnetDecoder, PyramidPoolingModule3D


def is_transformer(model_name):
    substrings = ["swin", "vit"]
    return any(substring in model_name for substring in substrings)


class Unet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_type: str = "normal",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        use_attention: bool = False,
        decoder_interpolation_mode: str = "nearest",
        in_channels: int = 3,
        in_planes: int = 1,
        classes: int = 1,
        img_size: Optional[int] = None,
        pretrained: bool = True,
        iso_z: bool = False,
        iso_z_zmix: bool = True,
        slicer_trainable: bool = False,
        grad_ckpt: bool = False,
        use_ppm: bool = False,
        ppm_out_channels: int = 256,
        ppm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        div_head: bool = False,
        coarse_head: bool = False,
        deep_supervision: int = 0,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        # coarse_head (arch=unet25d_coarse): truncate the decoder at the
        # stride-4 stage and predict [cls_logit, dz, dy, dx] on the [1,4,4]
        # grid (z full-res REQUIRES iso_z so the stride-4 skip keeps full z).
        self.coarse_head = coarse_head
        if coarse_head:
            assert iso_z, "coarse_head needs iso_z=true (full-z stride-4 skip)"
            assert not div_head, "coarse_head + div_head is unsupported"
            # stop the decoder at the stride-4 stage for ANY depth (a
            # hard-coded 3 would leave a stride-2 output for encoder_depth=4);
            # the unused decoder_channels tail is dropped
            encoder_depth_dec = encoder_depth - 2
            decoder_channels = tuple(decoder_channels[:encoder_depth_dec])
        else:
            encoder_depth_dec = encoder_depth

        if is_transformer(encoder_name) or img_size is not None:
            # dl_543: coatnet / maxvit are not in is_transformer's list but bind
            # their rel-pos attention to a fixed 2D size — honour an explicit one
            kwargs["img_size"] = img_size

        if encoder_type == "normal":
            encoder_class = TimmUniversalEncoder
            kwargs["iso_z"] = iso_z
            kwargs["iso_z_zmix"] = iso_z_zmix
            kwargs["in_planes"] = in_planes
            kwargs["slicer_trainable"] = slicer_trainable
            kwargs["grad_ckpt"] = grad_ckpt
        elif encoder_type == "omni":
            if iso_z:
                raise NotImplementedError("iso_z is only supported for encoder_type='normal'")
            if in_planes > 1:
                raise NotImplementedError(
                    "in_planes > 1 is only supported for encoder_type='normal'")
            if slicer_trainable:
                raise NotImplementedError(
                    "slicer_trainable is only supported for encoder_type='normal'")
            encoder_class = TimmUniversalEncoderOmni
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.encoder = encoder_class(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=pretrained,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        # PPM (Pyramid Pooling Module) as bridge between encoder and decoder
        self.use_ppm = use_ppm
        if use_ppm:
            encoder_last_channels = self.encoder.out_channels[-1]
            self.ppm = PyramidPoolingModule3D(
                in_channels=encoder_last_channels,
                out_channels=ppm_out_channels,
                pool_sizes=tuple(ppm_pool_sizes),
            )
            # Modify encoder channels for decoder
            encoder_channels_for_decoder = list(self.encoder.out_channels)
            encoder_channels_for_decoder[-1] = ppm_out_channels
        else:
            self.ppm = None
            encoder_channels_for_decoder = self.encoder.out_channels

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels_for_decoder,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth_dec,
            use_batchnorm=decoder_use_batchnorm,
            use_attention=use_attention,
            interpolation_mode=decoder_interpolation_mode,
        )

        if coarse_head:
            self.cls_head = nn.Conv3d(decoder_channels[-1], 1, 3, padding=1)
            self.off_head = nn.Conv3d(decoder_channels[-1], 3, 3, padding=1)
            # RetinaNet prior init: an untrained head must NOT sit at p~0.5,
            # or the first mini-val extracts ~10^4 peaks/frame and hangs the
            # O(n^3) linking. p0=0.01 -> bias = -log((1-p0)/p0) ~ -4.6.
            nn.init.constant_(self.cls_head.bias, -4.595)
            self.segmentation_head = None
        else:
            self.segmentation_head = nn.Conv3d(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                kernel_size=3,
                padding=1,
            )
        # multi-scale deep supervision: aux 1ch heads on the decoder stages
        # ABOVE full-res (deep_supervision=k -> stride-2..stride-2^k levels).
        # Training-time only: forward returns (main, aux_s2, aux_s4, ...);
        # every eval path already unwraps tuples to [0].
        self.deep_supervision = int(deep_supervision)
        if self.deep_supervision:
            assert not coarse_head, "deep_supervision + coarse_head unsupported"
            assert self.deep_supervision <= len(decoder_channels) - 1
            # decoder_channels[-2] is the stride-2 stage, [-3] stride-4, ...
            self.ds_heads = nn.ModuleList([
                nn.Conv3d(decoder_channels[-2 - k], 1, 3, padding=1)
                for k in range(self.deep_supervision)])
        # dl_195: parallel daughter-heatmap head on the final decoder feature
        # output becomes (B, classes+1, Z, Y, X)
        self.div_head = None
        if div_head:
            self.div_head = nn.Sequential(
                nn.Conv3d(decoder_channels[-1], 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, 3, padding=1),
            )
        # self.initialize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        # Apply PPM to the deepest feature
        if self.use_ppm and self.ppm is not None:
            features[-1] = self.ppm(features[-1])

        if self.deep_supervision and self.training:
            feats = self.decoder(features, return_all=True)
            main = self.segmentation_head(feats[-1])
            # feats[-2] = stride-2 stage, feats[-3] = stride-4, ...
            return tuple([main] + [self.ds_heads[k](feats[-2 - k])
                                   for k in range(self.deep_supervision)])
        decoder_output = self.decoder(features)
        if self.coarse_head:
            # (B, 4, Z, Y/4, X/4) = [cls_logit, dz, dy, dx]
            return torch.cat([self.cls_head(decoder_output),
                              self.off_head(decoder_output)], 1)
        masks = self.segmentation_head(decoder_output)
        if self.div_head is not None:
            masks = torch.cat([masks, self.div_head(decoder_output)], 1)
        return masks
