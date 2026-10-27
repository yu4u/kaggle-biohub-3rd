from typing import Optional, Sequence, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBNReLU(nn.Sequential):
    """Conv3d + BatchNorm3d + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPoolingModule3D(nn.Module):
    """Pyramid Pooling Module (PPM) for 3D volumes.

    From PSPNet: Pyramid Scene Parsing Network (https://arxiv.org/abs/1612.01105)

    Applies adaptive average pooling at multiple scales, reduces channels with 1x1 conv,
    upsamples back to original size, and concatenates with input features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        pool_sizes: Tuple[int, ...] = (1, 2, 4, 8),
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (after final conv)
            pool_sizes: Pooling sizes for each pyramid level
        """
        super().__init__()

        self.pool_sizes = pool_sizes
        hidden_channels = in_channels // len(pool_sizes)

        # Pooling branches
        self.branches = nn.ModuleList()
        for pool_size in pool_sizes:
            branch = nn.Sequential(
                nn.AdaptiveAvgPool3d(pool_size),
                Conv3dBNReLU(in_channels, hidden_channels, kernel_size=1),
            )
            self.branches.append(branch)

        # Final conv to reduce concatenated channels
        self.bottleneck = Conv3dBNReLU(
            in_channels + hidden_channels * len(pool_sizes),
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Output tensor of shape (B, out_channels, D, H, W)
        """
        size = x.shape[2:]  # (D, H, W)

        # Collect pooled features from each branch
        pooled_features = [x]
        for branch in self.branches:
            pooled = branch(x)
            upsampled = F.interpolate(pooled, size=size, mode='trilinear', align_corners=False)
            pooled_features.append(upsampled)

        # Concatenate and reduce
        out = torch.cat(pooled_features, dim=1)
        out = self.bottleneck(out)

        return out


class Conv3dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()
        super().__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class SSEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.sSE = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.sSE(x)


class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        use_attention: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = SSEModule(in_channels + skip_channels) if use_attention else nn.Identity()
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = SCSEModule(out_channels) if use_attention else nn.Identity()

    def forward(
        self,
        feature_map: torch.Tensor,
        target_depth: int,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_depth, target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        use_attention: bool = True,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_batchnorm=use_batchnorm,
                use_attention=use_attention,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor],
                return_all: bool = False) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        x = features[0]
        skip_connections = features[1:]

        outs = []
        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            depth, height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, depth, height, width, skip_connection=skip_connection)
            if return_all:
                outs.append(x)

        return outs if return_all else x
