"""Small fully-convolutional 3-D flow network with reusable descriptors."""
import torch
from torch import nn
import torch.nn.functional as F

from biohub046.core.unet.decoder import UnetDecoder
from biohub046.core.unet.timm_3d_encoder import TimmUniversalEncoder


def local_correlation_flow(a, b, radius=(2, 3, 3), temperature=0.07):
    """Soft-argmax local matching on feature grids, output in grid voxels.

    For each location p in a, score b[p + delta]. Padding candidates are
    masked so borders cannot attract probability mass.
    """
    rz, ry, rx = (int(v) for v in radius)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    bp = F.pad(b, (rx, rx, ry, ry, rz, rz))
    valid = F.pad(torch.ones_like(b[:, :1]), (rx, rx, ry, ry, rz, rz))
    z, y, x = a.shape[-3:]
    scores, offsets = [], []
    for dz in range(-rz, rz + 1):
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                sl = (slice(rz + dz, rz + dz + z),
                      slice(ry + dy, ry + dy + y),
                      slice(rx + dx, rx + dx + x))
                candidate = bp[(...,) + sl]
                ok = valid[(...,) + sl]
                score = (a * candidate).sum(1) / temperature
                scores.append(score.masked_fill(ok[:, 0] == 0, -1e4))
                offsets.append((dz, dy, dx))
    prob = torch.softmax(torch.stack(scores, 1), 1)
    # Contract the candidate axis directly.  Materializing
    # ``prob[:, None] * delta`` creates a Bx3xKxDxHxW temporary (tens of GiB
    # for full-volume batch 16) although the requested output is only Bx3xD...
    delta = a.new_tensor(offsets).T
    return torch.einsum("bkzyx,ck->bczyx", prob, delta)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """Two-convolution residual block with projection when widths differ."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1))
        self.skip = nn.Identity() if cin == cout else nn.Conv3d(cin, cout, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class FlowNet3D(nn.Module):
    """Pair network returning dense a->b flow and per-frame descriptors."""
    def __init__(self, base_channels=16, feature_dim=16,
                 max_displacement_vox=(4.0, 6.0, 6.0),
                 correlation_radius=(2, 3, 3), correlation_temperature=0.07,
                 multiscale_pair_fusion=False, residual_blocks=False):
        super().__init__()
        c = base_channels
        block = ResidualBlock if residual_blocks else Block
        self.enc0 = block(1, c)
        self.enc1 = block(c, 2 * c)
        self.enc2 = block(2 * c, 4 * c)
        self.multiscale_pair_fusion = bool(multiscale_pair_fusion)
        if self.multiscale_pair_fusion:
            self.pair_fuse = nn.ModuleList([
                nn.Sequential(nn.Conv3d(3 * width, width, 1), nn.GELU())
                for width in (c, 2 * c, 4 * c)
            ])
        else:
            self.fuse = block(8 * c, 4 * c)
        self.dec1 = block(6 * c, 2 * c)
        self.dec0 = block(3 * c, c)
        self.flow_head = nn.Conv3d(c, 3, 3, padding=1)
        self.feature_head = nn.Conv3d(c, feature_dim, 1)
        self.correlation_radius = tuple(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.register_buffer("max_disp", torch.tensor(max_displacement_vox).view(1, 3, 1, 1, 1))
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def encode(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(F.avg_pool3d(f0, 2))
        f2 = self.enc2(F.avg_pool3d(f1, 2))
        return f0, f1, f2

    def forward(self, a, b, return_features=False):
        aa, bb = self.encode(a), self.encode(b)
        coarse = local_correlation_flow(aa[1], bb[1], self.correlation_radius,
                                        self.correlation_temperature)
        initial = F.interpolate(coarse, size=aa[0].shape[-3:], mode="trilinear",
                                align_corners=False) * 2.0
        if self.multiscale_pair_fusion:
            fused = [module(torch.cat((x, y, torch.abs(x - y)), 1))
                     for module, x, y in zip(self.pair_fuse, aa, bb)]
            x = fused[2]
            skip1, skip0 = fused[1], fused[0]
        else:
            x = self.fuse(torch.cat((aa[2], bb[2]), 1))
            skip1, skip0 = aa[1], aa[0]
        x = F.interpolate(x, size=aa[1].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec1(torch.cat((x, skip1), 1))
        x = F.interpolate(x, size=aa[0].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec0(torch.cat((x, skip0), 1))
        residual = torch.tanh(self.flow_head(x)) * self.max_disp
        flow = initial + residual
        if not return_features:
            return flow
        fa = F.normalize(self.feature_head(aa[0]), dim=1)
        fb = F.normalize(self.feature_head(bb[0]), dim=1)
        return {"flow": flow, "features_a": fa, "features_b": fb}


class DeepFlowNet3D(nn.Module):
    """FlowNet3D with an additional 1/8 encoder stage and 1/8->1/4 decode.

    The correlation prior remains at 1/2 resolution, matching FlowNet3D.  The
    learned residual path fuses both frames at 1/8 and uses the source-frame
    1/4 feature as the first decoder skip.
    """

    def __init__(self, base_channels=16, feature_dim=16,
                 max_displacement_vox=(4.0, 6.0, 6.0),
                 correlation_radius=(2, 3, 3), correlation_temperature=0.07):
        super().__init__()
        c = base_channels
        self.enc0 = Block(1, c)
        self.enc1 = Block(c, 2 * c)
        self.enc2 = Block(2 * c, 4 * c)
        self.enc3 = Block(4 * c, 8 * c)
        self.fuse = Block(16 * c, 8 * c)
        self.dec2 = Block(12 * c, 4 * c)
        self.dec1 = Block(6 * c, 2 * c)
        self.dec0 = Block(3 * c, c)
        self.flow_head = nn.Conv3d(c, 3, 3, padding=1)
        self.feature_head = nn.Conv3d(c, feature_dim, 1)
        self.correlation_radius = tuple(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.register_buffer(
            "max_disp",
            torch.tensor(max_displacement_vox).view(1, 3, 1, 1, 1))
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def encode(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(F.avg_pool3d(f0, 2))
        f2 = self.enc2(F.avg_pool3d(f1, 2))
        f3 = self.enc3(F.avg_pool3d(f2, 2))
        return f0, f1, f2, f3

    def forward(self, a, b, return_features=False):
        aa, bb = self.encode(a), self.encode(b)
        coarse = local_correlation_flow(
            aa[1], bb[1], self.correlation_radius,
            self.correlation_temperature)
        initial = F.interpolate(
            coarse, size=aa[0].shape[-3:], mode="trilinear",
            align_corners=False) * 2.0
        x = self.fuse(torch.cat((aa[3], bb[3]), 1))
        x = F.interpolate(
            x, size=aa[2].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec2(torch.cat((x, aa[2]), 1))
        x = F.interpolate(
            x, size=aa[1].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec1(torch.cat((x, aa[1]), 1))
        x = F.interpolate(
            x, size=aa[0].shape[-3:], mode="trilinear", align_corners=False)
        x = self.dec0(torch.cat((x, aa[0]), 1))
        flow = initial + torch.tanh(self.flow_head(x)) * self.max_disp
        if not return_features:
            return flow
        return {
            "flow": flow,
            "features_a": F.normalize(self.feature_head(aa[0]), dim=1),
            "features_b": F.normalize(self.feature_head(bb[0]), dim=1),
        }


class FlowNet25D(nn.Module):
    """Dense flow model using the production detector's 2.5-D encoder.

    A timm 2-D backbone is shared across z slices; the detector encoder then
    progressively average-pools depth and a 3-D U-Net decoder restores the
    full grid.  Only the deepest feature is pair-fused, while local
    correlation at the first encoder stage supplies an explicit motion prior.
    """

    def __init__(self, feature_dim=16, max_displacement_vox=(4.0, 6.0, 6.0),
                 correlation_radius=(2, 3, 3), correlation_temperature=0.07,
                 encoder_name="tf_efficientnetv2_s.in21k_ft_in1k",
                 encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 pretrained=True):
        super().__init__()
        self.encoder = TimmUniversalEncoder(
            name=encoder_name, in_channels=1, depth=int(encoder_depth),
            pretrained=bool(pretrained))
        channels = list(self.encoder.out_channels)
        self.deep_fuse = nn.Sequential(
            nn.Conv3d(2 * channels[-1], channels[-1], 1), nn.GELU())
        self.decoder = UnetDecoder(
            encoder_channels=channels, decoder_channels=tuple(decoder_channels),
            n_blocks=int(encoder_depth), use_batchnorm=True,
            use_attention=False, interpolation_mode="nearest")
        self.flow_head = nn.Conv3d(int(decoder_channels[-1]), 3, 3, padding=1)
        self.feature_head = nn.Conv3d(channels[1], int(feature_dim), 1)
        self.correlation_radius = tuple(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.register_buffer(
            "max_disp", torch.tensor(max_displacement_vox).view(1, 3, 1, 1, 1))
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def _descriptors(self, features, size):
        x = self.feature_head(features[1])
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return F.normalize(x, dim=1)

    def forward(self, a, b, return_features=False):
        # Batch-fold both frames exactly as detector/matcher siamese encoding.
        paired = torch.cat((a, b), dim=0)
        features = self.encoder(paired)
        n = a.shape[0]
        aa = [x[:n] for x in features]
        bb = [x[n:] for x in features]

        coarse = local_correlation_flow(
            aa[1], bb[1], self.correlation_radius,
            self.correlation_temperature)
        size = aa[0].shape[-3:]
        scale = coarse.new_tensor([
            size[i] / coarse.shape[-3 + i] for i in range(3)
        ]).view(1, 3, 1, 1, 1)
        initial = F.interpolate(
            coarse, size=size, mode="trilinear", align_corners=False) * scale

        fused = list(aa)
        fused[-1] = self.deep_fuse(torch.cat((aa[-1], bb[-1]), dim=1))
        decoded = self.decoder(fused)
        residual = torch.tanh(self.flow_head(decoded)) * self.max_disp
        flow = initial + residual
        if not return_features:
            return flow
        return {"flow": flow,
                "features_a": self._descriptors(aa, size),
                "features_b": self._descriptors(bb, size)}


class FlowNetMonaiEffNet(nn.Module):
    """Dense flow with MONAI's fully-3D EfficientNet feature pyramid."""

    def __init__(self, feature_dim=16, max_displacement_vox=(4.0, 6.0, 6.0),
                 correlation_radius=(2, 3, 3), correlation_temperature=0.07,
                 model_name="efficientnet-b0",
                 decoder_channels=(256, 128, 64, 32, 16),
                 multiscale_pair_fusion=False):
        super().__init__()
        # Lazy import keeps legacy flow inference independent of MONAI startup.
        from monai.networks.nets import EfficientNetBNFeatures
        self.encoder = EfficientNetBNFeatures(
            model_name=model_name, pretrained=False, spatial_dims=3,
            in_channels=1)
        # MONAI B0 emits strides 2/4/8/16/32 with these documented widths.
        channels = [1, 16, 24, 40, 112, 320]
        self.multiscale_pair_fusion = bool(multiscale_pair_fusion)
        if self.multiscale_pair_fusion:
            self.pair_fuse = nn.ModuleList([
                nn.Sequential(nn.Conv3d(3 * c, c, 1), nn.GELU())
                for c in channels
            ])
        else:
            # Retain the original module layout for old checkpoint loading.
            self.deep_fuse = nn.Sequential(
                nn.Conv3d(2 * channels[-1], channels[-1], 1), nn.GELU())
        self.decoder = UnetDecoder(
            encoder_channels=channels, decoder_channels=tuple(decoder_channels),
            n_blocks=5, use_batchnorm=True, use_attention=False,
            interpolation_mode="nearest")
        self.flow_head = nn.Conv3d(int(decoder_channels[-1]), 3, 3, padding=1)
        self.feature_head = nn.Conv3d(channels[1], int(feature_dim), 1)
        self.correlation_radius = tuple(correlation_radius)
        self.correlation_temperature = float(correlation_temperature)
        self.register_buffer(
            "max_disp", torch.tensor(max_displacement_vox).view(1, 3, 1, 1, 1))
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def _descriptors(self, features, size):
        x = self.feature_head(features[1])
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return F.normalize(x, dim=1)

    def forward(self, a, b, return_features=False):
        paired = torch.cat((a, b), dim=0)
        encoded = [paired, *self.encoder(paired)]
        n = a.shape[0]
        aa = [x[:n] for x in encoded]
        bb = [x[n:] for x in encoded]
        coarse = local_correlation_flow(
            aa[1], bb[1], self.correlation_radius,
            self.correlation_temperature)
        size = aa[0].shape[-3:]
        scale = coarse.new_tensor([
            size[i] / coarse.shape[-3 + i] for i in range(3)
        ]).view(1, 3, 1, 1, 1)
        initial = F.interpolate(
            coarse, size=size, mode="trilinear", align_corners=False) * scale
        if self.multiscale_pair_fusion:
            fused = [module(torch.cat((x, y, torch.abs(x - y)), dim=1))
                     for module, x, y in zip(self.pair_fuse, aa, bb)]
        else:
            fused = list(aa)
            fused[-1] = self.deep_fuse(torch.cat((aa[-1], bb[-1]), dim=1))
        decoded = self.decoder(fused)
        flow = initial + torch.tanh(self.flow_head(decoded)) * self.max_disp
        if not return_features:
            return flow
        return {"flow": flow,
                "features_a": self._descriptors(aa, size),
                "features_b": self._descriptors(bb, size)}


def build_flow_model(config):
    """Build old and new checkpoints from their serialized model config."""
    params = dict(config)
    architecture = str(params.pop("architecture", "3d")).lower()
    if architecture in {"2.5d", "2p5d", "25d"}:
        params.pop("base_channels", None)
        return FlowNet25D(**params)
    if architecture in {"monai_effnet", "monai_effnet_b0",
                        "monai_effnet_b0_multiscale"}:
        return FlowNetMonaiEffNet(
            feature_dim=params.get("feature_dim", 16),
            max_displacement_vox=params.get(
                "max_displacement_vox", (4.0, 6.0, 6.0)),
            correlation_radius=params.get("correlation_radius", (2, 3, 3)),
            correlation_temperature=params.get("correlation_temperature", 0.07),
            model_name=params.get("monai_model_name", "efficientnet-b0"),
            decoder_channels=params.get(
                "decoder_channels", (256, 128, 64, 32, 16)),
            multiscale_pair_fusion=(
                architecture == "monai_effnet_b0_multiscale" or
                params.get("multiscale_pair_fusion", False)))
    if architecture == "3d":
        return FlowNet3D(**params)
    if architecture == "3d_multiscale":
        params["multiscale_pair_fusion"] = True
        return FlowNet3D(**params)
    if architecture == "3d_residual":
        params["residual_blocks"] = True
        return FlowNet3D(**params)
    if architecture in {"3d_deep", "3d_1over8"}:
        return DeepFlowNet3D(**params)
    raise ValueError(f"unknown flow model architecture: {architecture}")
