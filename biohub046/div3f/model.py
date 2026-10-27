"""3-frame pre-division classifier on the FlowNet3D backbone shape (random init)."""
import torch
from torch import nn
import torch.nn.functional as F

from biohub046.ssl_flow.model import Block
from biohub046.ssl_flow.ops import sample_volume


class Div3FNet(nn.Module):
    """Encode (t-1, t, t+1) with a shared FlowNet3D-style encoder, concat the
    three frames at every stage, decode once, output a 1-channel logit volume
    for "frame-t cell divides into two daughters at t+1"."""

    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc0 = Block(1, c)
        self.enc1 = Block(c, 2 * c)
        self.enc2 = Block(2 * c, 4 * c)
        self.fuse = Block(12 * c, 4 * c)
        self.dec1 = Block(10 * c, 2 * c)
        self.dec0 = Block(5 * c, c)
        self.head = nn.Conv3d(c, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(F.avg_pool3d(f0, 2))
        f2 = self.enc2(F.avg_pool3d(f1, 2))
        return f0, f1, f2

    def forward(self, x):
        """(B, 3, Z, Y, X) frames [t-1, t, t+1] -> (B, 1, Z, Y, X) logits.

        The batch fold is item-major, so reshape(b, 3*ch, ...) on the encoder
        output restores per-item [f_{t-1}, f_t, f_{t+1}] channel order (a
        chunk(3, dim=0) here would interleave items and frames)."""
        b = x.shape[0]
        f0, f1, f2 = self.encode(x.reshape(b * 3, 1, *x.shape[-3:]))
        f0, f1, f2 = (f.reshape(b, -1, *f.shape[-3:]) for f in (f0, f1, f2))
        y = self.fuse(f2)
        y = F.interpolate(y, size=f1.shape[-3:], mode="trilinear", align_corners=False)
        y = self.dec1(torch.cat((y, f1), 1))
        y = F.interpolate(y, size=f0.shape[-3:], mode="trilinear", align_corners=False)
        y = self.dec0(torch.cat((y, f0), 1))
        return self.head(y)


def _bn_to_gn(module, groups=8):
    """Replace every BatchNorm3d under `module` with GroupNorm(groups, C)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            _bn_to_gn(child, groups)


def decode_to_tap(decoder, features, out_tap):
    """UnetDecoder.forward truncated after block index out_tap (shared by
    Div3FNet25D and trackid.TrackIdNet)."""
    shapes = [f.shape[2:] for f in features][::-1]
    feats = features[1:][::-1]
    h, skips = feats[0], feats[1:]
    stop = out_tap % len(decoder.blocks) + 1
    for i, block in enumerate(decoder.blocks[:stop]):
        skip = skips[i] if i < len(skips) else None
        h = block(h, *shapes[i + 1], skip_connection=skip)
    return h


class Div3FNet25D(nn.Module):
    """3-frame variant on the production detector's 2.5-D timm encoder
    (tf_efficientnetv2_s, ImageNet-pretrained): batch-fold the frames through
    the shared encoder, concat the three frames at every stage, decode with
    the detector's UnetDecoder to a 1-channel logit volume."""

    def __init__(self, encoder_name="tf_efficientnetv2_s.in21k_ft_in1k",
                 encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16),
                 pretrained=True, z_conv=False, n_frames=3,
                 drop_path_rate=0.0, freeze_encoder=False,
                 decoder_dropout=0.0, decoder_norm="bn", out_tap=-1,
                 encoder_init_ckpt=None, encoder_img_size=None,
                 encoder_grad_ckpt=False, in_planes=1, stem_init="timm"):
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        # >1: per-frame input planes (image + det heatmap). stem_init "timm" =
        # every plane gets pretrained RGB filters (adapt_input_conv); "zero" =
        # plane 0 keeps the 1-ch stem, extra planes start at zero (_widen_stem)
        self.in_planes = int(in_planes)
        # div_39x "no over-engineering" (dl_352 analogue): out_tap=-2 stops the
        # decoder at the stride-2 block (the last block only mixes the stride-2
        # feature with the raw 1-ch image = no new information, and its 3x3x3
        # convs are the most expensive + spatially mixing stage); the 1x1 head
        # sits on that feature and the 1-ch logit map is upsampled trilinearly
        # to the input grid so point sampling / export stay unchanged
        self.out_tap = int(out_tap)
        from biohub046.core.unet.decoder import UnetDecoder
        from biohub046.core.unet.timm_3d_encoder import TimmUniversalEncoder
        self.n_frames = int(n_frames)
        # div_378: stochastic depth in the timm encoder (regularisation sweep)
        enc_kwargs = ({"drop_path_rate": float(drop_path_rate)}
                      if float(drop_path_rate) > 0 else {})
        # div_508: transformer encoders (swin) fix their token grid at
        # creation; pass the YX size of the ds grid (128 at ds[1,2,2])
        if encoder_img_size:
            enc_kwargs["img_size"] = int(encoder_img_size)
        self.encoder = TimmUniversalEncoder(
            name=encoder_name, in_channels=1, in_planes=self.in_planes,
            stem_init=str(stem_init),
            depth=int(encoder_depth),
            pretrained=bool(pretrained), grad_ckpt=bool(encoder_grad_ckpt),
            **enc_kwargs)
        # zero-init residual 3x1x1 conv per decoder-consumed stage (div_315):
        # explicit z mixing on the concatenated features before the decoder;
        # zero-init keeps the pretrained forward identical at init
        self.z_convs = None
        if z_conv:
            self.z_convs = nn.ModuleList()
            for c in self.encoder.out_channels[1:]:
                conv = nn.Conv3d(self.n_frames * c, self.n_frames * c,
                                 (3, 1, 1), padding=(1, 0, 0))
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
                self.z_convs.append(conv)
        self.decoder = UnetDecoder(
            encoder_channels=[self.n_frames * c
                              for c in self.encoder.out_channels],
            decoder_channels=tuple(decoder_channels),
            n_blocks=int(encoder_depth), use_batchnorm=True,
            use_attention=False, interpolation_mode="nearest")
        # div_39x decoder regularisation (the post-peak collapse lives in the
        # decoder+head, div_393): channel dropout on the decoder's inputs
        # (3-frame skip features) and on its output; BN -> GroupNorm(8)
        self.feat_drop = (nn.Dropout3d(float(decoder_dropout))
                          if float(decoder_dropout) > 0 else None)
        if str(decoder_norm) == "gn":
            _bn_to_gn(self.decoder)
        self.head = nn.Conv3d(int(decoder_channels[self.out_tap]), 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # the det ckpt's 1-plane stem cannot load into a widened one: the
        # stem would silently stay ImageNet (load_det_encoder skips it)
        assert not (encoder_init_ckpt and self.in_planes > 1), (
            "model.encoder_init_ckpt with in_planes > 1 is not supported")
        if encoder_init_ckpt:
            self.load_det_encoder(str(encoder_init_ckpt))
        if self.freeze_encoder:
            # div_39x: ImageNet encoder as a fixed feature extractor — no
            # grads and BN statistics frozen (see train() below); only the
            # decoder + head (13.1M) learn
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def load_det_encoder(self, path):
        """div_423: replace the ImageNet encoder weights with those of a
        stage-1 2.5-D detector Lightning ckpt (dl_079: the same
        TimmUniversalEncoder, `model.encoder.*`, EMA weights at
        `model_ema.module.encoder.*` when present). The decoder (3-frame skip
        widths differ from the detector's) and the head stay as built.
        Use the fold-matched ckpt."""
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        pre = ("model_ema.module.encoder." if any(k.startswith("model_ema.") for k in sd)
               else "model.encoder.")
        det = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        own = self.encoder.state_dict()
        load = {k: v for k, v in det.items() if k in own and own[k].shape == v.shape}
        self.encoder.load_state_dict(load, strict=False)
        skipped = sorted(k for k in own if k not in load)
        print(f"det encoder init from {path} ({pre.rstrip('.')}): "
              f"{len(load)}/{len(own)} tensors loaded"
              + (f", skipped {skipped}" if skipped else ""), flush=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        """(B, n_frames[, in_planes], Z, Y, X) -> (B, 1, Z, Y, X) logits.
        Same item-major fold/unfold as Div3FNet.forward."""
        b, n = x.shape[0], x.shape[1]
        feats = self.encode(x.reshape(b * n, self.in_planes, *x.shape[-3:]))
        return self.decode([f.reshape(b, -1, *f.shape[-3:]) for f in feats], x.shape[-3:])

    def encode(self, frames):
        """(N, in_planes, Z, Y, X) single frames -> per-stage features. Frames do
        not interact here, so a frame's features can be reused across windows."""
        return self.encoder(frames)

    def decode(self, feats, size):
        """Per-stage features with the n_frames concatenated item-major along
        channels ([f_{t-1}, f_t, f_{t+1}]) -> (B, 1, *size) logits."""
        feats = list(feats)
        if self.z_convs is not None:
            feats[1:] = [f + zc(f) for f, zc in zip(feats[1:], self.z_convs)]
        if self.feat_drop is not None:
            feats[1:] = [self.feat_drop(f) for f in feats[1:]]
        if self.out_tap != -1:
            h = self._decode_to_tap(feats)
            if self.feat_drop is not None:
                h = self.feat_drop(h)
            return F.interpolate(self.head(h), size=size,
                                 mode="trilinear", align_corners=False)
        if self.feat_drop is not None:
            return self.head(self.feat_drop(self.decoder(feats)))
        return self.head(self.decoder(feats))

    def _decode_to_tap(self, features):
        return decode_to_tap(self.decoder, features, self.out_tap)


class Div3FSegResNet(nn.Module):
    """Fully-3D MONAI SegResNet (random init) in the dl_275 detector
    configuration (GN8 + LeakyReLU, blocks_down (1,1,1,1), init 32) for the
    3-frame task. fusion="early": the n_frames frames are the input channels
    of one net. fusion="mid": frames are batch-folded through the shared
    encoder and the per-stage features are concatenated (item-major, as in
    Div3FNet.forward) and reduced back to the stage width by a 1x1x1 conv
    before the additive-skip decoder."""

    def __init__(self, init_filters=32, blocks_down=(1, 1, 1, 1),
                 blocks_up=(1, 1, 1), fusion="early", n_frames=3,
                 grad_ckpt=False, init_ckpt=None, fuse_init="random"):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.n_frames, self.fusion = int(n_frames), str(fusion)
        # recompute each ResBlock stage in backward instead of storing its
        # activations (full-frame ds [1,2,2] (64,128,128) inputs: 18.3 GB@bs8 without);
        # the forward/gradients are unchanged, only train-time memory/speed
        self.grad_ckpt = bool(grad_ckpt)
        assert self.fusion in ("early", "mid"), self.fusion
        self.net = SegResNet(
            spatial_dims=3,
            in_channels=self.n_frames if self.fusion == "early" else 1,
            out_channels=1, init_filters=int(init_filters),
            blocks_down=tuple(int(v) for v in blocks_down),
            blocks_up=tuple(int(v) for v in blocks_up),
            norm=("GROUP", {"num_groups": 8}),
            act=("LEAKYRELU", {"negative_slope": 0.01}))
        if self.fusion == "mid":
            self.fuse = nn.ModuleList(
                nn.Conv3d(self.n_frames * c, c, 1)
                for c in (int(init_filters) * 2 ** i
                          for i in range(len(blocks_down))))
        # MONAI wraps the final 1x1x1 conv in a Convolution(Sequential)
        head = [m for m in self.net.conv_final.modules()
                if isinstance(m, nn.Conv3d)][-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        if init_ckpt:
            self.load_det_weights(str(init_ckpt))
        if self.fusion == "mid" and str(fuse_init) == "center":
            # pass the centre frame's features through unchanged at init
            # (identity on frame t, zero on t+-1): with a det-pretrained
            # trunk the initial decoder input is exactly the det trunk on
            # frame t; the t+-1 planes open up through training
            mid = self.n_frames // 2
            for conv in self.fuse:
                c = conv.out_channels
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
                with torch.no_grad():
                    conv.weight[:, mid * c:(mid + 1) * c, 0, 0, 0] = torch.eye(c)

    def load_det_weights(self, path):
        """Initialise the SegResNet trunk from a stage-1 detector Lightning
        ckpt (dl_275: same MONAI config, in_channels=1, EMA weights at
        `model_ema.module.*`). The final 1x1x1 conv (heatmap head) is left
        zero-initialised; tensors whose shape differs (early fusion's 3-ch
        convInit) are skipped and reported."""
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        pre = ("model_ema.module." if any(k.startswith("model_ema.") for k in sd)
               else "model.")
        det = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        own = self.net.state_dict()
        head = {k for k in own if k.startswith("conv_final.2.")}
        load = {k: v for k, v in det.items()
                if k in own and k not in head and own[k].shape == v.shape}
        self.net.load_state_dict(load, strict=False)
        skipped = sorted(k for k in own if k not in load and k not in head)
        print(f"det init from {path} ({pre.rstrip('.')}): "
              f"{len(load)}/{len(own) - len(head)} trunk tensors loaded"
              + (f", skipped {skipped}" if skipped else ""), flush=True)

    def _layer(self, layer, x):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def _encode(self, x):
        """= SegResNet.encode (no dropout) with optional checkpointing."""
        x, down_x = self.net.convInit(x), []
        for down in self.net.down_layers:
            x = self._layer(down, x)
            down_x.append(x)
        return down_x

    def _decode(self, down_x):
        """= SegResNet.decode on the reversed stage list."""
        x = down_x[0]
        for i, (up, upl) in enumerate(zip(self.net.up_samples,
                                          self.net.up_layers)):
            x = self._layer(upl, up(x) + down_x[i + 1])
        return self.net.conv_final(x)

    def forward(self, x):
        """(B, n_frames, Z, Y, X) -> (B, 1, Z, Y, X) logits. Z, Y, X must be
        multiples of 2 ** (len(blocks_down) - 1) (isotropic stride-2 stages,
        interpolation upsampling)."""
        if self.fusion == "early":
            down_x = self._encode(x)
        else:
            b, n = x.shape[0], x.shape[1]
            down_x = self._encode(x.reshape(b * n, 1, *x.shape[-3:]))
            down_x = [fuse(f.reshape(b, -1, *f.shape[-3:]))
                      for f, fuse in zip(down_x, self.fuse)]
        down_x.reverse()
        return self._decode(down_x)


def build_model(cfg_model):
    in_planes = int(cfg_model.get("in_planes", 1))
    assert in_planes == 1 or str(cfg_model.get("arch", "3d")) == "25d", (
        "model.in_planes > 1 (det heatmap input) is only wired for arch=25d")
    if str(cfg_model.get("arch", "3d")) == "segresnet":
        return Div3FSegResNet(
            init_filters=int(cfg_model.get("segres_init_filters", 32)),
            blocks_down=tuple(cfg_model.get("segres_blocks_down", (1, 1, 1, 1))),
            blocks_up=tuple(cfg_model.get("segres_blocks_up", (1, 1, 1))),
            fusion=str(cfg_model.get("segres_fusion", "early")),
            n_frames=int(cfg_model.get("n_frames", 3)),
            grad_ckpt=bool(cfg_model.get("grad_ckpt", False)),
            init_ckpt=cfg_model.get("segres_init_ckpt"),
            fuse_init=str(cfg_model.get("segres_fuse_init", "random")))
    if str(cfg_model.get("arch", "3d")) == "25d":
        return Div3FNet25D(
            encoder_name=cfg_model.get(
                "encoder_name", "tf_efficientnetv2_s.in21k_ft_in1k"),
            decoder_channels=tuple(cfg_model.get(
                "decoder_channels", (256, 128, 64, 32, 16))),
            pretrained=bool(cfg_model.get("pretrained", True)),
            z_conv=bool(cfg_model.get("z_conv", False)),
            n_frames=int(cfg_model.get("n_frames", 3)),
            drop_path_rate=float(cfg_model.get("drop_path_rate", 0.0)),
            freeze_encoder=bool(cfg_model.get("freeze_encoder", False)),
            decoder_dropout=float(cfg_model.get("decoder_dropout", 0.0)),
            decoder_norm=str(cfg_model.get("decoder_norm", "bn")),
            out_tap=int(cfg_model.get("out_tap", -1)),
            encoder_init_ckpt=cfg_model.get("encoder_init_ckpt"),
            encoder_img_size=cfg_model.get("encoder_img_size"),
            encoder_grad_ckpt=bool(cfg_model.get("encoder_grad_ckpt", False)),
            in_planes=in_planes,
            stem_init=str(cfg_model.get("stem_init", "timm")))
    return Div3FNet(base_channels=int(cfg_model["base_channels"]))


def sample_logits(logits, points_zyx):
    """Trilinearly sample a (1, 1, Z, Y, X) logit volume at (N, 3) zyx grid
    coords -> (N,) logits. Detection coords are fractional after to_ds, so
    nearest-index sampling would be off by up to ~2 um."""
    return sample_volume(logits, points_zyx)[0, :, 0]
