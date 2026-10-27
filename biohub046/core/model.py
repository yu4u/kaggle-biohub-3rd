"""Heatmap detector: 2.5D UNet from kaggle-vesuvius (core/unet).

Input (B, 1, Z, Y, X) float -> heatmap logits (B, 1, Z, Y, X). The timm 2D
encoder runs per z-slice with depth avg-pooling; the 3D conv decoder fuses
across z. Every spatial dim must be divisible by 2**encoder_depth.
"""
import torch

from .unet import Unet


class _AmpFp32(torch.nn.Module):
    """model.amp_fp32_modules wrapper: run the wrapped module OUTSIDE autocast
    in fp32 (fp16 activation-overflow guard, kaggle-rogii dl_xunet pattern).
    Transparent under full fp32 (autocast off + .float() is a no-op). NOTE:
    wrapping inserts a `.mod` level into the wrapped paths' state-dict keys,
    so checkpoints trained with the lever must be rebuilt with the same cfg
    (restore_cfg restores model.* from the ckpt, keeping keys consistent)."""

    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, *args):
        with torch.autocast(args[0].device.type, enabled=False):
            return self.mod(*(a.float() if torch.is_tensor(a) else a
                              for a in args))


def _finalize(model, m, resume_path):
    """Apply the selective-precision guard (no RNG, no init change;
    [] = byte-identical), then load weights via _load_resume."""
    for _path in list(m.get("amp_fp32_modules", []) or []):
        parts = str(_path).split(".")
        parent = model
        for q in parts[:-1]:
            parent = parent[int(q)] if q.isdigit() else getattr(parent, q)
        leaf = parts[-1]
        child = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        wrapped = _AmpFp32(child)
        if leaf.isdigit():
            parent[int(leaf)] = wrapped
        else:
            setattr(parent, leaf, wrapped)
    return _load_resume(model, resume_path)


def get_model_from_cfg(cfg, resume_path=None):
    m = cfg.model
    # the pair/siamese datasets build their own two-frame image and never
    # call dataset.to_model_input, so the flag would train a plain 1-plane
    # model while the config claims otherwise -- refuse instead
    assert not cfg.data.get("local_contrast", False) \
        or str(m.get("temporal", "none")) == "none", (
            "data.local_contrast is only wired for the single-frame detector, "
            "not model.temporal=" + str(m.get("temporal", "none")))
    if str(m.get("temporal", "none")) == "siamese" \
            and str(m.get("arch", "unet25d")) == "segresnet":
        # fully-3D SegResNet trunk (dl_275 config), optionally det-pretrained;
        # the init ckpt is only read when no resume_path supplies the weights
        from .unet.embed_segres import EmbedSegResNet
        model = EmbedSegResNet(
            init_filters=int(m.get("segres_init_filters", 32)),
            blocks_down=tuple(int(v) for v in m.get("segres_blocks_down", (1, 1, 1, 1))),
            blocks_up=tuple(int(v) for v in m.get("segres_blocks_up", (1, 1, 1))),
            in_channels=int(m.in_channels),
            init_ckpt=None if resume_path else m.get("segres_init_ckpt"),
            embed_tap=int(m.get("embed_tap", -2)),
            norm=str(m.get("segres_norm", "group")),
            matcher=str(m.get("matcher", "none")),
            matcher_drift_dpos=bool(m.get("matcher_drift_dpos", False)),
            matcher_dpos_noise_um=float(m.get("matcher_dpos_noise_um", 0.0)),
            matcher_attn_window_um=float(m.get("matcher_attn_window_um", 0.0)),
            matcher_drift_frame=bool(m.get("matcher_drift_frame", False)),
            matcher_cls_head=bool(m.get("matcher_cls_head", False)),
            matcher_role_embed=bool(m.get("matcher_role_embed", False)),
            matcher_d=int(m.get("matcher_d", 128)),
        )
        return _finalize(model, m, resume_path)
    if str(m.get("temporal", "none")) == "siamese":
        from .unet.embed_unet import EmbedUnet
        model = EmbedUnet(
            embed_dim=int(m.get("embed_dim", 32)),
            embed_head_arch=str(m.get("embed_head_arch", "linear")),
            freeze_trunk=bool(m.get("freeze_trunk", False)),
            matcher=str(m.get("matcher", "none")),
            matcher_drift_dpos=bool(m.get("matcher_drift_dpos", False)),
            matcher_dpos_noise_um=float(m.get("matcher_dpos_noise_um", 0.0)),
            matcher_attn_window_um=float(m.get("matcher_attn_window_um", 0.0)),
            matcher_drift_frame=bool(m.get("matcher_drift_frame", False)),
            matcher_cls_head=bool(m.get("matcher_cls_head", False)),
            matcher_role_embed=bool(m.get("matcher_role_embed", False)),
            matcher_d=int(m.get("matcher_d", 128)),
            embed_tap=int(m.get("embed_tap", -2)),
            encoder_name=m.encoder_name,
            encoder_depth=int(m.encoder_depth),
            decoder_channels=tuple(m.decoder_channels),
            in_channels=int(m.in_channels),
            slicer_trainable=bool(m.get("slicer_trainable", False)),
            classes=1,
            pretrained=bool(m.pretrained),
            iso_z=bool(m.get("iso_z", False)),
            iso_z_zmix=bool(m.get("iso_z_zmix", True)),
        )
        return _finalize(model, m, resume_path)
    if str(m.get("temporal", "none")) == "pair":
        from .unet.pair_unet import PairUnet
        assert int(m.in_channels) == 1, "pair model encodes each frame with 1 channel"
        model = PairUnet(
            encoder_name=m.encoder_name,
            encoder_depth=int(m.encoder_depth),
            decoder_channels=tuple(m.decoder_channels),
            classes=4,  # [hm_t, flow dz, dy, dx] -- frame-t predictions only
            pretrained=bool(m.pretrained),
            xattn_stages=tuple(m.get("xattn_stages", (3, 2))),
            iso_z=bool(m.get("iso_z", False)),
            iso_z_zmix=bool(m.get("iso_z_zmix", True)),
        )
        return _finalize(model, m, resume_path)
    if str(m.get("arch", "unet25d")) == "segresnet":
        from monai.networks.nets import SegResNet
        # additive-skip residual UNet with per-stage block counts and
        # interpolation upsampling. NOTE: downsampling is isotropic stride-2
        # (no aniso first stride) — z is pooled at the same rate as y/x.
        # MONAI requires len(blocks_up) == len(blocks_down) - 1.
        model = SegResNet(
            spatial_dims=3,
            in_channels=int(m.in_channels),
            out_channels=1,
            init_filters=int(m.get("segres_init_filters", 32)),
            blocks_down=tuple(int(v) for v in m.get("segres_blocks_down", (1, 1, 1, 1))),
            blocks_up=tuple(int(v) for v in m.get("segres_blocks_up", (1, 1, 1))),
            norm=("GROUP", {"num_groups": 8}),
            act=("LEAKYRELU", {"negative_slope": 0.01}),
        )
        return _finalize(model, m, resume_path)
    if str(m.get("arch", "unet25d")) == "flowdet3d":
        from .unet.flowdet import FlowDetNet
        model = FlowDetNet(base_channels=int(m.get("flow_base_channels", 32)),
                           flow_ckpt=m.get("flow_backbone_ckpt"),
                           norm_groups=int(m.get("flow_norm_groups", 0)))
        return _finalize(model, m, resume_path)
    if str(m.get("arch", "unet25d")) == "unet25d_coarse":
        # [1,4,4] coarse-grid head: cls + offset on the stride-4 decoder
        # feature (z full via iso_z). Only 21_test / eval_det_tradeoff /
        # detect_sample importers can consume these ckpts (4-channel output).
        assert str(m.get("temporal", "none")) == "none", \
            "arch=unet25d_coarse requires model.temporal=none"
        assert not cfg.data.get("local_contrast", False), \
            "arch=unet25d_coarse does not support local_contrast"
        # this branch returns before the freeze_trunk/div_head tail below;
        # refuse those flags instead of silently ignoring them (freeze_trunk
        # would freeze nothing, then crash in on_train_epoch_start on
        # model.div_head.train() with div_head=None)
        assert not m.get("freeze_trunk", False), \
            "arch=unet25d_coarse does not support model.freeze_trunk"
        assert not m.get("div_head", False), \
            "arch=unet25d_coarse does not support model.div_head"
        model = Unet(
            encoder_name=m.encoder_name,
            encoder_depth=int(m.encoder_depth),
            decoder_channels=tuple(m.decoder_channels),
            in_channels=int(m.in_channels),
            in_planes=1,
            classes=1,
            pretrained=bool(m.pretrained),
            iso_z=True,                      # load-bearing: full-z stride-4 skip
            iso_z_zmix=bool(m.get("iso_z_zmix", True)),
            coarse_head=True,
        )
        if str(m.get("coarse_target", "binary")) == "gauss":
            # MSE regression head: start near 0 output, not the BCE prior
            torch.nn.init.zeros_(model.cls_head.bias)
        return _finalize(model, m, resume_path)
    model = Unet(
        encoder_name=m.encoder_name,
        encoder_depth=int(m.encoder_depth),
        decoder_channels=tuple(m.decoder_channels),
        in_channels=int(m.in_channels),
        # derived, not configured: the dataset decides how many planes it
        # emits (dataset.to_model_input), so a separate model key could desync
        in_planes=2 if cfg.data.get("local_contrast", False) else 1,
        classes=1,
        pretrained=bool(m.pretrained),
        iso_z=bool(m.get("iso_z", False)),
        iso_z_zmix=bool(m.get("iso_z_zmix", True)),
        div_head=bool(m.get("div_head", False)),
        deep_supervision=int(m.get("deep_supervision", 0)),
        grad_ckpt=bool(m.get("encoder_grad_ckpt", False)),
        # dl_543: timm models whose attention is built for a fixed 2D size
        # (coatnet / maxvit rel-pos MLP) take it here; the 2.5D slices must
        # then have exactly this yx size at train AND inference
        img_size=(int(m.encoder_img_size) if m.get("encoder_img_size") else None),
        # dl_805: timm stochastic depth; None is dropped by Unet (timm default)
        drop_path_rate=(float(m.encoder_drop_path_rate)
                        if m.get("encoder_drop_path_rate") is not None else None),
    )
    model = _finalize(model, m, resume_path)
    if bool(m.get("freeze_trunk", False)):
        # dl_195: head-only training on a frozen production detector. The det
        # channel stays identical to the resume ckpt (BN running stats are
        # frozen too — pl_module keeps the trunk in eval mode during training)
        assert model.div_head is not None, "freeze_trunk without model.div_head"
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("div_head.")
    return model


def _resume_state(resume_path):
    if not resume_path:
        return None
    ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # prefer EMA weights when present (ckpt trained with model.ema=true);
    # mini val during training evaluates the EMA model, so test must match
    ema = {k.removeprefix("model_ema.module."): v for k, v in state.items()
           if k.startswith("model_ema.module.")}
    if ema:
        return ema
    return {k.removeprefix("model."): v for k, v in state.items()
            if not k.startswith("model_ema.")}


def _load_resume(model, resume_path):
    state = _resume_state(resume_path)
    if state is not None:
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # trunk-only checkpoint into an EmbedUnet (freeze_trunk init from
            # a plain detector, e.g. dl_028): fill what matches, keep the
            # embedding head at its random init
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
            assert not unexpected, f"unexpected checkpoint keys: {unexpected[:5]}"
            # only the embedding head / matcher / div head may be absent from
            # the ckpt; any other missing key means a broken/mismatched ckpt
            bad = [k for k in missing
                   if not k.startswith(("embed_", "matcher.", "div_head."))]
            assert not bad, f"missing non-embed checkpoint keys: {bad[:5]}"
        print(f"loaded weights from {resume_path}")
    return model
