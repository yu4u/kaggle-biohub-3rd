"""Siamese linking model (model.temporal=siamese).

A plain single-frame 2.5D Unet plus a 32-dim embedding head tapped from the
stride-2 decoder stage (decoder.blocks[-2] output, captured via a forward
hook — decoder.py stays untouched). Each frame of a pair is encoded
INDEPENDENTLY with shared weights (the caller batch-folds the pair), so
embeddings are comparable across arbitrary time gaps (gap-closing reuse).

forward(x) -> (heatmap (B,1,Z,Y,X), embedding (B,C,z',y',x')). The embedding
map lives on the stride-2 grid; with embed_only_forward=True (consumers that
never read the heatmap: embed_only training / eval on a det cache) the decoder
stops at the tapped block and the heatmap is None — the unused stride-1 block
+ seg head were 65% of the forward (dl_369 measurement). sample it at full-res peak coords with
embed_loss.sample_embeddings (trilinear). L2 normalization happens in the
loss, not here.

embed_logit_scale / embed_null are the InfoNCE temperature (CLIP-style
learnable log scale) and the learnable "no link" logit; they live on the
model so the standard optimizer setup picks them up.
"""
import math

import torch
import torch.nn as nn

from .unet import Unet


class EmbedUnet(Unet):
    def __init__(self, embed_dim: int = 32, embed_head_arch: str = "linear",
                 freeze_trunk: bool = False, matcher: str = "none",
                 matcher_drift_dpos: bool = False,
                 matcher_dpos_noise_um: float = 0.0,
                 matcher_attn_window_um: float = 0.0,
                 matcher_drift_frame: bool = False,
                 matcher_cls_head: bool = False,
                 matcher_role_embed: bool = False,
                 matcher_d: int = 128,
                 embed_tap: int = -2, **kwargs):
        super().__init__(**kwargs)
        # dl_369: embed_head_arch="none" drops the head entirely — the
        # embedding IS the tapped decoder feature, so its dimension follows
        # the decoder channel count at the tap (embed_dim is ignored)
        c = kwargs["decoder_channels"][int(embed_tap)]
        if embed_head_arch == "none":
            embed_dim = c
        self.matcher = None
        self.matcher_drift_frame = bool(matcher_drift_frame)
        if matcher in ("xattn", "xattn_cos"):
            # node-level cross-attention matcher, trained end-to-end with the
            # embedding trunk (core/matcher.py); xattn_cos adds the
            # cosine-prior + zero-init residual scoring
            from ..matcher import NodeMatcher
            self.matcher = NodeMatcher(feat_dim=embed_dim, d=int(matcher_d),
                                       cos_prior=(matcher == "xattn_cos"),
                                       drift_dpos=matcher_drift_dpos,
                                       dpos_noise_um=matcher_dpos_noise_um,
                                       attn_window_um=matcher_attn_window_um,
                                       align_pos_encoding=matcher_drift_frame,
                                       cls_head=matcher_cls_head,
                                       role_embed=matcher_role_embed)
        elif matcher != "none":
            raise ValueError(f"unknown matcher: {matcher}")
        if self.matcher_drift_frame and self.matcher is None:
            # the cosine path has no drift-frame inference counterpart: with
            # matcher=none the flag would silently shift the train-time radius
            # gates (pos_far/far_ab) into a frame eval never uses — drop it
            # (the flag defaults ON, so matcher-less siamese runs must not die)
            print("matcher_drift_frame ignored: model.matcher is none")
            self.matcher_drift_frame = False
        # embed_tap: which decoder block feeds the embedding head. -2 =
        # stride-2 stage (default, 0.8 µm y/x grid at full res); -1 = finest
        # stride-1 stage (e.g. the 1.625 µm iso grid under downsample [1,4,4])
        self.embed_tap = int(embed_tap)
        if embed_head_arch == "none":
            self.embed_head = nn.Identity()
        elif embed_head_arch == "linear":
            self.embed_head = nn.Conv3d(c, embed_dim, kernel_size=1)
        elif embed_head_arch == "conv3":
            # deeper head for the frozen-trunk mode: the trunk features were
            # trained for detection only, so the head needs mixing capacity
            self.embed_head = nn.Sequential(
                nn.Conv3d(c, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(64, embed_dim, kernel_size=1),
            )
        else:
            raise ValueError(f"unknown embed_head_arch: {embed_head_arch}")
        self.embed_logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.embed_null = nn.Parameter(torch.tensor(0.0))
        self.freeze_trunk = bool(freeze_trunk)
        if self.freeze_trunk:
            # embedding-only training on top of a frozen detector (e.g. the
            # dl_028 champion): no grads into the trunk, BN stats frozen too
            for name, p in self.named_parameters():
                if not (name.startswith("embed_") or name.startswith("matcher.")):
                    p.requires_grad_(False)
        self._embed_feat = None
        self.decoder.blocks[self.embed_tap].register_forward_hook(self._capture)
        # set by consumers that never read the heatmap (see module docstring)
        self.embed_only_forward = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_trunk and mode:
            self.encoder.eval()
            self.decoder.eval()
            self.segmentation_head.eval()
        return self

    def _capture(self, module, inputs, output):
        self._embed_feat = output

    def forward(self, x: torch.Tensor):
        # re-assert the frozen trunk's eval mode HERE: Lightning's eval loop
        # restores training flags by direct attribute assignment
        # (_ModuleMode.restore), bypassing the train() override — without this
        # the trunk BN stats silently update all through training (dl_059 bug:
        # 238/238 BN buffers drifted, lambda=0 0.8818 vs dl_028's 0.8847)
        if self.freeze_trunk and self.encoder.training:
            self.encoder.eval()
            self.decoder.eval()
            self.segmentation_head.eval()
        if self.embed_only_forward:
            emb = self.embed_head(self._forward_to_tap(x))
            self._embed_feat = None
            return None, emb
        hm = super().forward(x)
        emb = self.embed_head(self._embed_feat)
        self._embed_feat = None
        return hm, emb

    def _forward_to_tap(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder + decoder blocks up to and including the tapped one (same
        loop as UnetDecoder.forward, truncated); returns the tap feature."""
        features = self.encoder(x)
        if self.use_ppm and self.ppm is not None:
            features[-1] = self.ppm(features[-1])
        shapes = [f.shape[2:] for f in features][::-1]
        feats = features[1:][::-1]
        h, skips = feats[0], feats[1:]
        n_blocks = len(self.decoder.blocks)
        stop = self.embed_tap % n_blocks + 1
        for i, block in enumerate(self.decoder.blocks[:stop]):
            skip = skips[i] if i < len(skips) else None
            h = block(h, *shapes[i + 1], skip_connection=skip)
        return h
