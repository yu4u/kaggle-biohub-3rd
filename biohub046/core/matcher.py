"""Node-level cross-attention matcher (learned linking v2).

Trained END-TO-END with the siamese embedding model (model.matcher=xattn):
per-node features = embedding-map samples at champion detections + sinusoidal
positional encoding of centroid-centered µm coords. Blocks of self-attention
(within frame) + bidirectional cross-attention (t <-> t+1) let each node's
representation condition on the other frame — the relational reasoning that
per-node descriptors + cosine cannot express (descriptor line saturated at
argmax ~96.9% vs NN 97.8%, additive gain stuck at ~+0.004).

Pair scores are computed SPARSELY on pairs within the candidate radius
(12 µm): MLP([f_i, f_j, dpos/10]); everything else stays NEG_INF. A per-node
null head provides the "no link" logit (same InfoNCE + null training as the
descriptor line, minus same-frame negatives — self-attention already sees
intra-frame context).
"""
import torch
import torch.nn as nn

NEG_INF = -1e4


def sin_pos_enc(pos, dim_per_axis=16, max_um=128.0):
    """Sinusoidal encoding of (N, 3) µm coords -> (N, 3 * dim_per_axis).
    Wavelengths log-spaced from ~2 µm to max_um (candidate geometry scale)."""
    n_freq = dim_per_axis // 2
    freqs = torch.logspace(0, torch.log10(torch.tensor(max_um / 2.0)), n_freq,
                           device=pos.device) * torch.pi / max_um * 2
    ang = pos[:, :, None] * freqs[None, None, :]        # (N, 3, F)
    return torch.cat([ang.sin(), ang.cos()], dim=2).flatten(1)


class MatcherBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, a, b, m_aa=None, m_bb=None, m_ab=None, m_ba=None):
        # pre-LN residual; the same modules serve both frames (symmetric).
        # m_*: bool attn masks (True = disallowed), see NodeMatcher.forward
        for x_ref in ("a", "b"):
            x = a if x_ref == "a" else b
            m = m_aa if x_ref == "a" else m_bb
            h = self.n1(x)
            x = x + self.self_attn(h, h, h, need_weights=False, attn_mask=m)[0]
            if x_ref == "a":
                a = x
            else:
                b = x
        ha, hb = self.n2(a), self.n2(b)
        a = a + self.cross_attn(ha, hb, hb, need_weights=False, attn_mask=m_ab)[0]
        b = b + self.cross_attn(hb, ha, ha, need_weights=False, attn_mask=m_ba)[0]
        a = a + self.ffn(self.n3(a))
        b = b + self.ffn(self.n3(b))
        return a, b


class NodeMatcher(nn.Module):
    def __init__(self, feat_dim=32, d=128, heads=4, blocks=4, pe_axis=16,
                 cos_prior=False, drift_dpos=False, dpos_noise_um=0.0,
                 attn_window_um=0.0, align_pos_encoding=False, cls_head=False,
                 role_embed=False):
        # d: token width (model.matcher_d). dl_397 = 256 on the dl_369 recipe;
        # every checkpoint before that was trained at the default 128.
        super().__init__()
        self.proj = nn.Linear(feat_dim + 3 * pe_axis, d)
        # cls_head: per-node 3-way classification off the final cross-attended
        # features — classes follow dataset DIV_* (0 other / 1 daughter at its
        # birth frame / 2 division parent), supervised in BOTH frame contexts.
        # forward() stashes the logits on self (cls_logits_a/b) so its return
        # signature — consumed by every eval/export script — stays untouched.
        # NOTE: dl_199/dl_200 checkpoints carry role-split 2-way heads
        # (cls_head_a/b) and do not load here.
        self.cls_head = nn.Linear(d, 3) if cls_head else None
        self.cls_logits_a = self.cls_logits_b = None
        # role_embed: learned frame-role embeddings added to the projected
        # tokens (x_a += e[0] earlier frame, x_b += e[1] later frame). The
        # encoder is otherwise SYMMETRIC under frame exchange (shared
        # self/cross-attention), so node representations cannot sign time
        # direction — a division (2 partners in the LATER frame) collapses
        # onto merge-like detection patterns (2 partners in the EARLIER
        # frame, which real data only produces via detection noise). dl_199/
        # dl_200 showed a head-only fix cannot recover this: the collision
        # happens inside the trunk, before any readout. Zero-init so training
        # starts exactly at the symmetric model (cos_prior philosophy); roles
        # are consistent train/inference (frame a is ALWAYS the earlier one).
        self.role_embed = nn.Parameter(torch.zeros(2, d)) if role_embed \
            else None
        self.pe_axis = pe_axis
        self.blocks = nn.ModuleList(MatcherBlock(d, heads) for _ in range(blocks))
        self.norm = nn.LayerNorm(d)
        self.pair_mlp = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(),
                                      nn.Linear(d, 1))
        self.null_head = nn.Linear(d, 1)
        # cos_prior: score = logit_scale * cos(raw feats) + ZERO-INIT MLP
        # residual — starts exactly at the descriptor baseline (argmax ~96.9%)
        # and learns relational corrections (pure-matcher dl_063 trained
        # unstably from random init at lr 1e-3)
        self.cos_prior = bool(cos_prior)
        if self.cos_prior:
            import math
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            nn.init.zeros_(self.pair_mlp[-1].weight)
            nn.init.zeros_(self.pair_mlp[-1].bias)
        # drift_dpos: pair-MLP geometry input becomes the COLLECTIVE-DRIFT-
        # COMPENSATED residual (dpos - median NN displacement of the boundary)
        # — the physically correct frame ("residuals are small", not "raw
        # displacements are small"). Raw dpos made the matcher lose 167
        # NN-solvable large-drift links (dl_067 error decomposition).
        # dpos_noise_um: train-time Gaussian noise (µm) on the geometry input;
        # weakens the geometry shortcut so appearance must carry fine
        # discrimination.
        self.drift_dpos = bool(drift_dpos)
        self.dpos_noise_um = float(dpos_noise_um)
        # attn_window_um > 0: restrict self/cross attention to pairs within
        # this µm distance; 0 attends to all supplied nodes.
        # dl_369 uses crop_size == downsampled frame size (64^3), so training
        # does not spatially crop the frame. Candidate caps and GT fill can
        # still change the node set and hence the attention normalization.
        # A window bounds spatial context, but does not guarantee identical
        # train/inference neighborhood statistics.
        self.attn_window_um = float(attn_window_um)
        # align_pos_encoding: a drift-frame-trained model (matcher_drift_frame)
        # consistently aligns PE with candidate gating and pair geometry when a
        # candidate offset is given; old checkpoints were trained with
        # raw-coordinate PE, so drift-probe inference keeps this false.  A
        # construction-time property (not a forward kwarg) so call sites cannot
        # silently disagree with how the checkpoint was trained.
        self.align_pos_encoding = bool(align_pos_encoding)

    def forward(self, feat_a, pos_a, feat_b, pos_b, radius_um,
                candidate_offset_a=None):
        """feat_*: (N, C) raw embedding samples; pos_*: (N, 3) µm.
        candidate_offset_a optionally moves each frame-a node into the
        frame-b coordinate system before candidate gating / relative-geometry
        scoring.  It is an inference-safe, GT-free collective/local motion
        estimate; leaving it None exactly preserves the original raw-distance
        matcher.  NOTE: when an offset is given, the internal drift_dpos
        median-NN compensation is skipped (the offset already carries the
        collective drift), so probing a drift_dpos-trained checkpoint with an
        offset swaps its geometry-input estimator.

        Returns (logits_ab (Na, Nb) with NEG_INF outside radius,
                 null_a (Na,), null_b (Nb,))."""
        if candidate_offset_a is None:
            pos_a_match = pos_a
        else:
            pos_a_match = pos_a + candidate_offset_a
        pos_a_pe = pos_a_match if self.align_pos_encoding else pos_a
        center = torch.cat([pos_a_pe, pos_b]).mean(0, keepdim=True)
        # unit-normalize the raw embedding inputs: unbounded trunk feature
        # magnitudes fed fp16 attention internals -> NaN vals (dl_067 ep39;
        # the tanh output bound cannot catch QK^T overflow inside MHA)
        fa_in = nn.functional.normalize(feat_a.float(), dim=1)
        fb_in = nn.functional.normalize(feat_b.float(), dim=1)
        a = self.proj(torch.cat([fa_in, sin_pos_enc(pos_a_pe - center,
                                                    self.pe_axis)], dim=1))[None]
        b = self.proj(torch.cat([fb_in, sin_pos_enc(pos_b - center,
                                                    self.pe_axis)], dim=1))[None]
        if self.role_embed is not None:
            a = a + self.role_embed[0]
            b = b + self.role_embed[1]
        # candidate gating and the cross-frame attention window both live in
        # the (possibly drift-aligned) matching frame; within-frame masks stay
        # on raw coords (a collective offset cancels there anyway)
        d = torch.cdist(pos_a_match, pos_b)
        m_aa = m_bb = m_ab = m_ba = None
        if self.attn_window_um > 0:
            w = self.attn_window_um
            m_aa = torch.cdist(pos_a, pos_a) > w   # self rows never empty (d=0)
            m_bb = torch.cdist(pos_b, pos_b) > w
            m_ab = d > w
            m_ba = d.t() > w
            # a fully-masked row would NaN the softmax; nodes with no in-window
            # partner in the other frame fall back to full attention
            for m in (m_ab, m_ba):
                m[m.all(dim=1)] = False
        for blk in self.blocks:
            a, b = blk(a, b, m_aa, m_bb, m_ab, m_ba)
        a = self.norm(a)[0]
        b = self.norm(b)[0]
        if self.cls_head is not None:
            self.cls_logits_a = self.cls_head(a)
            self.cls_logits_b = self.cls_head(b)

        ii, jj = torch.nonzero(d <= radius_um, as_tuple=True)
        logits = torch.full_like(d, NEG_INF)
        if len(ii):
            dp = pos_b[jj] - pos_a_match[ii]
            if (self.drift_dpos and candidate_offset_a is None
                    and len(pos_a) >= 5):
                # per-boundary collective drift = median NN displacement
                # (first iteration of the production drift estimator; GT-free,
                # identical at train and inference)
                nn_j = d.argmin(dim=1)
                drift = (pos_b[nn_j] - pos_a).median(dim=0).values
                dp = dp - drift
            if self.training and self.dpos_noise_um > 0:
                dp = dp + torch.randn_like(dp) * self.dpos_noise_um
            dp = dp / 10.0
            s = self.pair_mlp(torch.cat([a[ii], b[jj], dp], dim=1))[:, 0]
            if self.cos_prior:
                s = s + self.logit_scale.exp().clamp(max=100.0) \
                    * (fa_in[ii] * fb_in[jj]).sum(1)
            # bound pair logits: unbounded MLP scores occasionally exploded
            # (dl_063 train-loss spikes of 250-450, val NaN under fp16)
            s = 30.0 * torch.tanh(s / 30.0)
            logits = logits.index_put((ii, jj), s)
        return logits, self.null_head(a)[:, 0], self.null_head(b)[:, 0]
