import torch
import torch.nn as nn

from .dataset import MASK_THRESH, T15


def get_loss(cfg):
    if str(cfg.model.get("arch", "unet25d")) == "unet25d_coarse":
        assert str(cfg.model.get("temporal", "none")) == "none", \
            "arch=unet25d_coarse requires model.temporal=none"
        if str(cfg.model.get("coarse_target", "binary")) == "gauss":
            return CoarseGaussLoss(cfg)
        return CoarseLoss(cfg)
    if str(cfg.model.get("temporal", "none")) == "pair":
        return PairLoss(cfg)
    # siamese: detection loss is plain MyLoss with the frame pair folded into
    # the batch dim; the embedding InfoNCE lives in core/embed_loss.py
    return MyLoss(cfg)


class MyLoss(nn.Module):
    """Masked heatmap regression with separate positive/background normalization.

    The mask (from the dataset; see core/dataset.py docstring) is a float WEIGHT
    map that down-weights suspected unlabeled cells (0 = excluded, the original
    hard carve; loss.carve_weight raises it). The positive region (Gaussian
    support, target >= MASK_THRESH) and the background are averaged SEPARATELY,
    each as a weighted mean:
        loss = pos_term + bg_weight * bg_term
    so the Gaussian-bump gradient does not get diluted when the supervised
    background is ~80% of the crop (carved mode). pos_weight > 0 additionally
    weights positive voxels by (1 + pos_weight * target).
    """

    def __init__(self, cfg):
        super().__init__()
        self.bg_weight = float(cfg.loss.bg_weight)
        self.pos_weight = float(cfg.loss.pos_weight)
        self.pooled = bool(cfg.loss.get("pooled", False))
        self.pseudo_weight = float(cfg.loss.get("pseudo_weight", 0.3))
        self.pseudo_pool_bg = bool(cfg.loss.get("pseudo_pool_bg", True))
        if cfg.loss.name == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif cfg.loss.name == "l1":
            self.loss = nn.L1Loss(reduction="none")
        else:
            raise NotImplementedError(f"loss {cfg.loss.name} not implemented")

    def forward(self, y_pred, y_true, mask, pseudo=None, distill=None):
        l = self.loss(y_pred, y_true)
        m = mask.unsqueeze(1).float()               # (B,D,H,W) -> (B,1,D,H,W)
        pos = (y_true >= MASK_THRESH).float()
        w = m * pos * (1 + self.pos_weight * y_true) if self.pos_weight > 0 else m * pos
        bg = m * (1 - pos)
        bg_term = (l * bg).sum() / bg.sum().clamp_min(1.0)
        if pseudo is None:
            pos_term = (l * w).sum() / w.sum().clamp_min(1.0)
            if distill is not None:
                # dl_622 combine: plain sum, no balancing weight. The GT term
                # is ~5.6x the distill term at init (measured, 24 crops), so
                # the sum starts out roughly 85% GT / 15% teacher.
                # a plain mean over the crop: the teacher supervises every
                # voxel, there is no positive/background split to make, the GT
                # carve must not apply, and combine asserts rot_scale_aug off
                # so no augmentation fill is in there
                ps = self.loss(y_pred, distill).mean()
                return {"loss": pos_term + self.bg_weight * bg_term + ps,
                        "pos": pos_term, "bg": bg_term, "distill": ps}
            if self.pooled:
                # dl_622: one mean over the whole weight map (= plain masked
                # MSE). The split above exists because a GT Gaussian leaves a
                # 344:1 background (measured: support 0.29% of a crop) that
                # would drown the bump gradient. A soft teacher heatmap
                # supervises 11-66% of the volume instead, so there is nothing
                # to drown — and MASK_THRESH, calibrated for a render that is
                # exactly 0 outside 3 sigma, cuts the teacher's low nonzero
                # floor at a DIFFERENT place in every sample (11% vs 66%),
                # which would make bg_weight mean something different per
                # sample. pos/bg stay in the dict as diagnostics.
                return {"loss": (l * m).sum() / m.sum().clamp_min(1.0),
                        "pos": pos_term, "bg": bg_term}
            return {"loss": pos_term + self.bg_weight * bg_term,
                    "pos": pos_term, "bg": bg_term}
        assert not self.pooled, \
            "loss.pooled with pseudo positives: the pseudo term has no pooled form"
        assert distill is None, \
            "distill_mode=combine with pseudo positives is not supported"
        # dl_173: two supervision sources, one mean each. GT Gaussians keep an
        # undiluted term of their own; EVERYTHING the teacher supervises —
        # its positives AND the background zeros implied by its silence — is
        # one pooled mean at the teacher's trust level pseudo_weight
        # (bg_weight is unused on this path). dl_174 (pseudo_pool_bg=false):
        # the background stays its own weight-bg_weight term — dl_173 showed
        # pooling it at 0.3 costs FP suppression (FP 1,198, worst arm).
        # Dataset ownership follows the source of the target value (with GT
        # cores protected), including pseudo peaks overlapping GT tails.
        ps = pseudo.unsqueeze(1).float()
        w_gt = w * (1 - ps)
        pos_term = (l * w_gt).sum() / w_gt.sum().clamp_min(1.0)
        if self.pseudo_pool_bg:
            w_side = w * ps + bg
            ps_term = (l * w_side).sum() / w_side.sum().clamp_min(1.0)
            return {"loss": pos_term + self.pseudo_weight * ps_term,
                    "pos": pos_term, "bg": bg_term, "ps": ps_term}
        w_ps = w * ps
        ps_term = (l * w_ps).sum() / w_ps.sum().clamp_min(1.0)
        return {"loss": pos_term + self.pseudo_weight * ps_term
                        + self.bg_weight * bg_term,
                "pos": pos_term, "bg": bg_term, "ps": ps_term}


class DivLoss(nn.Module):
    """Daughter-channel loss (dl_195).

    Masked MSE on the <=1.5 sigma cores of supervised GT nodes only; the
    weight map (dataset) is the det-channel GT Gaussian there, so the gradient
    concentrates where inference reads the channel (ch1 peaks). Positive cores
    (div target >= T15, i.e. inside a daughter core) and negative cores are
    averaged SEPARATELY — 302 daughters must not drown in ~133k negatives:
        loss = pos_term + div_neg_weight * neg_term
    """

    def __init__(self, cfg):
        super().__init__()
        self.neg_weight = float(cfg.loss.get("div_neg_weight", 1.0))

    def forward(self, y_pred, y_true, w):
        # upcast before squaring: masked-out regions can saturate under fp16
        # autocast on full frames (same guard as PairLoss.flow)
        l = (y_pred.float() - y_true.float()) ** 2
        w = w.unsqueeze(1).float()
        pos = (y_true >= T15).float()
        wp, wn = w * pos, w * (1 - pos)
        pos_term = (l * wp).sum() / wp.sum().clamp_min(1.0)
        neg_term = (l * wn).sum() / wn.sum().clamp_min(1.0)
        return {"loss": pos_term + self.neg_weight * neg_term,
                "pos": pos_term, "neg": neg_term}


class CoarseLoss(nn.Module):
    """arch=unet25d_coarse: BCE center classification + L1 offset regression
    on the [1,4,4] coarse grid.

    y_pred: (B, 4, Z, Yc, Xc) = [cls_logit, dz, dy, dx]
    cls_t:  (B, 1, Z, Yc, Xc) {0,1}
    off_t:  (B, 3, Z, Yc, Xc) cell-unit offsets, valid at positive cells
    mask:   (B, Z, Yc, Xc) loss weights (min-pooled carve/validity; the
            ignore ring around positives is already zeroed by the dataset)

    Same normalized pos/bg split as MyLoss: each region is a weighted mean and
    bg_weight scales the background term, so the ~1e-3 positive rate does not
    dilute the center gradient. Offsets are gated by offw (the whole near
    block, binary, zeroed where the min-pooled validity died) — NOT by the
    cls weight; a validity-zeroed cell trains neither head.
    """

    def __init__(self, cfg):
        super().__init__()
        self.bg_weight = float(cfg.loss.bg_weight)
        self.off_weight = float(cfg.loss.get("off_weight", 1.0))
        # dl_299 v3: focal modulation of the BACKGROUND term only, as a
        # weighted MEAN (soft-OHEM). The classic focal role (easy negatives
        # drowning the positives under sum normalization) does not exist here
        # — pos/bg are separate means — so gamma>0 instead redistributes the
        # fixed bg budget onto confident false positives: weight p^gamma
        # (detached) turns the ~1/110k per-cell pressure into ~1/N_hard.
        # gamma=0 reproduces the plain mean exactly. Risk (measured caveat):
        # dim unlabeled cells missed by the DoG carve are "bright but
        # negative" and become the hardest-weighted targets.
        self.focal_gamma = float(cfg.loss.get("focal_gamma", 0.0))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_pred, cls_t, off_t, mask, offw):
        # upcast: fp16 autocast on full frames can saturate (DivLoss guard)
        l = self.bce(y_pred[:, 0:1].float(), cls_t.float())
        m = mask.unsqueeze(1).float()
        pos = (cls_t >= 0.5).float()
        wp, wn = m * pos, m * (1 - pos)
        if self.focal_gamma > 0:
            wn = wn * torch.sigmoid(y_pred[:, 0:1].float()).detach() \
                .pow(self.focal_gamma)
        pos_term = (l * wp).sum() / wp.sum().clamp_min(1.0)
        bg_term = (l * wn).sum() / wn.sum().clamp_min(1e-6)
        # offsets are supervised on the whole near block (offw), not just the
        # positive cell — whichever block cell wins NMS must decode correctly
        wo = offw.unsqueeze(1).float()
        loff = (y_pred[:, 1:4].float() - off_t.float()).abs()
        off_term = (loff * wo).sum() / (3.0 * wo.sum()).clamp_min(1.0)
        return {"loss": pos_term + self.bg_weight * bg_term
                + self.off_weight * off_term,
                "cls_pos": pos_term, "cls_bg": bg_term, "off": off_term}


class CoarseGaussLoss(nn.Module):
    """coarse_target=gauss: Gaussian heatmap MSE on the [1,4,4] grid (the
    MyLoss pos/bg split verbatim, MASK_THRESH support) + the same offset-L1
    term as CoarseLoss. focal_gamma does not apply to the MSE teacher."""

    def __init__(self, cfg):
        super().__init__()
        assert float(cfg.loss.get("focal_gamma", 0.0)) == 0, \
            "coarse_target=gauss ignores focal_gamma; set it to 0"
        self.bg_weight = float(cfg.loss.bg_weight)
        self.off_weight = float(cfg.loss.get("off_weight", 1.0))

    def forward(self, y_pred, cls_t, off_t, mask, offw):
        l = (y_pred[:, 0:1].float() - cls_t.float()) ** 2
        m = mask.unsqueeze(1).float()
        pos = (cls_t >= MASK_THRESH).float()
        wp, wn = m * pos, m * (1 - pos)
        pos_term = (l * wp).sum() / wp.sum().clamp_min(1.0)
        bg_term = (l * wn).sum() / wn.sum().clamp_min(1.0)
        wo = offw.unsqueeze(1).float()
        loff = (y_pred[:, 1:4].float() - off_t.float()).abs()
        off_term = (loff * wo).sum() / (3.0 * wo.sum()).clamp_min(1.0)
        return {"loss": pos_term + self.bg_weight * bg_term
                + self.off_weight * off_term,
                "cls_pos": pos_term, "cls_bg": bg_term, "off": off_term}


class PairLoss(nn.Module):
    """Loss for the two-frame pair model (model.temporal=pair).

    y_pred/y_true: (B, 4, D, H, W) = [hm_t, flow dz, dy, dx]
    mask:          (B, 2, D, H, W) = [mask_t, flow weight]

    The frame-t keypoint channel gets the standard masked pos/bg heatmap loss
    (MyLoss); the flow channels get a weighted MSE where the weight map is the
    <=1.5 sigma Gaussian core of linked frame-t nodes (dataset.render_flow):
        loss = hm_t + flow_weight * sum(w * err^2) / (3 * sum(w))
    Flow targets are voxel offsets divided by loss.flow_scale_vox (O(1) after
    normalization, dl_042); flow_weight stays small so the shared decoder/head
    remains heatmap-dominated.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hm = MyLoss(cfg)
        self.flow_weight = float(cfg.loss.get("flow_weight", 0.1))

    def forward(self, y_pred, y_true, mask):
        a = self.hm(y_pred[:, 0:1], y_true[:, 0:1], mask[:, 0])
        w = mask[:, 1].unsqueeze(1)
        # unsupervised flow regions (w == 0) can saturate to +-inf under
        # fp16 autocast on full val frames; inf * 0 = NaN, so zero the diff
        # BEFORE it can meet the weight (torch.where never touches the
        # discarded branch). Upcast to float32 before squaring.
        diff = torch.where(w > 0, (y_pred[:, 1:4] - y_true[:, 1:4]).float(),
                           torch.zeros((), device=y_pred.device))
        flow = (diff ** 2 * w).sum() / (3.0 * w.sum()).clamp_min(1e-6)
        return {"loss": a["loss"] + self.flow_weight * flow,
                "pos": a["pos"], "bg": a["bg"], "flow": flow}
