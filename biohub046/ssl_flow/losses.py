"""Label-free objectives for dense volume correspondence."""
import torch
import torch.nn.functional as F
from .ops import forward_backward_mask, warp


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x.square() + eps * eps)


def census(x):
    """Local contrast descriptor, robust to illumination/gain changes."""
    mean = F.avg_pool3d(x, 3, stride=1, padding=1)
    d = x - mean
    return d / torch.sqrt(d.square() + 0.01)


def gradient_loss(flow):
    terms = []
    for axis in (-3, -2, -1):
        n = flow.shape[axis]
        terms.append(charbonnier(flow.narrow(axis, 1, n - 1) - flow.narrow(axis, 0, n - 1)).mean())
    return sum(terms) / len(terms)


def synthetic_equivariance_loss(model, frame, max_shift=(2.0, 3.0, 3.0),
                                 shift=None, smooth_amplitude=(0.0, 0.0, 0.0)):
    """Teach flow from label-free pairs made by translating a real frame.

    A synthetic image is defined as ``translated[p] = frame[p + flow[p]]``;
    therefore translated->frame flow is known exactly even for non-rigid
    fields. Only non-padded voxels are supervised.
    """
    batch, _, z, y, x = frame.shape
    if shift is None:
        limit = frame.new_tensor(max_shift).view(1, 3)
        shift = (2.0 * torch.rand(batch, 3, device=frame.device) - 1.0) * limit
    else:
        shift = torch.as_tensor(shift, device=frame.device, dtype=frame.dtype)
        if shift.ndim == 1:
            shift = shift[None].expand(batch, -1)
    target = shift[:, :, None, None, None].expand(batch, 3, z, y, x).clone()
    amplitude = frame.new_tensor(smooth_amplitude).view(1, 3, 1, 1, 1)
    if torch.any(amplitude > 0):
        coarse = torch.randn(batch, 3, 4, 4, 4, device=frame.device,
                             dtype=frame.dtype)
        smooth = F.interpolate(coarse, size=(z, y, x), mode="trilinear",
                               align_corners=True)
        reduce_dims = (2, 3, 4)
        smooth = smooth - smooth.mean(reduce_dims, keepdim=True)
        smooth = smooth / (smooth.std(reduce_dims, keepdim=True) + 1e-6)
        target = target + amplitude * smooth
    translated = warp(frame, target, padding_mode="zeros")
    valid = warp(torch.ones_like(frame), target, padding_mode="zeros") > 0.999
    predicted = model(translated, frame)
    error = charbonnier(predicted - target)
    return (error * valid).sum() / (valid.sum() * 3 + 1e-6)


def descriptor_loss(features_a, features_b, flow_ab, mask, margin=0.2):
    """Warped positive consistency plus deterministic spatial-shift negatives."""
    warped_b = F.normalize(warp(features_b, flow_ab), dim=1)
    positive = 1.0 - (features_a * warped_b).sum(1, keepdim=True)
    # A half-volume shift supplies non-corresponding cells without N^2 storage.
    shifted_b = torch.roll(warped_b, shifts=(max(features_a.shape[-2] // 2, 1),), dims=(-2,))
    negative = F.relu((features_a * shifted_b).sum(1, keepdim=True) - margin)
    return ((positive + negative) * mask).sum() / (mask.sum() + 1e-6)


def self_supervised_loss(a, b, flow_ab, flow_ba, cfg, features=None):
    wa, wb = warp(b, flow_ab), warp(a, flow_ba)
    threshold = float(cfg.occlusion_threshold_vox)
    residual_ab = flow_ab + warp(flow_ba, flow_ab)
    residual_ba = flow_ba + warp(flow_ab, flow_ba)
    ma = forward_backward_mask(residual=residual_ab, threshold=threshold)
    mb = forward_backward_mask(residual=residual_ba, threshold=threshold)

    # Most voxels are dark background. Emphasize fluorescent structures and
    # local contrast so zero flow cannot win by explaining only background.
    def importance(x):
        contrast = (x - F.avg_pool3d(x, 5, stride=1, padding=2)).abs()
        signal = torch.maximum(x.clamp(0, 1), (5.0 * contrast).clamp(0, 1))
        return 1.0 + float(cfg.get("foreground_gain", 0.0)) * signal

    ia, ib = importance(a), importance(b)

    def masked(value, mask, weight):
        w = mask * weight
        return (value * w).sum() / (w.sum() * value.shape[1] + 1e-6)

    photo = .5 * (masked(charbonnier(a - wa), ma, ia) + masked(charbonnier(b - wb), mb, ib))
    cen = .5 * (masked(charbonnier(census(a) - census(wa)), ma, ia) +
                masked(charbonnier(census(b) - census(wb)), mb, ib))
    # Do not mask cycle residuals by their own magnitude: inconsistent voxels
    # are exactly the locations this term needs to correct.
    cyc = .5 * (charbonnier(residual_ab).mean() + charbonnier(residual_ba).mean())
    smooth = .5 * (gradient_loss(flow_ab) + gradient_loss(flow_ba))
    desc = flow_ab.new_zeros(())
    if features is not None:
        desc = .5 * (descriptor_loss(features[0], features[1], flow_ab, ma,
                                     float(cfg.descriptor_margin)) +
                     descriptor_loss(features[1], features[0], flow_ba, mb,
                                     float(cfg.descriptor_margin)))
    total = (cfg.photometric * photo + cfg.census * cen + cfg.smoothness * smooth +
             cfg.cycle * cyc + float(cfg.get("descriptor", 0.0)) * desc)
    return total, {"photo": photo.detach(), "census": cen.detach(), "smooth": smooth.detach(),
                   "cycle": cyc.detach(), "descriptor": desc.detach(),
                   "valid": .5 * (ma.mean() + mb.mean())}
