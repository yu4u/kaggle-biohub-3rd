"""Frozen warm-start teacher and pointwise Bernoulli consistency loss."""
import copy

import torch
import torch.nn.functional as F


def make_teacher(model, initialization, weight):
    if weight < 0:
        raise ValueError('Teacher weight must be nonnegative')
    if weight == 0:
        return None
    if initialization is None:
        raise ValueError('Teacher consistency requires a verified warm-start checkpoint')
    return copy.deepcopy(model).requires_grad_(False).eval()


def teacher_kl(student_logits, teacher_logits, class_balanced=False):
    """KL(teacher || student), optionally averaged equally over teacher classes."""
    teacher = teacher_logits.detach().float()
    target = torch.sigmoid(teacher)
    if not class_balanced:
        return (F.binary_cross_entropy_with_logits(student_logits.float(), target)
                - F.binary_cross_entropy_with_logits(teacher, target))
    kl = (F.binary_cross_entropy_with_logits(student_logits.float(), target, reduction='none')
          - F.binary_cross_entropy_with_logits(teacher, target, reduction='none'))
    positive = target >= .5
    groups = [kl[mask].mean() for mask in (positive, ~positive) if bool(mask.any())]
    return torch.stack(groups).mean()
