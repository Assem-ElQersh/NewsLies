import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def class_weights_from(labels, num_classes, scheme="balanced", beta=0.9999):
    """scheme: 'balanced' (inverse-frequency, mean-normalized) or
    'effective_number' (Cui et al., 2019)."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    if scheme == "balanced":
        w = counts.sum() / (num_classes * counts)
    elif scheme == "effective_number":
        effective = (1 - beta ** counts) / (1 - beta)
        w = 1.0 / effective
        w = w / w.mean()
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return w


def get_loss_fn(loss_type: str, class_weights=None, label_smoothing=0.1, focal_gamma=2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=class_weights)
    if loss_type == "balanced_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "effective_number":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"Unknown loss {loss_type}")


def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, T=4.0):
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft, hard.item(), soft.item()
