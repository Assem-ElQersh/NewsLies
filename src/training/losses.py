import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def get_loss_fn(loss_type: str, class_weights=None):
    """Factory for selecting the loss function."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "label_smoothing":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    elif loss_type == "focal":
        return FocalLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss {loss_type}")
