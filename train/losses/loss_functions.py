import torch
import torch.nn as nn

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    return (1.0 - (num / den)).mean()

def weighted_bce_dice_loss(logits, targets, bce_w=0.5, dice_w=0.5, pos_weight=None):
    if pos_weight is not None:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    else:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    
    dice = dice_loss_from_logits(logits, targets)
    return bce_w * bce + dice_w * dice

def bce_dice_loss(logits, targets, pos_weight=None):
    return weighted_bce_dice_loss(logits, targets, bce_w=0.5, dice_w=0.5, pos_weight=pos_weight)
