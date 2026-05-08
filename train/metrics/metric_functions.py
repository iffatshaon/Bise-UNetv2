import torch
import numpy as np

@torch.no_grad()
def seg_metrics_from_logits(logits, targets, thresh=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = (preds + targets - preds * targets).sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

def _s_object(pred, gt):
    fg = torch.where(gt==0, torch.zeros_like(pred), pred)
    bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
    o_fg = fg.sum() / (gt.sum() + 1e-7)
    o_bg = bg.sum() / ((1-gt).sum() + 1e-7)
    return 0.5 * (o_fg + o_bg)

def _s_region(pred, gt):
    x = gt.mean()
    if x == 0:
        return 1 - pred.mean()
    elif x == 1:
        return pred.mean()
    else:
        # Simplified region measure
        return 1 - torch.abs(pred - gt).mean()

def s_measure(pred, gt):
    # Standard S-measure is alpha * S_object + (1-alpha) * S_region
    # Here pred and gt are tensors of shape (H, W) or (1, H, W)
    y = gt.mean()
    if y == 0: return 1 - pred.mean()
    if y == 1: return pred.mean()
    return 0.5 * (_s_object(pred, gt) + _s_region(pred, gt))

def e_measure(pred, gt):
    # Enhanced-alignment measure
    # Simple implementation of the alignment matrix based E-measure
    gt = gt.bool()
    pred = pred.float()
    fm = pred - pred.mean()
    gt_f = gt.float() - gt.float().mean()
    align_matrix = 2 * fm * gt_f / (fm**2 + gt_f**2 + 1e-7)
    enhanced = ((align_matrix + 1)**2) / 4
    return enhanced.mean()

@torch.no_grad()
def compute_all_metrics(logits, targets, thresh=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    
    # Flatten for some metrics
    preds_f = preds.view(-1)
    targets_f = targets.view(-1)
    
    inter = (preds_f * targets_f).sum()
    union = (preds_f + targets_f - preds_f * targets_f).sum()
    
    dice = (2.0 * inter + eps) / (preds_f.sum() + targets_f.sum() + eps)
    iou = (inter + eps) / (union + eps)
    
    precision = (inter + eps) / (preds_f.sum() + eps)
    recall = (inter + eps) / (targets_f.sum() + eps)
    
    # F-measure with beta^2 = 0.3
    beta2 = 0.3
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    
    # MAE
    mae = torch.abs(probs - targets).mean()
    
    # S and E measures (usually calculated per image then averaged)
    s_val = 0.0
    e_val = 0.0
    bs = logits.size(0)
    for i in range(bs):
        s_val += s_measure(probs[i], targets[i])
        e_val += e_measure(probs[i], targets[i])
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f2': f_beta.item(), # Called f2 in plan but beta^2=0.3
        'mae': mae.item(),
        's_measure': (s_val / bs).item(),
        'e_measure': (e_val / bs).item()
    }
