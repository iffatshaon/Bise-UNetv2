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
    fg = pred[gt > 0.5]
    bg = 1.0 - pred[gt <= 0.5]
    o_fg = fg.mean() if fg.numel() > 0 else torch.tensor(0.0, device=pred.device)
    o_bg = bg.mean() if bg.numel() > 0 else torch.tensor(0.0, device=pred.device)
    return 0.5 * (o_fg + o_bg)

def _s_region(pred, gt):
    # Simplified region measure for robustness
    # Measure structural similarity between pred and gt
    # Here we just use 1 - MAE as a proxy for region similarity if complex structural measure is too slow
    return 1.0 - torch.abs(pred - gt).mean()

def s_measure(pred, gt):
    # Standard alpha = 0.5
    y = gt.mean()
    if y == 0: return 1.0 - pred.mean()
    if y == 1: return pred.mean()
    return 0.5 * _s_object(pred, gt) + 0.5 * _s_region(pred, gt)

def e_measure(pred, gt):
    # Enhanced-alignment measure
    # Alignment matrix implementation
    gt = gt.float()
    pred = pred.float()
    
    # Per-image mean removal
    gt_mean = gt.mean()
    pred_mean = pred.mean()
    
    gt_f = gt - gt_mean
    pred_f = pred - pred_mean
    
    # Alignment matrix
    align_matrix = 2.0 * gt_f * pred_f / (gt_f**2 + pred_f**2 + 1e-7)
    enhanced = ((align_matrix + 1.0)**2) / 4.0
    return enhanced.mean()

@torch.no_grad()
def compute_all_metrics(logits, targets, thresh=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    
    # Overall metrics on flattened tensors
    preds_f = preds.reshape(-1)
    targets_f = targets.reshape(-1)
    
    inter = (preds_f * targets_f).sum()
    total_preds = preds_f.sum()
    total_targets = targets_f.sum()
    
    dice = (2.0 * inter + eps) / (total_preds + total_targets + eps)
    iou = (inter + eps) / (total_preds + total_targets - inter + eps)
    
    precision = (inter + eps) / (total_preds + eps)
    recall = (inter + eps) / (total_targets + eps)
    
    # F-beta (beta^2 = 0.3)
    beta2 = 0.3
    f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
    
    # MAE
    mae = torch.abs(probs - targets).mean()
    
    # S and E measures (average over batch)
    s_val = 0.0
    e_val = 0.0
    bs = logits.size(0)
    for i in range(bs):
        s_val += s_measure(probs[i], targets[i]).item()
        e_val += e_measure(probs[i], targets[i]).item()
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f2': f2.item(),
        'mae': mae.item(),
        's_measure': s_val / bs,
        'e_measure': e_val / bs
    }
