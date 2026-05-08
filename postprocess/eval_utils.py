# eval_utils.py
import os, platform, time, warnings
import torch
from torch.amp import autocast

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- metrics ----------
def dice_iou_from_logits(logits, target, eps: float = 1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2*inter + eps) / (union + eps)

    inter2 = (pred * target).sum(dim=(2,3))
    denom = (pred + target - pred*target).sum(dim=(2,3)) + eps
    iou = inter2 / denom
    return dice.mean().item(), iou.mean().item()

import numpy as np
import cv2

def calculate_hd95_asd(pred_mask, target_mask, spacing=1.0):
    """
    Compute HD95 and ASD using OpenCV Distance Transform.
    pred_mask, target_mask: (H, W) numpy boolean/uint8 arrays.
    spacing: pixel spacing (default 1.0).
    Returns (hd95, asd).
    If one mask is empty, returns (NaN, NaN) or (Inf, Inf).
    """
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    target_mask = (target_mask > 0.5).astype(np.uint8)
    
    if np.sum(pred_mask) == 0 and np.sum(target_mask) == 0:
        return 0.0, 0.0
    if np.sum(pred_mask) == 0 or np.sum(target_mask) == 0:
        # One empty, one not. Distance is effectively undefined or huge.
        # We'll return 0 for practical purposes if both empty, but if one missing, 
        # it's usually penalized. Let's return nan.
        return float('nan'), float('nan')
        
    # Get boundaries
    # boundary = mask - eroded(mask)
    # Using 3x3 struct element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    pred_border = pred_mask - cv2.erode(pred_mask, kernel)
    target_border = target_mask - cv2.erode(target_mask, kernel)
    
    # Distance Transform on INVERTED borders
    # dt[x,y] = distance to nearest boundary pixel
    # We want distance FROM pred_border TO target_border
    # So we compute DT of (NOT target_border)
    dt_target = cv2.distanceTransform(1 - target_border, cv2.DIST_L2, 5)
    
    # Distances from Pred boundary points to Target boundary
    # We only care about dt values WHERE pred_border is 1
    d_pred_to_target = dt_target[pred_border == 1]
    
    # Symmetric: Pred to Target
    dt_pred = cv2.distanceTransform(1 - pred_border, cv2.DIST_L2, 5)
    d_target_to_pred = dt_pred[target_border == 1]
    
    # Concatenate for HD95
    # HD95 is usually defined as the 95th percentile of the symmetric Hausdorff distance
    # The set of all distances is d_pred_to_target U d_target_to_pred
    all_dists = np.concatenate([d_pred_to_target, d_target_to_pred])
    
    if len(all_dists) > 0:
        hd95 = np.percentile(all_dists, 95) * spacing
        asd = np.mean(all_dists) * spacing
    else:
        hd95, asd = 0.0, 0.0
        
    return hd95, asd

def get_advanced_metrics(preds, targets):
    """
    Compute MAE, max F-measure, S-measure, and max E-measure using py_sod_metrics.
    preds: List of (H, W) numpy float arrays (probabilities [0, 1]).
    targets: List of (H, W) numpy uint8 arrays (binary masks {0, 1}).
    """
    import py_sod_metrics
    
    mae_metric = py_sod_metrics.MAE()
    fm_metric = py_sod_metrics.FmeasureV2()
    fm_metric.add_handler("fm", py_sod_metrics.FmeasureHandler(with_dynamic=True, with_adaptive=False, with_binary=False))
    sm_metric = py_sod_metrics.Smeasure()
    em_metric = py_sod_metrics.Emeasure()
    
    for p, t in zip(preds, targets):
        # py_sod_metrics expects inputs as numpy arrays:
        # pred: float array in [0, 1] or uint8 in [0, 255]
        # target: uint8 array in [0, 255]
        p_np = p.astype(np.float32)
        t_np = (t.astype(np.uint8) * 255)
        
        mae_metric.step(p_np, t_np)
        fm_metric.step(p_np, t_np)
        sm_metric.step(p_np, t_np)
        em_metric.step(p_np, t_np)
        
    mae = mae_metric.get_results()["mae"]
    fm_results = fm_metric.get_results()
    fm = fm_results["fm"]["dynamic"].max() if "fm" in fm_results else 0.0
    sm = sm_metric.get_results()["sm"]
    em_results = em_metric.get_results()
    em = em_results["em"]["curve"].max() if "em" in em_results else 0.0
    
    if hasattr(fm, "item"): fm = fm.item()
    if hasattr(em, "item"): em = em.item()
    
    return float(mae), float(fm), float(sm), float(em)


# ---------- params / macs / fps / mem ----------
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _try_macs_thop(model: torch.nn.Module, img_size: int, device: torch.device):
    try:
        from thop import profile
        model_eval = model.eval()
        dummy = torch.randn(1,3,img_size,img_size, device=device)
        macs, params = profile(model_eval, inputs=(dummy,), verbose=False)
        return int(macs), int(params)
    except Exception as e:
        print(f"[thop] Failed: {e}")
        return None

def _try_macs_fvcore(model: torch.nn.Module, img_size: int, device: torch.device):
    try:
        from fvcore.nn import FlopCountAnalysis
        model_eval = model.eval()
        dummy = torch.randn(1,3,img_size,img_size, device=device)
        flops = FlopCountAnalysis(model_eval, dummy).total()
        params = count_params(model)
        return int(flops), int(params)
    except Exception as e:
        print(f"[fvcore] Failed: {e}")
        return None

def get_macs(model: torch.nn.Module, img_size: int, device: torch.device):
    r = _try_macs_thop(model, img_size, device)
    if r is not None: return r
    r = _try_macs_fvcore(model, img_size, device)
    if r is not None: return r
    return None

def benchmark_fps(model, device, img_size=256, iters=60, warmup=10, mixed=True):
    model.eval()
    x = torch.randn(1,3,img_size,img_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            with autocast(device_type='cuda', enabled=(device.type=='cuda') and mixed):
                _ = model(x)
    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            with autocast(device_type='cuda', enabled=(device.type=='cuda') and mixed):
                _ = model(x)
    if device.type == 'cuda': torch.cuda.synchronize()
    return iters / (time.time()-t0)

def peak_cuda_mem_mb(model, device, img_size=256, mixed=True):
    if device.type != 'cuda': return None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        x = torch.randn(1,3,img_size,img_size).to(device)
        with autocast(device_type='cuda', enabled=mixed):
            _ = model(x)
    return torch.cuda.max_memory_allocated()/1024/1024

# ---------- device / edge simulation ----------
def apply_edge_simulation(sim_mode: str | None, prefer_cuda: bool = True, profile: dict | None = None):
    sim_mode = (sim_mode or '').lower()
    device = torch.device('cuda' if (prefer_cuda and torch.cuda.is_available()) else 'cpu')

    if sim_mode == 'pi5':
        device = torch.device('cpu')
        max_threads = profile.get("MAX_THREADS", 4) if profile else 4
        interop_threads = profile.get("INTEROP_THREADS", 1) if profile else 1
        try:
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(interop_threads)
            print(f"[pi5] CPU-only, threads={max_threads}, interop={interop_threads}")
        except Exception:
            pass
        try:
            torch.backends.mkldnn.enabled = False
        except Exception:
            pass
    elif sim_mode == 'nano':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        print("[nano-sim] Deterministic cuDNN for reproducibility.")
    return device

def setup_cuda_limits_for_linux_mps(sm_percent: int | None):
    """
    Desktop-only approximation (Linux): throttle SM share via CUDA MPS.
    Jetson Nano itself does NOT support MPS.
    Usage before running Python:
      $ nvidia-cuda-mps-control -d
      $ export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
    """
    if sm_percent is None: return
    if platform.system() != 'Linux':
        print('[MPS] SM partitioning requires Linux. Ignoring sm_percent.')
        return
    os.environ.setdefault('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', str(max(1, min(100, sm_percent))))
    print(f"[MPS] Set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']} (start MPS daemon separately)")
