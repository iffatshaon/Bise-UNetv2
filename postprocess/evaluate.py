import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.amp import autocast
import numpy as np

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from train.models.factory import get_model
    from preprocess.dataset_prep import set_seed, list_image_mask_pairs, split_pairs, get_loaders
    from postprocess.eval_utils import dice_iou_from_logits, count_params, get_macs, benchmark_fps, peak_cuda_mem_mb, calculate_hd95_asd, get_advanced_metrics
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed", action="store_true", help="Use AMP")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save results")
    return parser.parse_args()

def compute_confusion_matrix(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = targets.float()
    
    tp = (preds * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    return tp, tn, fp, fn

def calculate_metrics(tp, tn, fp, fn, eps=1e-7):
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps) # Sensitivity
    specificity = tn / (tn + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps) # Dice
    iou = tp / (tp + fp + fn + eps)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Dice": f1,
        "IoU": iou
    }

def evaluate_loader(model, loader, device, mixed, subset_name="Test"):
    model.eval()
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    hd95_list, asd_list = [], []
    
    all_preds_np = []
    all_targets_np = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast('cuda', enabled=mixed):
                logits = model(xb)
            
            tp, tn, fp, fn = compute_confusion_matrix(logits, yb)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            # HD95 / ASD and py_sod_metrics probabilities
            probs = torch.sigmoid(logits)
            preds_np = (probs > 0.5).cpu().numpy().astype(np.uint8)
            probs_np = probs.cpu().numpy().astype(np.float32)
            targs_np = yb.cpu().numpy().astype(np.uint8)
            
            # Collect for SOD metrics
            for i in range(len(probs_np)):
                p_prob = probs_np[i,0] if probs_np.ndim == 4 else probs_np[i]
                t_mask = targs_np[i,0] if targs_np.ndim == 4 else targs_np[i]
                all_preds_np.append(p_prob)
                all_targets_np.append(t_mask)
            
            # Handle batch for HD95/ASD
            for i in range(len(preds_np)):
                # shape (1, H, W) -> (H, W) if channel dim exists
                p = preds_np[i,0] if preds_np.ndim == 4 else preds_np[i]
                t = targs_np[i,0] if targs_np.ndim == 4 else targs_np[i]
                
                h, a = calculate_hd95_asd(p, t)
                if not np.isnan(h):
                    hd95_list.append(h)
                    asd_list.append(a)
            
    metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
    metrics["HD95"] = np.mean(hd95_list) if hd95_list else 0.0
    metrics["ASD"] = np.mean(asd_list) if asd_list else 0.0
    
    # Calculate SOD metrics at the end safely
    try:
        mae, fm, sm, em = get_advanced_metrics(all_preds_np, all_targets_np)
        metrics["MAE"] = mae
        metrics["F-Measure (Max)"] = fm
        metrics["S-Measure"] = sm
        metrics["E-Measure (Max)"] = em
    except Exception as e:
        print(f"Warning: Failed to calculate SOD metrics: {e}")
        metrics["MAE"] = 0.0
        metrics["F-Measure"] = 0.0
        metrics["S-Measure"] = 0.0
        metrics["E-Measure"] = 0.0
    
    cm_str = (f"Confusion Matrix ({subset_name}):\n"
              f"  TP: {int(total_tp)}, FN: {int(total_fn)}\n"
              f"  FP: {int(total_fp)}, TN: {int(total_tn)}")
    
    return metrics, cm_str

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Identify output file
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        results_file = os.path.join(args.out_dir, "results.txt")
    else:
        # Default to directory of checkpoint
        ckpt_dir = os.path.dirname(args.ckpt)
        results_file = os.path.join(ckpt_dir, "results.txt")
        
    # Open file for writing results
    with open(results_file, "w") as f_out:
        def log(msg):
            print(msg)
            f_out.write(msg + "\n")
            
        log(f"Evaluation Report for Model: {args.model}")
        log(f"Checkpoint: {args.ckpt}")
        log(f"Date: {time.ctime()}")
        log(f"Device: {device}")
        log("-" * 40)

        # Data
        img_dir = os.path.join(args.data_dir, "images")
        msk_dir = os.path.join(args.data_dir, "masks")
        pairs = list_image_mask_pairs(img_dir, msk_dir)
        tr, va, te = split_pairs(pairs, seed=args.seed)
        
        loader_config = {"IMG_SIZE": args.img_size, "BATCH": args.batch}
        # Zero augmentation for evaluation
        eval_aug = {"AUG_FLIP_H":0, "AUG_FLIP_V":0, "AUG_ROTATE_P":0, "AUG_BRIGHTC_P":0}
        
        train_loader, val_loader, test_loader = get_loaders(tr, va, te, {**loader_config, **eval_aug})
        
        # Model
        # Attempt to load args from checkpoint to instantiate correctly if needed
        try:
            if torch.cuda.is_available():
                ckpt = torch.load(args.ckpt, map_location=device)
            else:
                ckpt = torch.load(args.ckpt, map_location='cpu')
                
            base_ch = 32
            if 'args' in ckpt and 'base_ch' in ckpt['args']:
                base_ch = ckpt['args']['base_ch']
                
            model = get_model(args.model, base_ch=base_ch).to(device)
            model.load_state_dict(ckpt['model'])
            log("Model loaded successfully.")
        except Exception as e:
            log(f"Error loading checkpoint: {e}")
            sys.exit(1)
            
        # 1. Benchmarks
        log("\n--- Model Performance Benchmarks ---")
        params = count_params(model)
        log(f"Parameters: {params/1e6:.3f} M")
        
        macs = get_macs(model, args.img_size, device)
        if macs:
            log(f"MACs: {macs[0]/1e9:.2f} G")
            log(f"Params (thop/fvcore verified): {macs[1]/1e6:.3f} M")
        else:
            log("MACs: N/A (thop/fvcore not installed or failed)")
            
        fps = benchmark_fps(model, device, args.img_size, mixed=args.mixed)
        log(f"FPS (Inference Speed): {fps:.2f}")
        
        mem = peak_cuda_mem_mb(model, device, args.img_size, mixed=args.mixed)
        if mem:
            log(f"Peak CUDA Memory: {mem:.1f} MB")
            
        # 2. Evaluation on Splits
        loaders = [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]
        
        for name, loader in loaders:
            log(f"\n--- {name} Set Evaluation ---")
            metrics, cm_str = evaluate_loader(model, loader, device, args.mixed, name)
            
            log(cm_str)
            log("Metrics:")
            for k, v in metrics.items():
                log(f"  {k:<12}: {v:.4f}")

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
