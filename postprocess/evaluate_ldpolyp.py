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
    from preprocess.dataset_prep import set_seed
    from preprocess.ldpolyp_prep import get_ldpolyp_loaders # LDPolyp Loader
    from postprocess.eval_utils import dice_iou_from_logits, count_params, get_macs, benchmark_fps, peak_cuda_mem_mb, calculate_hd95_asd
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="LDPolyp Evaluation Script")
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
    recall = tp / (tp + fn + eps) 
    specificity = tn / (tn + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "Specificity": specificity, "Dice": f1, "IoU": iou
    }

def evaluate_loader(model, loader, device, mixed, subset_name="Test"):
    model.eval()
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    hd95_list, asd_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with autocast('cuda', enabled=mixed):
                logits = model(xb)
            tp, tn, fp, fn = compute_confusion_matrix(logits, yb)
            total_tp += tp; total_tn += tn; total_fp += fp; total_fn += fn
            
            # HD95 / ASD
            probs = torch.sigmoid(logits)
            preds_np = (probs > 0.5).cpu().numpy().astype(np.uint8)
            targs_np = yb.cpu().numpy().astype(np.uint8)
            
            for i in range(len(preds_np)):
                p = preds_np[i,0] if preds_np.ndim == 4 else preds_np[i]
                t = targs_np[i,0] if targs_np.ndim == 4 else targs_np[i]
                h, a = calculate_hd95_asd(p, t)
                if not np.isnan(h):
                    hd95_list.append(h)
                    asd_list.append(a)
                    
    metrics = calculate_metrics(total_tp, total_tn, total_fp, total_fn)
    metrics["HD95"] = np.mean(hd95_list) if hd95_list else 0.0
    metrics["ASD"] = np.mean(asd_list) if asd_list else 0.0
    
    cm_str = (f"Confusion Matrix ({subset_name}):\n"
              f"  TP: {int(total_tp)}, FN: {int(total_fn)}\n"
              f"  FP: {int(total_fp)}, TN: {int(total_tn)}")
    return metrics, cm_str

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        results_file = os.path.join(args.out_dir, "results.txt")
    else:
        results_file = os.path.join(os.path.dirname(args.ckpt), "results.txt")
        
    with open(results_file, "w") as f_out:
        def log(msg):
            print(msg); f_out.write(msg + "\n")
            
        log(f"LDPolyp Evaluation: {args.model}")
        log(f"Checkpoint: {args.ckpt}")
        log("-" * 40)
        
        # Loader
        loader_config = {"IMG_SIZE": args.img_size, "BATCH": args.batch,
                         "AUG_FLIP_H":0, "AUG_FLIP_V":0, "AUG_ROTATE_P":0, "AUG_BRIGHTC_P":0}
        
        train_loader, val_loader, test_loader = get_ldpolyp_loaders(args.data_dir, loader_config, args.seed)
        
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
            log("Model loaded.")
        except Exception as e:
            log(f"Error loading checkpoint: {e}")
            sys.exit(1)
            
        # Benchmarks
        log("\n--- Benchmarks ---")
        log(f"Params: {count_params(model)/1e6:.3f} M")
        fps = benchmark_fps(model, device, args.img_size, mixed=args.mixed)
        log(f"FPS: {fps:.2f}")
        
        # Eval
        loaders = [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]
        for name, loader in loaders:
            log(f"\n--- {name} Set ---")
            metrics, cm_str = evaluate_loader(model, loader, device, args.mixed, name)
            log(cm_str)
            for k,v in metrics.items(): log(f"  {k}: {v:.4f}")

    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()
