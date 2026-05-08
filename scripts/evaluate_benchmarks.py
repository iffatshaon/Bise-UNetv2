import os
import argparse
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.pranet_datasets import get_kvasir_pairs, get_clinicdb_pairs, get_eval_pairs
from preprocess.dataset_prep import get_eval_loader
from train.models.factory import get_model
from train.metrics.metric_functions import compute_all_metrics

@torch.no_grad()
def evaluate_on_benchmark(model, loader, device, mixed):
    model.eval()
    all_metrics = []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=mixed):
            logits = model(xb)
        
        # We need per-image metrics if possible, but compute_all_metrics handles batches
        # Here we collect batch results and average them later
        metrics = compute_all_metrics(logits, yb)
        all_metrics.append(metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])
    return avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--base-ch", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Prepare Benchmarks
    benchmarks = {
        "Kvasir-Test": get_kvasir_pairs(args.data_root)[1],
        "CVC-ClinicDB-Test": get_clinicdb_pairs(args.data_root)[1],
        "CVC-300": get_eval_pairs(args.data_root, "cvc300"),
        "EndoScene-Val": get_eval_pairs(args.data_root, "endoscene_val"),
        "CVC-ColonDB": get_eval_pairs(args.data_root, "colondb"),
        "ETIS": get_eval_pairs(args.data_root, "etis")
    }

    # 2. Find Checkpoints
    ckpt_paths = []
    for root, dirs, files in os.walk(os.path.join(args.ckpt_dir, args.model)):
        if "best.pt" in files:
            ckpt_paths.append(os.path.join(root, "best.pt"))
    
    print(f"Found {len(ckpt_paths)} checkpoints for evaluation.")

    full_results = []

    # 3. Evaluate each checkpoint on each benchmark
    for ckpt_path in tqdm(ckpt_paths, desc="Checkpoints"):
        # Extract seed and fold info from path
        parts = ckpt_path.split(os.sep)
        seed = next((p for p in parts if p.startswith("seed_")), "unknown")
        fold = next((p for p in parts if p.startswith("fold_")), "unknown")
        
        model = get_model(args.model, base_ch=args.base_ch).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        for name, pairs in benchmarks.items():
            loader = get_eval_loader(pairs, args.img_size, args.batch_size)
            metrics = evaluate_on_benchmark(model, loader, device, args.mixed)
            metrics.update({
                "checkpoint": ckpt_path,
                "dataset": name,
                "seed": seed,
                "fold": fold
            })
            full_results.append(metrics)

    # 4. Process and Save Results
    df = pd.DataFrame(full_results)
    df.to_csv(os.path.join(args.out_dir, f"eval_detailed_{args.model}.csv"), index=False)

    # Calculate summary: mean and std per dataset
    summary = df.groupby("dataset").agg({
        "dice": ["mean", "std"],
        "iou": ["mean", "std"],
        "f2": ["mean", "std"],
        "mae": ["mean", "std"],
        "s_measure": ["mean", "std"],
        "e_measure": ["mean", "std"]
    })
    
    summary.to_csv(os.path.join(args.out_dir, f"eval_summary_{args.model}.csv"))
    print("\nEvaluation Summary:")
    print(summary)

if __name__ == "__main__":
    main()
