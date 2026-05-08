import os
import argparse
import sys
import json
import time
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.pranet_datasets import get_merged_train_pairs
from preprocess.dataset_prep import set_seed, get_train_val_loaders
from train.models.factory import get_model
from train.losses.loss_functions import weighted_bce_dice_loss
from train.metrics.metric_functions import seg_metrics_from_logits

def train_one_epoch(model, loader, device, optimizer, scaler, mixed, bce_w, dice_w, scheduler=None, scheduler_step_per_batch=False):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    n = 0
    pbar = tqdm(loader, desc="  Training", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=mixed):
            logits = model(xb)
            loss = weighted_bce_dice_loss(logits, yb, bce_w=bce_w, dice_w=dice_w)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler and scheduler_step_per_batch:
            scheduler.step()
            
        bs = xb.size(0)
        total_loss += loss.item() * bs
        dice, _ = seg_metrics_from_logits(logits, yb)
        total_dice += dice * bs
        n += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / n, total_dice / n

@torch.no_grad()
def evaluate(model, loader, device, mixed, bce_w, dice_w):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with autocast('cuda', enabled=mixed):
            logits = model(xb)
            loss = weighted_bce_dice_loss(logits, yb, bce_w=bce_w, dice_w=dice_w)
        bs = xb.size(0)
        total_loss += loss.item() * bs
        dice, _ = seg_metrics_from_logits(logits, yb)
        total_dice += dice * bs
        n += bs
    return total_loss / n, total_dice / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Load HParams
    with open(args.hparams, 'r') as f:
        hparams = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = get_merged_train_pairs(args.data_root)
    
    results = {}

    for seed in args.seeds:
        print(f"\n--- Starting Seed {seed} ---")
        set_seed(seed)
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
        
        seed_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(pairs)):
            print(f"\nFold {fold+1}/{args.folds}")
            tr_pairs = [pairs[i] for i in train_idx]
            va_pairs = [pairs[i] for i in val_idx]
            
            config = {
                "IMG_SIZE": hparams["img_size"],
                "BATCH": hparams["batch_size"],
                "WORKERS": args.workers,
                "AUG_FLIP_H": 0.5, "AUG_FLIP_V": 0.5,
                "AUG_ROTATE_P": 0.5, "AUG_BRIGHTC_P": 0.5,
                "AUG_BRIGHT_A": 0.2, "AUG_BRIGHT_B": 15,
                "AUG_ELASTIC_P": 0.2, "AUG_CUTOUT_P": 0.2
            }
            
            train_loader, val_loader = get_train_val_loaders(tr_pairs, va_pairs, config)
            
            model = get_model(args.model, base_ch=args.base_ch).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
            scaler = GradScaler('cuda', enabled=args.mixed)
            
            bce_w = hparams["bce_weight"]
            dice_w = 1.0 - bce_w
            
            if hparams["scheduler"] == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                step_per_batch = False
            elif hparams["scheduler"] == "reduceLR":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
                step_per_batch = False
            else: # onecycle
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams["lr"], total_steps=args.epochs * len(train_loader))
                step_per_batch = True
                
            best_dice = 0.0
            patience = 10
            patience_counter = 0
            min_delta = 1e-4
            
            fold_dir = os.path.join(args.out_dir, args.model, f"seed_{seed}", f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            
            history = []
            for epoch in range(1, args.epochs + 1):
                tr_loss, tr_dice = train_one_epoch(model, train_loader, device, optimizer, scaler, args.mixed, bce_w, dice_w, scheduler, step_per_batch)
                va_loss, va_dice = evaluate(model, val_loader, device, args.mixed, bce_w, dice_w)
                
                # ... existing scheduler code ...
                if hparams["scheduler"] == "reduceLR":
                    scheduler.step(va_dice)
                elif hparams["scheduler"] == "cosine":
                    scheduler.step()
                
                # Check for improvement
                if va_dice > best_dice + min_delta:
                    best_dice = va_dice
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(fold_dir, "best.pt"))
                else:
                    patience_counter += 1
                
                history.append({
                    "epoch": epoch,
                    "tr_loss": tr_loss, "tr_dice": tr_dice,
                    "va_loss": va_loss, "va_dice": va_dice
                })
                
                print(f"Epoch {epoch:03d}/{args.epochs} | Tr Dice: {tr_dice:.4f} | Va Dice: {va_dice:.4f} | Best: {best_dice:.4f} | Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Save history
            import pandas as pd
            pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
            
            seed_results.append(best_dice)
            print(f"Fold {fold+1} Best Dice: {best_dice:.4f}")

        results[str(seed)] = seed_results
        
    # Save results
    with open(os.path.join(args.out_dir, args.model, "kfold_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate mean and std
    all_dices = [d for s in results.values() for d in s]
    import numpy as np
    print(f"\nFinal K-Fold Result: {np.mean(all_dices):.4f} ± {np.std(all_dices):.4f}")

if __name__ == "__main__":
    main()
