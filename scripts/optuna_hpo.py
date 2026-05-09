import os
import argparse
import sys
import json
import torch
import torch.optim as optim
import optuna
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.pranet_datasets import get_merged_train_pairs
from preprocess.dataset_prep import set_seed, get_train_val_loaders
from train.models.factory import get_model
from train.losses.loss_functions import weighted_bce_dice_loss
from train.metrics.metric_functions import seg_metrics_from_logits

def objective(trial, args, device):
    # ... previous code ...
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "reduceLR", "onecycle"])
    bce_weight = trial.suggest_float("bce_weight", 0.3, 0.7)
    img_size = trial.suggest_categorical("img_size", [256, 352])
    
    # Transformer models like rfdetr, eomt, cdpolypnet are VERY memory intensive at 352+batch=128
    # We add a safety check for these models to avoid known OOM configurations
    if args.model in ['rfdetr', 'eomt', 'cdpolypnet'] and img_size == 352 and batch_size > 64:
        raise optuna.exceptions.TrialPruned("Configuration too heavy for VRAM")

    dice_weight = 1.0 - bce_weight
    
    try:
        # K-fold CV for HPO (using 3 folds as per plan)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        pairs = get_merged_train_pairs(args.data_root)
        
        if not pairs:
            print("ERROR: No dataset pairs found. Check your data-root.")
            return 0.0
        
        print(f"HPO starting for {args.model}. Total pairs: {len(pairs)}")
        
        fold_dices = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(pairs)):
            tr_pairs = [pairs[i] for i in train_idx]
            va_pairs = [pairs[i] for i in val_idx]
            
            config = {
                "IMG_SIZE": img_size,
                "BATCH": batch_size,
                "WORKERS": args.workers,
                "AUG_FLIP_H": 0.5, "AUG_FLIP_V": 0.5,
                "AUG_ROTATE_P": 0.5, "AUG_BRIGHTC_P": 0.5,
                "AUG_BRIGHT_A": 0.2, "AUG_BRIGHT_B": 15,
                "AUG_ELASTIC_P": 0.2, "AUG_CUTOUT_P": 0.2
            }
            
            train_loader, val_loader = get_train_val_loaders(tr_pairs, va_pairs, config)
            
            model = get_model(args.model, base_ch=args.base_ch).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scaler = GradScaler('cuda', enabled=args.mixed)
            
            if scheduler_name == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif scheduler_name == "reduceLR":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            else:
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=args.epochs * len(train_loader))
                
            best_fold_dice = 0.0
            history = []
            
            pbar = tqdm(range(1, args.epochs + 1), desc=f"Fold {fold} Trial {trial.number}", leave=False)
            for epoch in pbar:
                # Train
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda', enabled=args.mixed):
                        logits = model(xb)
                        loss = weighted_bce_dice_loss(logits, yb, bce_w=bce_weight, dice_w=dice_weight)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler_name == "onecycle":
                        scheduler.step()
                
                # Val
                model.eval()
                total_dice, n = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        with autocast('cuda', enabled=args.mixed):
                            logits = model(xb)
                        dice, _ = seg_metrics_from_logits(logits, yb)
                        total_dice += dice * xb.size(0)
                        n += xb.size(0)
                
                va_dice = total_dice / n
                if scheduler_name == "reduceLR":
                    scheduler.step(va_dice)
                elif scheduler_name == "cosine":
                    scheduler.step()
                    
                history.append({"epoch": epoch, "dice": va_dice})
                best_fold_dice = max(best_fold_dice, va_dice)
                pbar.set_postfix({"dice": f"{va_dice:.4f}"})
                
                if epoch % 5 == 0 or epoch == args.epochs:
                     print(f"Trial {trial.number} | Fold {fold} | Epoch {epoch:02d} | Dice: {va_dice:.4f} | Best: {best_fold_dice:.4f}")

                if fold == 0:
                    trial.report(va_dice, epoch)
                    if trial.should_prune():
                        print(f"  Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.exceptions.TrialPruned()
            
            # Save trial history
            import pandas as pd
            trial_dir = os.path.join(args.out_dir, f"trial_{trial.number}")
            os.makedirs(trial_dir, exist_ok=True)
            pd.DataFrame(history).to_csv(os.path.join(trial_dir, f"fold_{fold}_history.csv"), index=False)
            
            fold_dices.append(best_fold_dice)
            if fold == 0 and fold_dices[0] < 0.3: break
            
            # Memory cleanup after each fold
            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()

        return sum(fold_dices) / len(fold_dices)

    except torch.OutOfMemoryError:
        print(f"CUDA OOM for Trial {trial.number}. Skipping.")
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        return 0.0 # Return poor result instead of crashing
    except Exception as e:
        print(f"Error in Trial {trial.number}: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--base-ch", type=int, default=32)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use SQLite for parallel trials
    db_path = os.path.join(args.out_dir, f"optuna_{args.model}.db")
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"
    
    study = optuna.create_study(
        study_name=f"hpo_{args.model}",
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(lambda trial: objective(trial, args, device), n_trials=args.n_trials)
    
    print(f"Best hyperparameters for {args.model}:", study.best_params)
    
    # Save best params
    with open(os.path.join(args.out_dir, f"best_hparams_{args.model}.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

if __name__ == "__main__":
    main()
