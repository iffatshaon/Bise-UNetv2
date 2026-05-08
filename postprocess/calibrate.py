import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.dataset_prep import list_image_mask_pairs, split_pairs, get_loaders
from train.models.factory import get_model
from postprocess.calibration import ModelWithTemperature, IsotonicCalibrator, PlattScaling, ModelWithPlatt

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def make_reliability_diagram(confidences, accuracies, ece, title="Reliability Diagram", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(confidences, accuracies, marker='o', label=f'Model (ECE={ece:.4f})')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_calibration_curve(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        if np.sum(in_bin) > 0:
            accuracies.append(np.mean(labels[in_bin]))
            confidences.append(np.mean(probs[in_bin]))
            
    return confidences, accuracies

def collect_logits_labels(model, loader, device, max_samples=100000):
    all_logits = []
    all_labels = []
    current_samples = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            
            all_logits.append(logits.cpu())
            all_labels.append(yb.cpu())
            
            current_samples += xb.size(0) * xb.size(2) * xb.size(3) # approx pixels
            if current_samples > max_samples:
                break
                
    return torch.cat(all_logits), torch.cat(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. unet)')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--out', type=str, default='calibration_results', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading {args.model} from {args.ckpt}...")
    # Load Model using Factory
    # Need to know config? Assume standard or load from ckpt args if possible.
    # For now, just load standard.
    model = get_model(args.model, base_ch=32).to(device)
    
    # Load state
    state = torch.load(args.ckpt, map_location='cpu')
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
        
    model.eval()

    print("Loading Data...")
    img_dir = os.path.join(args.data, "images")
    msk_dir = os.path.join(args.data, "masks")
    pairs = list_image_mask_pairs(img_dir, msk_dir)
    # Just need val set, but split needs seed to match training... 
    # Use seed 42 as default from train.py
    tr, va, te = split_pairs(pairs, seed=42)
    
    # Validation Loader
    loader_config = {"IMG_SIZE": 256, "BATCH": 8, "AUG_FLIP_H": 0.0, "AUG_FLIP_V": 0.0, "AUG_ROTATE_P": 0.0, "AUG_BRIGHTC_P": 0.0}
    _, val_loader, _ = get_loaders(tr, va, te, loader_config)

    # 1. Uncalibrated
    print("Evaluating Baseline...")
    logits, labels = collect_logits_labels(model, val_loader, device)
    probs_base = torch.sigmoid(logits).flatten().numpy()
    labels_flat = labels.flatten().numpy()
    
    ece_base = compute_ece(probs_base, labels_flat)
    confs, accs = compute_calibration_curve(probs_base, labels_flat)
    make_reliability_diagram(confs, accs, ece_base, "Baseline", os.path.join(args.out, "baseline.png"))
    print(f"Baseline ECE: {ece_base:.4f}")
    
    import pickle
    
    # 2. Temperature Scaling
    print("Running Temperature Scaling...")
    ts_model = ModelWithTemperature(model)
    ts_model.set_temperature(val_loader, device=device)
    
    # Save TS
    torch.save(ts_model.state_dict(), os.path.join(args.out, "calibrated_ts.pt"))
    print(f"Saved Temperature Scaled model to {os.path.join(args.out, 'calibrated_ts.pt')}")
    
    # Re-eval ts
    logits_ts = ts_model.temperature_scale(logits.to(device)).cpu()
    probs_ts = torch.sigmoid(logits_ts).flatten().detach().numpy()
    ece_ts = compute_ece(probs_ts, labels_flat)
    confs, accs = compute_calibration_curve(probs_ts, labels_flat)
    make_reliability_diagram(confs, accs, ece_ts, "Temperature Scaling", os.path.join(args.out, "temperature_scaling.png"))
    print(f"TS ECE: {ece_ts:.4f}")

    # 3. Isotonic
    print("Running Isotonic Regression...")
    iso_model = IsotonicCalibrator(model)
    iso_model.fit(val_loader, device=device)
    
    # Save Isotonic
    with open(os.path.join(args.out, "isotonic_calibrator.pkl"), "wb") as f:
        pickle.dump(iso_model.ir, f)
    print(f"Saved Isotonic Regression model to {os.path.join(args.out, 'isotonic_calibrator.pkl')}")

    # Eval Isotonic
    orig_shape = logits.shape
    probs_for_iso = torch.sigmoid(logits).flatten().numpy()
    probs_iso = iso_model.ir.transform(probs_for_iso)
    ece_iso = compute_ece(probs_iso, labels_flat)
    confs, accs = compute_calibration_curve(probs_iso, labels_flat)
    make_reliability_diagram(confs, accs, ece_iso, "Isotonic Regression", os.path.join(args.out, "isotonic.png"))
    print(f"Isotonic ECE: {ece_iso:.4f}")
    
    # 4. Platt Scaling
    print("Running Platt Scaling...")
    platt_model = ModelWithPlatt(model)
    platt_model.fit(val_loader, device=device)
    
    # Save Platt
    torch.save(platt_model.state_dict(), os.path.join(args.out, "calibrated_platt.pt"))
    print(f"Saved Platt Scaled model to {os.path.join(args.out, 'calibrated_platt.pt')}")
    
    # Eval Platt
    logits_platt = platt_model.platt(logits.to(device)).cpu()
    probs_platt = torch.sigmoid(logits_platt).flatten().detach().numpy()
    ece_platt = compute_ece(probs_platt, labels_flat)
    confs, accs = compute_calibration_curve(probs_platt, labels_flat)
    make_reliability_diagram(confs, accs, ece_platt, "Platt Scaling", os.path.join(args.out, "platt.png"))
    print(f"Platt ECE: {ece_platt:.4f}")
    
    print(f"Calibration Complete. Results saved to {args.out}")

if __name__ == "__main__":
    main()
