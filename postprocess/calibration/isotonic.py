import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibrator:
    def __init__(self, model):
        self.model = model
        self.ir = IsotonicRegression(out_of_bounds='clip')
        
    def __call__(self, x):
        # Forward pass through model + calibration
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        # Isotonic regression works on probabilities, but IR is piece-wise constant/linear.
        # It maps scalar -> scalar. We apply it to flattened probabilities.
        # Note: This is slow for segmentation if done pixel-wise on CPU.
        # We might need to optimize or use a GPU implementation if available (rare).
        
        orig_shape = probs.shape
        probs_flat = probs.detach().cpu().numpy().flatten()
        calibrated_probs = self.ir.transform(probs_flat)
        return torch.from_numpy(calibrated_probs.reshape(orig_shape)).to(probs.device)

    def fit(self, valid_loader, device='cuda'):
        self.model.eval()
        self.model.to(device)
        
        probs_list = []
        labels_list = []
        
        # Collect data
        # For segmentation, this is huge. We MUST subsample.
        MAX_SAMPLES = 1000000
        
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                probs = torch.sigmoid(logits)
                
                probs_list.append(probs.cpu().numpy().flatten())
                labels_list.append(label.cpu().numpy().flatten())
                
                if sum(len(p) for p in probs_list) > MAX_SAMPLES * 2:
                    break
                    
        all_probs = np.concatenate(probs_list)
        all_labels = np.concatenate(labels_list)
        
        if len(all_probs) > MAX_SAMPLES:
            idx = np.random.choice(len(all_probs), MAX_SAMPLES, replace=False)
            all_probs = all_probs[idx]
            all_labels = all_labels[idx]
            
        self.ir.fit(all_probs, all_labels)
        print("Isotonic Regression Fitted.")
        return self
