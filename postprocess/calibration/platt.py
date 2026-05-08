import torch
from torch import nn, optim

class PlattScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return self.a * logits + self.b

    def fit(self, valid_loader, device='cuda'):
        self.to(device)
        optimizer = optim.LBFGS([self.a, self.b], lr=0.01, max_iter=50)
        criterion = nn.BCEWithLogitsLoss()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                # We need raw logits from model, assumption is 'input' here
                # actually needs to be processed by model first if passing cleaner.
                # BUT, PlattScaling usually wraps model or takes precomputed logits.
                # Let's assume we pass PRECOMPUTED logits to fit?
                # Or we wrap a model like TemperatureScaling.
                pass
        
        # To keep API consistent, let's assume valid_loader yields (input, label)
        # and we need a model to get logits. 
        # Actually, let's make PlattScaling take a LIST of precomputed logits/labels for fitting,
        # or implement it similarly to TemperatureScaling wrapping a model.
        return self

class ModelWithPlatt(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.platt = PlattScaling()

    def forward(self, x):
        logits = self.model(x)
        return self.platt(logits)
    
    def fit(self, valid_loader, device='cuda'):
        self.cuda()
        self.platt.to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        logits_list = []
        labels_list = []
        MAX_SAMPLES = 1000000
        current_samples = 0
        
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                l = self.model(input)
                # Keep on CPU
                logits_list.append(l.cpu())
                labels_list.append(label.cpu())
                
                current_samples += l.numel()
                if current_samples > MAX_SAMPLES * 2:
                    break
                
        # Concatenate and flatten
        logits = torch.cat(logits_list).flatten()
        labels = torch.cat(labels_list).flatten()
        
        # Subsample
        if len(logits) > MAX_SAMPLES:
            idx = torch.randperm(len(logits))[:MAX_SAMPLES]
            logits = logits[idx]
            labels = labels[idx]
            
        # Move to GPU
        logits = logits.to(device)
        labels = labels.to(device)
        
        optimizer = optim.LBFGS(self.platt.parameters(), lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.platt(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Platt Scaling Fitted: a={self.platt.a.item():.3f}, b={self.platt.b.item():.3f}")
        return self
