import torch
from torch import nn, optim
import torch.nn.functional as F

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with Temperature Scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        return logits / self.temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, device='cuda'):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.BCEWithLogitsLoss().to(device)
        ece_criterion = ECELoss().to(device)

        # First: collect a subset of logits and labels
        logits_list = []
        labels_list = []
        MAX_SAMPLES = 1000000
        current_samples = 0
        
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                l = self.model(input)
                # Keep on CPU to avoid OOM
                logits_list.append(l.cpu())
                labels_list.append(label.cpu())
                
                current_samples += l.numel()
                if current_samples > MAX_SAMPLES * 2:
                    break
        
        # Concatenate and flatten
        logits = torch.cat(logits_list).flatten()
        labels = torch.cat(labels_list).flatten()
        
        # Subsample if needed
        if len(logits) > MAX_SAMPLES:
            idx = torch.randperm(len(logits))[:MAX_SAMPLES]
            logits = logits[idx]
            labels = labels[idx]
            
        # Move to GPU for optimization
        logits = logits.to(device)
        labels = labels.to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just to see if it's working)
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        sigmoid_logits = torch.sigmoid(logits)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (sigmoid_logits > bin_lower.item()) * (sigmoid_logits <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = labels[in_bin].float().mean()
                avg_confidence_in_bin = sigmoid_logits[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
