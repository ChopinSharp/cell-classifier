import torch
from torch import nn, optim
from torch.nn import functional as F


def compute_temperature(model, val_loader=None, verbose=False):
    temperature = torch.tensor(1.5, requires_grad=True)

    # Declare criterions
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = _ECELoss()

    # Collect all the logits and labels of the validation set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
        logits_val = torch.cat(tuple(logits_list))
        labels_val = torch.cat(tuple(labels_list))

    # Optimize the temperature w.r.t. NLL
    def __update():
        loss = nll_criterion(logits_val.div(temperature), labels_val)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    optimizer.step(__update)

    # Stop tracking gradient for temperature parameter
    temperature.requires_grad_(False)

    # Print out stats if verbose flag is set
    if verbose:
        nll_before_scaling = nll_criterion(logits_val, labels_val).item()
        ece_before_scaling = ece_criterion(logits_val, labels_val).item()
        nll_after_scaling = nll_criterion(logits_val.div(temperature), labels_val).item()
        ece_after_scaling = ece_criterion(logits_val.div(temperature), labels_val).item()
        print('+ Optimal temperature: %.3f' % temperature.item())
        print('+ Before temperature scaling - NLL: %.3f, ECE: %.3f' % (nll_before_scaling, ece_before_scaling))
        print('+ After temperature scaling - NLL: %.3f, ECE: %.3f' % (nll_after_scaling, ece_after_scaling))

    return temperature.item()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece



