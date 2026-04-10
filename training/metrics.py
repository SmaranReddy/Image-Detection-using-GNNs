import torch


def accuracy(outputs, targets):
    """
    outputs: (B, num_classes)
    targets: (B)
    """
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)
