import torch


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str) -> None:
    """
    Save model and optimizer state to disk.

    Args:
        model:     Any nn.Module.
        optimizer: The optimizer used during training.
        epoch:     Current epoch number (1-indexed, stored for resuming).
        loss:      Loss value at save time (train or val, for reference).
        path:      Full file path, e.g. "./checkpoints/best.pt".
    """
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":                 loss,
        },
        path,
    )


def load_checkpoint(
    model,
    path: str,
    optimizer=None,
    device=None,
):
    """
    Restore model (and optionally optimizer) state from a checkpoint file.

    Args:
        model:     nn.Module to load weights into.
        path:      Path to the .pt file written by save_checkpoint().
        optimizer: If provided, optimizer state is also restored (for resuming training).
        device:    Device to map tensors to on load.

    Returns:
        epoch: int   — epoch stored in the checkpoint
        loss:  float — loss stored in the checkpoint
    """
    checkpoint = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]
