import torch
from torch.optim import Adam
from training.validate import validate
from training.losses import get_loss, detection_loss
from training.metrics import accuracy


def train(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=1e-4,
    device="cpu"
):
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = get_loss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )


def train_detector(
    model,
    train_loader,
    epochs:    int   = 10,
    lr:        float = 1e-4,
    device:    str   = "cpu",
    log_every: int   = 10,
) -> None:
    """
    Training loop for HybridDetector on COCO-style detection data.

    Expects train_loader to yield (images, targets) batches produced by
    COCODetectionDataset + dataset.coco_detection.collate_fn.

    Args:
        model:        HybridDetector instance
        train_loader: DataLoader with detection collate_fn
        epochs:       number of full passes over the training set
        lr:           Adam learning rate
        device:       "cpu" or "cuda"
        log_every:    print batch stats every N batches
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        epoch_cls  = 0.0
        epoch_box  = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # Individual box/label tensors inside targets are moved to device
            # inside detection_loss, keeping this loop device-agnostic.

            optimizer.zero_grad()

            logits, boxes = model(images)
            total, cls_l, box_l = detection_loss(logits, boxes, targets)

            total.backward()
            optimizer.step()

            epoch_loss += total.item()
            epoch_cls  += cls_l.item()
            epoch_box  += box_l.item()

            if (batch_idx + 1) % log_every == 0:
                print(
                    f"  [epoch {epoch+1}  batch {batch_idx+1:>4d}]  "
                    f"loss={total.item():.4f}  "
                    f"cls={cls_l.item():.4f}  "
                    f"box={box_l.item():.4f}"
                )

        n = len(train_loader)
        print(
            f"Epoch [{epoch+1}/{epochs}]  "
            f"avg_loss={epoch_loss/n:.4f}  "
            f"avg_cls={epoch_cls/n:.4f}  "
            f"avg_box={epoch_box/n:.4f}"
        )
