import os

import torch
from torch.optim import Adam

from training.losses import detection_loss
from utils.checkpoint import save_checkpoint


def train_detector(
    model,
    train_loader,
    val_loader=None,
    epochs: int = 30,
    lr: float = 1e-4,
    device=None,
    checkpoint_dir: str = "./checkpoints",
) -> None:
    """
    Training loop for HybridDetector on COCO-style detection data.

    Per epoch:
      1. Train — logs per-batch and per-epoch stats.
      2. (Optional) Validate — saves best.pt when val loss improves.
    After all epochs: saves last.pt unconditionally.

    Safety guarantees:
      - Batches with NaN loss are skipped (logged, not silently dropped).
      - Per-batch exceptions are caught; training continues with the next batch.
      - Gradients are clipped to max_norm=1.0 before every optimizer step.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    n_train = len(train_loader)

    for epoch in range(epochs):

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = epoch_cls = epoch_box = 0.0
        batches_ok = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                images = images.to(device)
                targets = [
                    {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                    for t in targets
                ]

                optimizer.zero_grad()
                logits, boxes = model(images)
                total, cls_l, box_l = detection_loss(logits, boxes, targets)

                # Skip batch if loss exploded to NaN
                if torch.isnan(total):
                    print(
                        f"  [WARNING] NaN loss at epoch {epoch+1} "
                        f"batch {batch_idx+1} — skipping"
                    )
                    continue

                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total.item()
                epoch_cls  += cls_l.item()
                epoch_box  += box_l.item()
                batches_ok += 1

                # Per-batch log (every batch, matches the format requested)
                print(
                    f"  Epoch [{epoch+1}/{epochs}]  "
                    f"Batch [{batch_idx+1}/{n_train}]  "
                    f"Loss: {total.item():.4f}  "
                    f"Cls: {cls_l.item():.4f}  "
                    f"Box: {box_l.item():.4f}"
                )

            except Exception as e:
                print(
                    f"  [ERROR] Skipping batch {batch_idx+1} at epoch {epoch+1}: {e}"
                )
                continue

        # Per-epoch summary
        if batches_ok > 0:
            avg_loss = epoch_loss / batches_ok
            print(
                f"\nEpoch [{epoch+1}/{epochs}] Summary  "
                f"Train Loss: {avg_loss:.4f}  "
                f"Cls: {epoch_cls/batches_ok:.4f}  "
                f"Box: {epoch_box/batches_ok:.4f}"
            )
        else:
            print(f"\nEpoch [{epoch+1}/{epochs}] — no valid batches.")
            avg_loss = float("nan")

        # ── Validation pass ───────────────────────────────────────────────────
        if val_loader is not None:
            val_loss = _validate(model, val_loader, device)
            print(f"Epoch [{epoch+1}/{epochs}] Val Loss: {val_loss:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch + 1, val_loss,
                    os.path.join(checkpoint_dir, "best.pt"),
                )
                print(
                    f"  [Checkpoint] best.pt saved  "
                    f"(val_loss={val_loss:.4f})\n"
                )
        else:
            print()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(
        model, optimizer, epochs, avg_loss,
        os.path.join(checkpoint_dir, "last.pt"),
    )
    print(f"Training complete. Checkpoints written to '{checkpoint_dir}/'")


def _validate(model, val_loader, device) -> float:
    """
    One pass over val_loader. Returns average detection loss.
    No gradients are computed. Model is restored to train mode on exit.
    """
    model.eval()
    total = 0.0
    batches_ok = 0

    with torch.no_grad():
        for images, targets in val_loader:
            try:
                images = images.to(device)
                targets = [
                    {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                    for t in targets
                ]
                logits, boxes = model(images)
                loss, _, _ = detection_loss(logits, boxes, targets)

                if torch.isnan(loss):
                    continue

                total += loss.item()
                batches_ok += 1

            except Exception as e:
                print(f"  [ERROR] Skipping val batch: {e}")
                continue

    model.train()
    return total / batches_ok if batches_ok > 0 else float("nan")
