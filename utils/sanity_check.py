"""
Sanity checker for HybridDetector.

Usage:
    from utils.sanity_check import sanity_check_detector
    sanity_check_detector(model, val_loader, device, class_names)
"""

import math

import torch
import torch.nn.functional as F

from training.losses import detection_loss


_CONF_THRESHOLD  = 0.30   # minimum foreground confidence to count as a detection
_MEAN_CONF_FLOOR = 0.10   # below this → model is effectively untrained


def sanity_check_detector(
    model,
    val_loader,
    device,
    class_names,
    num_batches: int = 8,
) -> None:
    """
    Evaluate whether a trained HybridDetector is functioning correctly.

    Runs inference on up to `num_batches` validation batches and reports:
      - Average detection loss
      - Foreground confidence statistics
      - Class diversity (unique classes predicted above threshold)
      - Bounding box validity

    Args:
        model:       HybridDetector instance in eval mode, already on `device`.
        val_loader:  DataLoader yielding (images, List[target_dict]).
        device:      torch.device to run inference on.
        class_names: List of foreground class name strings, length == num_classes.
                     Index 0 corresponds to the first foreground class (label 1).
        num_batches: Number of batches to evaluate (default 8, max ~10 recommended).
    """
    model.eval()

    loss_sum     = 0.0
    loss_batches = 0

    conf_chunks:   list[torch.Tensor] = []
    cls_chunks:    list[torch.Tensor] = []
    invalid_count = 0
    total_queries = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            images  = images.to(device)
            targets = [
                {"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                for t in targets
            ]

            # ── Forward pass ──────────────────────────────────────────────────
            logits, boxes = model(images)   # (B, Q, C+1)  /  (B, Q, 4)

            # ── A. Loss ───────────────────────────────────────────────────────
            try:
                loss, _, _ = detection_loss(logits, boxes, targets)
                if not math.isnan(loss.item()):
                    loss_sum     += loss.item()
                    loss_batches += 1
            except Exception:
                pass   # malformed targets on this batch; skip loss only

            # ── B. Confidence stats ───────────────────────────────────────────
            probs    = F.softmax(logits, dim=-1)    # (B, Q, C+1)
            fg_probs = probs[..., 1:]               # (B, Q, C)  — drop background
            conf     = fg_probs.max(dim=-1).values  # (B, Q)     — best fg score per query
            conf_chunks.append(conf.cpu().flatten())

            # ── C. Class diversity ────────────────────────────────────────────
            pred_cls  = fg_probs.argmax(dim=-1)     # (B, Q)  0-indexed into fg
            above_thr = conf > _CONF_THRESHOLD      # (B, Q)  bool mask
            if above_thr.any():
                cls_chunks.append(pred_cls[above_thr].cpu())

            # ── D. Box sanity ─────────────────────────────────────────────────
            # boxes are produced by sigmoid + clamp in the head, so out-of-range
            # values would indicate a numerical bug upstream.
            bad_mask       = (boxes < 0.0) | (boxes > 1.0)      # (B, Q, 4)
            invalid_count += int(bad_mask.any(dim=-1).sum().item())
            total_queries += boxes.shape[0] * boxes.shape[1]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    conf_all  = torch.cat(conf_chunks) if conf_chunks else torch.zeros(1)
    mean_conf = conf_all.mean().item()
    max_conf  = conf_all.max().item()
    avg_loss  = loss_sum / loss_batches if loss_batches > 0 else float("nan")

    if cls_chunks:
        all_pred = torch.cat(cls_chunks)
        unique_ids   = sorted(all_pred.unique().tolist())
        unique_names = [
            class_names[int(i)] if int(i) < len(class_names) else f"class_{int(i)}"
            for i in unique_ids
        ]
    else:
        unique_ids   = []
        unique_names = []

    # ── Print structured results ───────────────────────────────────────────────
    print("\n[Sanity Check]")
    if loss_batches > 0:
        print(f"  Avg Loss        : {avg_loss:.4f}")
    else:
        print(f"  Avg Loss        : N/A")
    print(f"  Mean Confidence : {mean_conf:.4f}")
    print(f"  Max Confidence  : {max_conf:.4f}")
    if unique_names:
        print(f"  Unique Classes  : {unique_names}")
    else:
        print(f"  Unique Classes  : none above threshold ({_CONF_THRESHOLD})")
    print(f"  Invalid Boxes   : {invalid_count} / {total_queries}")

    # ── Failure diagnostics ────────────────────────────────────────────────────
    failures: list[str] = []

    if mean_conf < _MEAN_CONF_FLOOR:
        failures.append(
            f"  [WARNING] Mean confidence {mean_conf:.4f} < {_MEAN_CONF_FLOOR}"
            f" — model is likely untrained."
        )

    if max_conf < _CONF_THRESHOLD:
        failures.append(
            f"  [WARNING] No prediction exceeds confidence threshold"
            f" {_CONF_THRESHOLD} — model predicts only background (dead model)."
        )

    if len(unique_ids) == 1:
        failures.append(
            f"  [WARNING] Only one class predicted: '{unique_names[0]}'"
            f" — possible class collapse."
        )

    if invalid_count > 0:
        failures.append(
            f"  [WARNING] {invalid_count} boxes have coordinates outside [0, 1]"
            f" — possible bug in detection head."
        )

    if failures:
        print()
        for msg in failures:
            print(msg)

    # ── Verdict ────────────────────────────────────────────────────────────────
    passed = (
        mean_conf  >= _MEAN_CONF_FLOOR
        and max_conf   >= _CONF_THRESHOLD
        and len(unique_ids) != 1
        and invalid_count == 0
    )

    print("\n[Verdict]")
    if passed:
        print("  Model is trained sufficiently for demonstration.")
    else:
        print("  Model likely not trained properly.")
    print()
