from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.coco_detection import cxcywh_to_xyxy
from training.matcher import greedy_match


# Module-level instance — CrossEntropyLoss is stateless, safe to reuse.
_cls_criterion = nn.CrossEntropyLoss()


def detection_loss(
    pred_logits:     torch.Tensor,
    pred_boxes:      torch.Tensor,
    targets:         List[dict],
    box_loss_weight: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined classification + box regression loss for one batch.

    Per image:
      1. Convert pred_boxes (cx, cy, w, h) → (x_min, y_min, x_max, y_max).
      2. Greedy-match predictions to GT boxes using IoU (detached).
      3. Classification loss (CrossEntropy) over all Q queries:
           matched queries   → GT foreground label (1 .. C)
           unmatched queries → background label (0)
      4. Box L1 loss on matched queries only, in xyxy space.

    Final: total = cls_loss + box_loss_weight × box_loss

    Args:
        pred_logits:     (B, Q, C+1)  raw class scores; index 0 = background
        pred_boxes:      (B, Q, 4)    (cx, cy, w, h) in [0, 1]
        targets:         list of B dicts, each with
                             "boxes"  FloatTensor (N, 4)  xyxy in [0, 1]
                             "labels" LongTensor  (N,)    1-indexed
        box_loss_weight: scalar weight applied to the box loss term

    Returns:
        total_loss: scalar Tensor — backpropagate this
        cls_loss:   scalar Tensor — detached, for logging
        box_loss:   scalar Tensor — detached, for logging
    """
    B, Q, _ = pred_logits.shape
    device  = pred_logits.device

    per_image_cls: List[torch.Tensor] = []
    per_image_box: List[torch.Tensor] = []

    for i in range(B):
        gt_boxes  = targets[i]["boxes"]    # (N, 4) xyxy  — already on device
        gt_labels = targets[i]["labels"]   # (N,)         — already on device

        N = gt_boxes.shape[0]

        # All Q queries default to background; matched ones are overwritten below
        cls_targets = torch.zeros(Q, dtype=torch.long, device=device)

        if N > 0:
            # Detach before matching so assignment has no gradient path
            pred_xyxy_det = cxcywh_to_xyxy(pred_boxes[i]).detach()  # (Q, 4)
            pred_idx, gt_idx = greedy_match(pred_xyxy_det, gt_boxes)

            if len(pred_idx) > 0:
                # Overwrite matched slots with the corresponding GT class
                cls_targets[pred_idx] = gt_labels[gt_idx]

                # Box L1 — gradients flow through the non-detached conversion
                pred_xyxy = cxcywh_to_xyxy(pred_boxes[i])  # (Q, 4)
                box_l1 = F.l1_loss(
                    pred_xyxy[pred_idx],   # (M, 4)  predicted xyxy
                    gt_boxes[gt_idx],      # (M, 4)  target xyxy
                    reduction="mean",
                )
                per_image_box.append(box_l1)

        # CrossEntropy across all Q queries for this image
        per_image_cls.append(_cls_criterion(pred_logits[i], cls_targets))

    cls_loss = torch.stack(per_image_cls).mean()

    # If no image in the batch had any matched box, box_loss = 0 (no gradient)
    box_loss = (
        torch.stack(per_image_box).mean()
        if per_image_box
        else torch.tensor(0.0, device=device)
    )

    total_loss = cls_loss + box_loss_weight * box_loss

    return total_loss, cls_loss.detach(), box_loss.detach()
