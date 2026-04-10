"""
Greedy prediction-to-ground-truth matcher for object detection.

Strategy: for each GT box (in order), assign the available prediction
with the highest IoU.  No prediction is reused.  Unmatched predictions
are handled by the loss function (assigned to background class).
"""

from typing import Tuple

import torch


def compute_iou_matrix(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.
    Both must be in (x_min, y_min, x_max, y_max) format.

    Args:
        boxes_a: (M, 4)
        boxes_b: (N, 4)

    Returns:
        iou: (M, N)
    """
    # Broadcast: (M, 1, 4) vs (1, N, 4)
    a = boxes_a.unsqueeze(1)  # (M, 1, 4)
    b = boxes_b.unsqueeze(0)  # (1, N, 4)

    # Intersection rectangle
    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_area = (
        (inter_x2 - inter_x1).clamp(min=0)
        * (inter_y2 - inter_y1).clamp(min=0)
    )  # (M, N)

    area_a = (
        (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    ).unsqueeze(1)  # (M, 1)

    area_b = (
        (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    ).unsqueeze(0)  # (1, N)

    union_area = area_a + area_b - inter_area          # (M, N)
    iou = inter_area / union_area.clamp(min=1e-6)      # (M, N)

    return iou


def greedy_match(
    pred_boxes: torch.Tensor,
    gt_boxes:   torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each GT box, greedily assign the prediction with the highest IoU
    that has not already been taken.

    Both inputs must be in (x_min, y_min, x_max, y_max) format.
    Always pass *detached* pred_boxes — matching is not differentiable.

    Args:
        pred_boxes: (Q, 4)  predicted boxes, detached, xyxy
        gt_boxes:   (N, 4)  ground-truth boxes, xyxy

    Returns:
        pred_indices: LongTensor (M,)  matched prediction slots
        gt_indices:   LongTensor (M,)  corresponding GT indices
        where M = min(actual matches, min(Q, N))
    """
    N = gt_boxes.shape[0]
    Q = pred_boxes.shape[0]

    if N == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty

    iou = compute_iou_matrix(pred_boxes, gt_boxes)  # (Q, N)

    assigned:     set  = set()   # prediction indices already used
    pred_indices: list = []
    gt_indices:   list = []

    for gt_i in range(N):
        if len(assigned) == Q:
            break  # every prediction slot is taken

        # Clone the column so masking doesn't modify the original iou matrix
        scores = iou[:, gt_i].clone()
        for used in assigned:
            scores[used] = -1.0  # exclude already-assigned predictions

        best = int(scores.argmax().item())
        assigned.add(best)
        pred_indices.append(best)
        gt_indices.append(gt_i)

    return (
        torch.tensor(pred_indices, dtype=torch.long),
        torch.tensor(gt_indices,   dtype=torch.long),
    )
