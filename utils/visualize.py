"""
Visualization utility for object detection predictions.

Draws predicted boxes (red) and ground-truth boxes (green) on a single image.
Predictions below a confidence threshold are silently dropped.

Usage example:
    from utils.visualize import visualize_predictions
    from dataset.coco_detection import DETECTION_CLASSES

    # model returns (logits, boxes) for a single image after unsqueeze(0)
    logits, boxes = model(image.unsqueeze(0))
    visualize_predictions(
        image       = image,                  # Tensor (C, H, W), normalized
        pred_logits = logits[0],              # (Q, C+1)
        pred_boxes  = boxes[0],               # (Q, 4)  cxcywh
        gt_boxes    = target["boxes"],        # (N, 4)  xyxy
        gt_labels   = target["labels"],       # (N,)
        class_names = DETECTION_CLASSES,      # foreground only, length C
    )
"""

from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.causal_caption import extract_top_detections, infer_relationships, generate_causal_caption
from utils.logger_utils import debug_print


# ImageNet statistics used during dataset normalization
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denormalize(image: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization and convert (C, H, W) tensor → (H, W, C) numpy array.
    Output values are clipped to [0, 1].
    """
    img = image.cpu().float() * _STD + _MEAN   # (C, H, W)
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()        # (H, W, C)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    (cx, cy, w, h) → (x_min, y_min, x_max, y_max)
    Input/output shape: (..., 4)
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - 0.5 * w,
                        cy - 0.5 * h,
                        cx + 0.5 * w,
                        cy + 0.5 * h], dim=-1)


def _to_pixels(boxes_norm: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Scale normalized [0, 1] boxes → pixel coordinates.
    boxes_norm: (..., 4) in (x_min, y_min, x_max, y_max) format
    """
    scale = torch.tensor([width, height, width, height],
                         dtype=boxes_norm.dtype)
    return boxes_norm * scale


def draw_relation_boxes(
    image,
    detections: List[dict],
    relations: List[dict],
    output_path: str,
    caption: str = "",
) -> None:
    """
    Draw detection boxes with relation arrows and save visualization.

    Args:
        image:      PIL Image or numpy array.
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        relations:  [{"subject": str, "predicate": str, "object": str,
                      "confidence": float, "subject_box": [...], "object_box": [...]}, ...]
        output_path: Path to save the visualization.
        caption:    Optional caption text to overlay.
    """
    try:
        from PIL import ImageDraw, ImageFont, Image as PILImage
    except ImportError:
        debug_print("[visualize] PIL not available \u2014 skipping draw_relation_boxes")
        return

    if isinstance(image, torch.Tensor):
        img_np = _denormalize(image)
        img_np = (img_np * 255).clip(0, 255).astype("uint8")
        pil = PILImage.fromarray(img_np)
    else:
        pil = image.convert("RGB") if hasattr(image, "convert") else PILImage.fromarray(image)

    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    colors = [
        "#FF4444", "#4488FF", "#44BB44", "#FF8800", "#AA44FF",
        "#00CCCC", "#FF44FF", "#888800",
    ]

    # Draw detection boxes
    for i, d in enumerate(detections):
        color = colors[i % len(colors)]
        box = d["box"]
        draw.rectangle(box, outline=color, width=2)
        label_text = f"{d['label']} {d.get('score', 0):.2f}"
        bbox = draw.textbbox((box[0], box[1] - 16), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1] - 16), label_text, fill="white", font=font)

    # Draw relation arrows
    for r in relations:
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx = (subj_box[0] + subj_box[2]) / 2
            sy = (subj_box[1] + subj_box[3]) / 2
            ox = (obj_box[0] + obj_box[2]) / 2
            oy = (obj_box[1] + obj_box[3]) / 2
            draw.line([(sx, sy), (ox, oy)], fill="#FFDD00", width=2)
            mx = (sx + ox) / 2
            my = (sy + oy) / 2 - 12
            rel_text = f"{r['predicate']} ({r.get('confidence', 0):.2f})"
            text_bbox = draw.textbbox((mx, my), rel_text, font=small_font)
            draw.rectangle(text_bbox, fill="#FFDD00")
            draw.text((mx, my), rel_text, fill="black", font=small_font)

    # Caption overlay
    if caption:
        w, h = pil.size
        lines = []
        words = caption.split()
        current = ""
        for word in words:
            test = current + " " + word if current else word
            if draw.textbbox((0, 0), test, font=font)[2] < w - 20:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        y = h - 20 - len(lines) * 18
        for line in lines:
            tb = draw.textbbox((10, y), line, font=font)
            draw.rectangle(tb, fill=(0, 0, 0, 180))
            draw.text((10, y), line, fill="white", font=font)
            y += 18

    pil.save(output_path)
    debug_print(f"[visualize] Saved relation visualization to {output_path}")


def visualize_predictions(
    image:       torch.Tensor,
    pred_logits: torch.Tensor,
    pred_boxes:  torch.Tensor,
    gt_boxes:    torch.Tensor,
    gt_labels:   torch.Tensor,
    class_names: List[str],
    threshold:   float = 0.5,
    save_path:   Optional[str] = None,
) -> None:
    """
    Draw predicted and ground-truth bounding boxes on a single image.

    Args:
        image:       Tensor (C, H, W) — ImageNet-normalized, as returned by the dataset.
        pred_logits: Tensor (Q, C+1)  — raw class scores; index 0 = background.
        pred_boxes:  Tensor (Q, 4)    — predicted boxes in (cx, cy, w, h), normalized [0,1].
        gt_boxes:    Tensor (N, 4)    — ground-truth boxes in (x_min, y_min, x_max, y_max), normalized.
        gt_labels:   Tensor (N,)      — ground-truth labels, 1-indexed (0 = background, not used here).
        class_names: list of C foreground class names, e.g. ["person", "car", ...].
                     Indexed as class_names[label - 1] since labels start at 1.
        threshold:   Minimum foreground confidence to display a prediction (default 0.5).
        save_path:   If given, save the figure to this path instead of calling plt.show().
    """
    # ── Prepare image ─────────────────────────────────────────────────────────
    img_np = _denormalize(image)               # (H, W, C), float32 in [0, 1]
    H, W   = img_np.shape[:2]

    # ── Filter predictions by confidence ──────────────────────────────────────
    # softmax over all C+1 classes, then examine foreground columns only
    probs      = torch.softmax(pred_logits.float(), dim=-1)   # (Q, C+1)
    fg_probs   = probs[:, 1:]                                  # (Q, C)  drop background
    confidence, pred_class_idx = fg_probs.max(dim=-1)         # (Q,), (Q,)
    # pred_class_idx is 0-indexed into foreground → actual label = pred_class_idx + 1

    keep = confidence > threshold                              # (Q,) boolean mask

    # ── Convert predicted boxes to pixel xyxy ─────────────────────────────────
    pred_xyxy_norm = _cxcywh_to_xyxy(pred_boxes.cpu().float())  # (Q, 4) normalized xyxy
    pred_xyxy_px   = _to_pixels(pred_xyxy_norm, W, H)           # (Q, 4) pixel xyxy

    # ── Convert ground-truth boxes to pixel xyxy ──────────────────────────────
    gt_xyxy_px = _to_pixels(gt_boxes.cpu().float(), W, H)       # (N, 4) pixel xyxy

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    ax.axis("off")

    # Ground-truth boxes — GREEN
    for i in range(gt_xyxy_px.shape[0]):
        x1, y1, x2, y2 = gt_xyxy_px[i].tolist()
        rect = patches.Rectangle(
            xy       = (x1, y1),
            width    = x2 - x1,
            height   = y2 - y1,
            linewidth = 2,
            edgecolor = "limegreen",
            facecolor = "none",
        )
        ax.add_patch(rect)

        # GT label (no score — ground truth is certain)
        label_idx = int(gt_labels[i].item()) - 1   # convert 1-indexed → list index
        if 0 <= label_idx < len(class_names):
            ax.text(
                x1, y1 - 4,
                class_names[label_idx],
                color      = "limegreen",
                fontsize   = 9,
                fontweight = "bold",
                clip_on    = True,
            )

    # Predicted boxes — RED (confidence-filtered only)
    for i in range(pred_xyxy_px.shape[0]):
        if not keep[i]:
            continue

        x1, y1, x2, y2 = pred_xyxy_px[i].tolist()
        score = confidence[i].item()
        cidx  = int(pred_class_idx[i].item())   # 0-indexed into foreground

        rect = patches.Rectangle(
            xy        = (x1, y1),
            width     = x2 - x1,
            height    = y2 - y1,
            linewidth = 2,
            edgecolor = "red",
            facecolor = "none",
        )
        ax.add_patch(rect)

        label_text = (
            f"{class_names[cidx]}  {score:.2f}"
            if cidx < len(class_names)
            else f"cls{cidx + 1}  {score:.2f}"
        )
        ax.text(
            x1, max(y2 + 12, 12),   # place below the box to avoid overlap with GT label
            label_text,
            color      = "red",
            fontsize   = 9,
            fontweight = "bold",
            clip_on    = True,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        patches.Patch(edgecolor="limegreen", facecolor="none", linewidth=2, label="Ground truth"),
        patches.Patch(edgecolor="red",       facecolor="none", linewidth=2, label=f"Prediction (conf > {threshold})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.8, fancybox=False)

    # ── Causal caption ────────────────────────────────────────────────────────
    detections   = extract_top_detections(pred_logits, pred_boxes, class_names, threshold)
    relationships = infer_relationships(detections)
    caption       = generate_causal_caption(detections, relationships)

    fig.text(
        0.5, -0.02,
        caption,
        ha        = "center",
        va        = "top",
        fontsize  = 10,
        wrap      = True,
        color     = "black",
        style     = "italic",
        bbox      = dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                         edgecolor="gray", alpha=0.85),
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
