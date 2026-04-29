"""
YOLOv11 detector using Ultralytics — drop-in replacement for the old detection stage.

Public API:
    load_model(device=None)            -> YOLO model
    run_inference(model, image)        -> raw Ultralytics Results object
    format_detections(raw, conf, topk) -> [{"label": str, "box": [x1,y1,x2,y2], "score": float}]
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def load_model(device: Optional[str] = None):
    """Load YOLOv11m and move it to *device* (auto-selects CUDA/CPU when None)."""
    from ultralytics import YOLO
    import torch

    model = YOLO("yolo11m.pt")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    return model


def run_inference(model, image: Union[str, np.ndarray, "PIL.Image.Image"]):
    """
    Run the model on *image* and return the raw Ultralytics Results list.

    Accepts a file path (str), a NumPy array (HWC BGR or RGB), or a PIL image.
    Returns the raw list so callers can inspect it before formatting.
    """
    results = model(image)
    return results


def format_detections(
    raw_results,
    conf_thres: float = 0.6,
    top_k: int = 10,
) -> List[dict]:
    """
    Convert raw Ultralytics results to the canonical detection format.

    Returns:
        [{"label": str, "box": [x1, y1, x2, y2], "score": float}, ...]

    Boxes are pixel-space xyxy.  At most *top_k* detections are returned,
    sorted by score descending, after filtering by *conf_thres*.
    """
    # Ultralytics returns a list; we always operate on the first frame.
    results = raw_results[0] if isinstance(raw_results, (list, tuple)) else raw_results

    boxes   = results.boxes.xyxy    # (N, 4) tensor
    scores  = results.boxes.conf    # (N,)   tensor
    classes = results.boxes.cls     # (N,)   tensor
    names   = results.names         # {int: str}

    # Move to CPU numpy for pure-Python processing.
    boxes_np   = boxes.cpu().numpy()
    scores_np  = scores.cpu().numpy()
    classes_np = classes.cpu().numpy()

    if len(scores_np) == 0:
        return []

    # Confidence filter.
    keep = scores_np >= conf_thres
    boxes_np   = boxes_np[keep]
    scores_np  = scores_np[keep]
    classes_np = classes_np[keep]

    if len(scores_np) == 0:
        return []

    # Sort descending by score and keep top-k.
    order      = np.argsort(scores_np)[::-1][:top_k]
    boxes_np   = boxes_np[order]
    scores_np  = scores_np[order]
    classes_np = classes_np[order]

    detections: List[dict] = []
    for i in range(len(scores_np)):
        box = boxes_np[i]

        if box.shape[0] != 4:
            continue

        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        # Clamp within image bounds if the result object carries orig_shape.
        if hasattr(results, "orig_shape") and results.orig_shape is not None:
            h, w = results.orig_shape[:2]
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)

        label = names[int(classes_np[i])]

        detections.append({
            "label": label,
            "box":   [x1, y1, x2, y2],
            "score": float(scores_np[i]),
        })

    return detections
