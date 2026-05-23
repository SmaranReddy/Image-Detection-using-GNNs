"""
CLIP-based semantic verification for YOLO detections.

Pipeline insertion point:
    YOLO detection
    → CLIP semantic verification (this module)
    → verified grounded objects
    → relation extraction + caption grounding

Verifies each detection by comparing CLIP similarity of the
detection crop against the predicted label vs competing labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from utils.clip_scorer import get_clip_scorer

# ---------------------------------------------------------------------------
# Competing label pools used for semantic comparison
# ---------------------------------------------------------------------------

_COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

# Hallucination-prone objects that need strong CLIP evidence to survive
_COMMON_HALLUCINATIONS: frozenset = frozenset({
    "cell phone", "remote", "bottle", "cup", "sports ball",
})

# Core objects that should almost never be rejected when YOLO confidence is high
_HIGH_CONFIDENCE_OBJECTS: frozenset = frozenset({
    "person", "bicycle", "chair", "dog", "car",
})

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# CLIP cosine similarity thresholds (higher = stricter)
_CLIP_THRESHOLD_DEFAULT: float = 0.18
_CLIP_THRESHOLD_HALLUCINATION: float = 0.25
_CLIP_THRESHOLD_HIGH_CONF: float = 0.12

# Combined trust score thresholds: 0.5 * yolo_conf + 0.5 * clip_sim
_TRUST_THRESHOLD_DEFAULT: float = 0.35
_TRUST_THRESHOLD_HALLUCINATION: float = 0.45
_TRUST_THRESHOLD_HIGH_CONF: float = 0.30

# When another label's CLIP similarity exceeds the predicted label by more than
# this margin, the detection is considered a false positive regardless of threshold.
_FALSE_POSITIVE_MARGIN: float = 0.15

# ---------------------------------------------------------------------------
# Text embedding cache (all COCO classes, computed once)
# ---------------------------------------------------------------------------

_coco_text_embeddings: Optional[torch.Tensor] = None


def _get_coco_text_embeddings() -> torch.Tensor:
    """Compute and cache CLIP text embeddings for all COCO class names."""
    global _coco_text_embeddings
    if _coco_text_embeddings is not None:
        return _coco_text_embeddings

    scorer = get_clip_scorer()
    texts = [f"a photo of a {label}" for label in _COCO_CLASSES]
    _coco_text_embeddings = scorer.encode_texts(texts)
    return _coco_text_embeddings


def _compute_clip_similarities(
    crop_embedding: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> Dict[str, float]:
    """Compute cosine similarity between crop embedding and all text embeddings.

    Returns dict mapping COCO label → similarity score.
    """
    similarities = crop_embedding @ text_embeddings.T  # (N,)
    sim_values = similarities.cpu().tolist()
    return {
        label: sim_values[i]
        for i, label in enumerate(_COCO_CLASSES)
    }


def _get_thresholds(label: str) -> Tuple[float, float]:
    """Get CLIP and trust thresholds for a given label."""
    if label in _COMMON_HALLUCINATIONS:
        return _CLIP_THRESHOLD_HALLUCINATION, _TRUST_THRESHOLD_HALLUCINATION
    if label in _HIGH_CONFIDENCE_OBJECTS:
        return _CLIP_THRESHOLD_HIGH_CONF, _TRUST_THRESHOLD_HIGH_CONF
    return _CLIP_THRESHOLD_DEFAULT, _TRUST_THRESHOLD_DEFAULT


def _verify_single_detection(
    det: Dict,
    crop_embedding: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[bool, float, Dict[str, float]]:
    """Verify a single detection using CLIP semantic consistency.

    Args:
        det: YOLO detection dict {"label", "box", "score"}
        crop_embedding: CLIP embedding of the detection crop
        text_embeddings: CLIP text embeddings for all COCO classes
        image_size: (width, height) of the original image

    Returns:
        (keep, trust_score, all_similarities)
    """
    yolo_conf: float = det["score"]
    pred_label: str = det["label"]
    box: List[float] = det["box"]
    x1, y1, x2, y2 = box
    img_w, img_h = image_size

    clip_similarities = _compute_clip_similarities(crop_embedding, text_embeddings)

    pred_sim = clip_similarities.get(pred_label, 0.0)

    # Find the best-matching COCO label
    best_label = max(clip_similarities, key=lambda k: clip_similarities[k])
    best_sim = clip_similarities[best_label]

    # Normalize CLIP similarity to [0, 1] for trust score computation
    pred_sim_norm = max(0.0, pred_sim)

    trust_score = 0.5 * yolo_conf + 0.5 * pred_sim_norm

    clip_thresh, trust_thresh = _get_thresholds(pred_label)

    # Compute box area to detect tiny objects
    box_area = (x2 - x1) * (y2 - y1)
    image_area = img_w * img_h
    area_ratio = box_area / max(image_area, 1)

    # ── Decision logic ──────────────────────────────────────────────
    # Rule 1: Predicted label is the best match AND meets CLIP threshold
    if best_label == pred_label and pred_sim >= clip_thresh:
        return True, trust_score, clip_similarities

    # Rule 2: Another label is a much better match → false positive
    # (the crop visually looks like something else entirely)
    if best_sim - pred_sim > _FALSE_POSITIVE_MARGIN and best_label != pred_label:
        return False, trust_score, clip_similarities

    # Rule 3: Trust score is high enough AND predicted label is not far behind the best
    if trust_score >= trust_thresh and pred_sim >= best_sim - 0.08:
        return True, trust_score, clip_similarities

    # Rule 4: Very tiny objects get extra scrutiny
    if area_ratio < 0.01 and pred_sim < clip_thresh + 0.05:
        return False, trust_score, clip_similarities

    # Rule 5: If trust score is very high, preserve (catches strong confident detections
    # where CLIP might slightly prefer a synonym)
    if trust_score >= 0.60:
        return True, trust_score, clip_similarities

    return False, trust_score, clip_similarities


def _print_verification_debug(
    det: Dict,
    keep: bool,
    trust_score: float,
    clip_similarities: Dict[str, float],
    label: str,
) -> None:
    """Print detailed verification debug output."""
    yolo_conf = det["score"]

    # Get top-3 competing labels by CLIP similarity
    sorted_labels = sorted(clip_similarities, key=lambda k: clip_similarities[k], reverse=True)
    top3 = [(lbl, clip_similarities[lbl]) for lbl in sorted_labels[:5] if lbl != label][:3]

    pred_sim = clip_similarities.get(label, 0.0)

    status = "✓" if keep else "✗"
    verdict = "verified" if keep else "rejected false detection"

    print(f"\n[verification]")
    print(f"  YOLO:")
    print(f"    {label} ({yolo_conf:.2f})")
    print(f"  CLIP:")
    print(f"    {label}: {pred_sim:.2f}")
    for comp_lbl, comp_sim in top3:
        print(f"    {comp_lbl}: {comp_sim:.2f}")
    print(f"  Trust score: {trust_score:.3f}")
    print(f"  Decision: {status} {verdict}")


def verify_detections(
    detections: List[Dict],
    image: Image.Image,
    debug: bool = True,
) -> List[Dict]:
    """Verify YOLO detections using CLIP semantic consistency.

    For each detection:
      1. Crop the detection region from the image
      2. Compute CLIP embedding of the crop
      3. Compute cosine similarity against text embeddings of all COCO class names
      4. Compute combined trust score: 0.5 * yolo_conf + 0.5 * clip_sim
      5. Reject detections where CLIP disagrees with YOLO's label

    Args:
        detections: Raw YOLO detections
                    [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        image:      Full PIL image (RGB)
        debug:      Print detailed verification output

    Returns:
        Verified detections (subset of input, may be filtered/re-ranked)
    """
    if not detections:
        return []

    scorer = get_clip_scorer()
    text_embeddings = _get_coco_text_embeddings()
    img_w, img_h = image.size
    image_size = (img_w, img_h)

    verified: List[Dict] = []
    rejected: List[Dict] = []

    for det in detections:
        label = det["label"]
        box = det["box"]

        # Handle underscore labels
        label_clean = label.replace("_", " ")
        det["label"] = label_clean

        # Crop detection region
        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))

        if x2 <= x1 or y2 <= y1:
            if debug:
                print(f"\n[verification]  ✗ {label}: invalid box, rejected")
            rejected.append(det)
            continue

        crop = image.crop((x1, y1, x2, y2))

        # Compute CLIP embedding for the crop
        with torch.no_grad():
            crop_inputs = scorer._processor(
                images=crop,
                return_tensors="pt",
            ).to(scorer.device)
            crop_features = scorer._model.get_image_features(**crop_inputs)
            if hasattr(crop_features, "pooler_output"):
                crop_embedding = F.normalize(crop_features.pooler_output[0], dim=-1)
            else:
                crop_embedding = F.normalize(crop_features[0], dim=-1)

        keep, trust_score, clip_similarities = _verify_single_detection(
            det, crop_embedding, text_embeddings, image_size,
        )

        if debug:
            _print_verification_debug(det, keep, trust_score, clip_similarities, label_clean)

        if keep:
            # Add verification metadata
            det["verification_score"] = round(trust_score, 3)
            det["clip_similarity"] = round(clip_similarities.get(label_clean, 0.0), 3)
            verified.append(det)
        else:
            det["verification_score"] = round(trust_score, 3)
            det["clip_similarity"] = round(clip_similarities.get(label_clean, 0.0), 3)
            rejected.append(det)

    if debug:
        print(f"\n[verification] {'=' * 40}")
        print(f"[verification]  ✓ verified: {len(verified)} detections")
        for d in verified:
            print(f"    ✓ {d['label']} (yolo={d['score']:.2f}, clip={d.get('clip_similarity', 0):.2f}, "
                  f"trust={d.get('verification_score', 0):.2f})")
        print(f"[verification]  ✗ rejected: {len(rejected)} detections")
        for d in rejected:
            print(f"    ✗ {d['label']} (yolo={d['score']:.2f}, clip={d.get('clip_similarity', 0):.2f}, "
                  f"trust={d.get('verification_score', 0):.2f})")
        print(f"[verification] {'=' * 40}")

    return verified
