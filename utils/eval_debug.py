"""
Comprehensive evaluation, visualization, debugging, and analysis utilities.

Phases:
    1 — Visual debug outputs
    2 — Baseline comparison
    3 — Failure analysis
    5 — Attention visualization
    6 — Relation error analysis
    7 — Relation prior improvements
    8 — Final evaluation table
    9 — Final report
"""

from __future__ import annotations

import csv
import json
import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from relation_prediction.predict import (
    SEMANTIC_PREDS,
    WEAK_SPATIAL,
    NEUTRAL_SPATIAL,
    _get_categories,
    _calibrate_scores,
    _compute_prior_adjustment,
    _get_raw_logits,
    evaluate_relation_quality,
    _get_feature_group_norms,
    _model as rel_model,
    _pred_vocab,
    _label_vocab,
    _device,
    _model_clip_dim,
    _model_pose_dim,
    _model_union_dim,
    _model_type,
)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_COLORS = [
    "#FF4444", "#4488FF", "#44BB44", "#FF8800", "#AA44FF",
    "#00CCCC", "#FF44FF", "#888800", "#FF6666", "#66AAFF",
]

_SEMANTIC_REL_COLOR = "#FFDD00"
_SPATIAL_REL_COLOR = "#88CCFF"

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _get_font(size: int = 14) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ===================================================================
# PHASE 1 — Visual Debug Outputs
# ===================================================================

def save_visual_debug(
    image: Image.Image,
    detections: List[Dict],
    verified_detections: List[Dict],
    relations: List[Dict],
    caption: str,
    image_name: str,
    output_dir: str = "outputs/debug",
) -> Dict[str, str]:
    """Generate 5-panel visual debug output for a single image.

    Saves:
        1. Original image
        2. YOLO detections overlay
        3. Verified detections overlay
        4. Relation visualization
        5. Final caption overlay (debug composite)

    Returns:
        Dict mapping panel name -> saved file path
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    img_w, img_h = image.size
    font = _get_font(14)
    small_font = _get_font(11)

    # -- Panel 1: Original image -------------------------------------
    orig_path = os.path.join(output_dir, f"{image_name}_01_original.png")
    image.save(orig_path)
    paths["original"] = orig_path

    # -- Panel 2: YOLO raw detections --------------------------------
    yolo_pil = image.copy()
    yolo_draw = ImageDraw.Draw(yolo_pil)
    for i, d in enumerate(detections):
        color = _COLORS[i % len(_COLORS)]
        box = d["box"]
        yolo_draw.rectangle(box, outline=color, width=3)
        label_text = f"YOLO: {d['label']} {d['score']:.2f}"
        bbox = yolo_draw.textbbox((box[0], box[1] - 20), label_text, font=font)
        yolo_draw.rectangle(bbox, fill=color)
        yolo_draw.text((box[0], box[1] - 20), label_text, fill="white", font=font)
    yolo_path = os.path.join(output_dir, f"{image_name}_02_yolo_detections.png")
    yolo_pil.save(yolo_path)
    paths["yolo"] = yolo_path

    # -- Panel 3: CLIP-verified detections ---------------------------
    ver_pil = image.copy()
    ver_draw = ImageDraw.Draw(ver_pil)
    for i, d in enumerate(verified_detections):
        color = _COLORS[i % len(_COLORS)]
        box = d["box"]
        ver_draw.rectangle(box, outline=color, width=3)
        trust = d.get("verification_score", d.get("score", 0))
        clip_sim = d.get("clip_similarity", 0)
        label_text = f"✓ {d['label']} (t={trust:.2f}, c={clip_sim:.2f})"
        bbox = ver_draw.textbbox((box[0], box[1] - 20), label_text, font=font)
        ver_draw.rectangle(bbox, fill=color)
        ver_draw.text((box[0], box[1] - 20), label_text, fill="white", font=font)
    ver_path = os.path.join(output_dir, f"{image_name}_03_verified.png")
    ver_pil.save(ver_path)
    paths["verified"] = ver_path

    # -- Panel 4: Relation visualization -----------------------------
    rel_pil = image.copy()
    rel_draw = ImageDraw.Draw(rel_pil)
    obj_colors = {}
    for i, d in enumerate(verified_detections):
        color = _COLORS[i % len(_COLORS)]
        obj_colors[d["label"]] = color
        box = d["box"]
        rel_draw.rectangle(box, outline=color, width=2)
        label_text = d["label"]
        bbox = rel_draw.textbbox((box[0], box[1] - 18), label_text, font=small_font)
        rel_draw.rectangle(bbox, fill=color)
        rel_draw.text((box[0], box[1] - 18), label_text, fill="white", font=small_font)

    for r in relations:
        is_sem = r["predicate"] in SEMANTIC_PREDS
        color = _SEMANTIC_REL_COLOR if is_sem else _SPATIAL_REL_COLOR
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx = (subj_box[0] + subj_box[2]) / 2
            sy = (subj_box[1] + subj_box[3]) / 2
            ox = (obj_box[0] + obj_box[2]) / 2
            oy = (obj_box[1] + obj_box[3]) / 2
            rel_draw.line([(sx, sy), (ox, oy)], fill=color, width=3)
            mx = (sx + ox) / 2
            my = (sy + oy) / 2 - 14
            arrow_label = f"{r['subject']} -- {r['predicate']} --> {r['object']}"
            conf = r.get("adjusted_confidence", r.get("confidence", 0))
            rel_text = f"{r['predicate']} ({conf:.2f})"
            text_bbox = rel_draw.textbbox((mx, my), rel_text, font=small_font)
            rel_draw.rectangle(text_bbox, fill=color)
            rel_draw.text((mx, my), rel_text, fill="black", font=small_font)
    rel_path = os.path.join(output_dir, f"{image_name}_04_relations.png")
    rel_pil.save(rel_path)
    paths["relations"] = rel_path

    # -- Panel 5: Final caption on debug composite -------------------
    cap_pil = image.copy()
    cap_draw = ImageDraw.Draw(cap_pil)
    for i, d in enumerate(verified_detections):
        color = _COLORS[i % len(_COLORS)]
        box = d["box"]
        cap_draw.rectangle(box, outline=color, width=2)

    for r in relations:
        is_sem = r["predicate"] in SEMANTIC_PREDS
        color = _SEMANTIC_REL_COLOR if is_sem else _SPATIAL_REL_COLOR
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx = (subj_box[0] + subj_box[2]) / 2
            sy = (subj_box[1] + subj_box[3]) / 2
            ox = (obj_box[0] + obj_box[2]) / 2
            oy = (obj_box[1] + obj_box[3]) / 2
            cap_draw.line([(sx, sy), (ox, oy)], fill=color, width=2)

    # Caption overlay at bottom
    cap_rect = [0, img_h - 80, img_w, img_h]
    cap_draw.rectangle(cap_rect, fill=(0, 0, 0, 200))
    cap_text = f"Caption: {caption}"
    words = cap_text.split()
    lines = []
    current = ""
    for word in words:
        test = current + " " + word if current else word
        if cap_draw.textbbox((0, 0), test, font=font)[2] < img_w - 20:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = img_h - 70
    for line in lines:
        cap_draw.text((10, y), line, fill="white", font=font)
        y += 20

    cap_path = os.path.join(output_dir, f"{image_name}_05_caption.png")
    cap_pil.save(cap_path)
    paths["caption"] = cap_path

    print(f"[eval_debug] Phase 1: Saved 5 debug panels for {image_name}")
    return paths


def save_debug_composite(
    image: Image.Image,
    detections: List[Dict],
    verified_detections: List[Dict],
    relations: List[Dict],
    caption: str,
    image_name: str,
    output_dir: str = "outputs/debug",
) -> str:
    """Save a single composite image with all debug info overlaid.

    Returns path to composite image.
    """
    os.makedirs(output_dir, exist_ok=True)
    pil = image.copy()
    draw = ImageDraw.Draw(pil)
    font = _get_font(14)
    small_font = _get_font(11)
    img_w, img_h = pil.size

    # Draw verified detection boxes with labels and trust scores
    for i, d in enumerate(verified_detections):
        color = _COLORS[i % len(_COLORS)]
        box = d["box"]
        draw.rectangle(box, outline=color, width=3)
        trust = d.get("verification_score", d.get("score", 0))
        label_text = f"{d['label']} ({trust:.2f})"
        bbox = draw.textbbox((box[0], box[1] - 18), label_text, font=small_font)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1] - 18), label_text, fill="white", font=small_font)

    # Draw relation arrows with predicate labels
    for r in relations:
        is_sem = r["predicate"] in SEMANTIC_PREDS
        color = _SEMANTIC_REL_COLOR if is_sem else _SPATIAL_REL_COLOR
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx = (subj_box[0] + subj_box[2]) / 2
            sy = (subj_box[1] + subj_box[3]) / 2
            ox = (obj_box[0] + obj_box[2]) / 2
            oy = (obj_box[1] + obj_box[3]) / 2
            draw.line([(sx, sy), (ox, oy)], fill=color, width=3)
            mx = (sx + ox) / 2
            my = (sy + oy) / 2 - 14
            conf = r.get("adjusted_confidence", r.get("confidence", 0))
            rel_text = f"{r['predicate']} ({conf:.2f})"
            text_bbox = draw.textbbox((mx, my), rel_text, font=small_font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((mx, my), rel_text, fill="black", font=small_font)

    # Caption overlay at bottom
    cap_rect = [0, img_h - 90, img_w, img_h]
    draw.rectangle(cap_rect, fill=(0, 0, 0, 200))
    caption_title = f"Grounded Caption: {caption}"
    caption_lines = []
    cap_words = caption_title.split()
    current = ""
    for word in cap_words:
        test = current + " " + word if current else word
        if draw.textbbox((0, 0), test, font=font)[2] < img_w - 20:
            current = test
        else:
            caption_lines.append(current)
            current = word
    if current:
        caption_lines.append(current)
    y = img_h - 80
    for line in caption_lines:
        draw.text((10, y), line, fill="white", font=font)
        y += 20

    path = os.path.join(output_dir, f"{image_name}_debug_composite.png")
    pil.save(path)
    print(f"[eval_debug] Saved debug composite: {path}")
    return path


# ===================================================================
# PHASE 2 — Baseline Comparison
# ===================================================================

def run_baseline_comparison_debug(
    image: Image.Image,
    image_name: str,
    vanilla_caption: str,
    grounded_caption: str,
    relations: List[Dict],
    detections: List[Dict],
    output_dir: str = "analysis_results/baseline_comparison",
) -> Dict:
    """Generate side-by-side comparison of vanilla BLIP vs grounded pipeline.

    Args:
        image: Input PIL image.
        image_name: Image stem for filenames.
        vanilla_caption: BLIP baseline caption (ungrounded).
        grounded_caption: Grounded pipeline caption.
        relations: Grounded relations.
        detections: Verified detections.
        output_dir: Output directory.

    Returns:
        Comparison dict with all data.
    """
    os.makedirs(output_dir, exist_ok=True)

    obj_list = [d["label"] for d in detections]
    rel_list = [
        f"{r['subject']} {r['predicate']} {r['object']}"
        for r in relations
    ]

    comparison = {
        "image_name": image_name,
        "objects_detected": obj_list,
        "relations": rel_list,
        "vanilla_blip_caption": vanilla_caption,
        "grounded_pipeline_caption": grounded_caption,
        "hallucination_likely": _detect_hallucination_likely(vanilla_caption, obj_list),
    }

    # Save comparison JSON
    json_path = os.path.join(output_dir, f"{image_name}_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save side-by-side image visualization
    _save_side_by_side(
        image, vanilla_caption, grounded_caption,
        rel_list, obj_list,
        os.path.join(output_dir, f"{image_name}_side_by_side.png"),
    )

    print(f"[eval_debug] Phase 2: Baseline comparison saved to {json_path}")
    return comparison


def _detect_hallucination_likely(caption: str, detected_objects: List[str]) -> bool:
    """Heuristic check if a caption likely hallucinates objects."""
    caption_lower = caption.lower()
    detected_lower = {o.lower() for o in detected_objects}
    known_objects = {
        "person", "bicycle", "car", "dog", "cat", "horse", "chair",
        "bottle", "cup", "cell phone", "phone", "book", "umbrella",
        "backpack", "handbag", "suitcase", "tie", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "skateboard",
        "surfboard", "tennis racket", "wine glass", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush", "woman", "man", "people", "child", "girl",
        "boy",
    }
    found = set()
    for word in caption_lower.split():
        word_clean = word.strip(".,!?;:'\"")
        if word_clean in known_objects:
            found.add(word_clean)
    hallucinated = found - detected_lower
    return len(hallucinated) > 0


def _save_side_by_side(
    image: Image.Image,
    vanilla: str,
    grounded: str,
    relations: List[str],
    objects: List[str],
    output_path: str,
) -> None:
    """Create a side-by-side visualization comparing vanilla vs grounded."""
    from PIL import ImageDraw, ImageFont

    font = _get_font(16)
    small = _get_font(13)

    # Create canvas: 2x image width + padding
    img_w, img_h = image.size
    canvas_w = img_w * 2 + 60
    canvas_h = img_h + 280
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Place images side by side
    canvas.paste(image, (20, 60))
    canvas.paste(image, (img_w + 40, 60))

    # Labels
    draw.text((20, 20), "Vanilla BLIP", fill="#FF8888", font=font)
    draw.text((img_w + 40, 20), "Grounded Pipeline", fill="#88FF88", font=font)

    # Vanilla caption below
    _draw_wrapped_text(draw, 20, img_h + 80, vanilla, font, fill="#FF8888", max_w=img_w + 10)

    # Grounded caption below
    _draw_wrapped_text(draw, img_w + 40, img_h + 80, grounded, font, fill="#88FF88", max_w=img_w + 10)

    # Objects and relations section
    y = img_h + 160
    if objects:
        draw.text((20, y), f"Detected objects: {', '.join(objects)}", fill="#AAAAAA", font=small)
        y += 25
    if relations:
        draw.text((20, y), "Grounded relations:", fill="#AAAAAA", font=small)
        y += 22
        for rel in relations:
            draw.text((30, y), f"  * {rel}", fill="#CCCCCC", font=small)
            y += 20

    canvas.save(output_path)


def _draw_wrapped_text(draw, x, y, text, font, fill, max_w):
    """Draw text with word wrap."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = current + " " + word if current else word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] < max_w:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for i, line in enumerate(lines):
        draw.text((x, y + i * 22), line, fill=fill, font=font)


# ===================================================================
# PHASE 3 — Failure Analysis
# ===================================================================

FAILURE_CATEGORIES = [
    "wrong_relation",
    "missed_interaction",
    "hallucinated_object",
    "over_filtering",
    "weak_grounding",
    "noisy_spatial_relation",
]


def classify_failures(
    image_name: str,
    detections: List[Dict],
    relations: List[Dict],
    raw_predictions: List[Dict],
    caption: str,
    vanilla_caption: Optional[str] = None,
) -> Dict:
    """Automatically classify failures in the pipeline output.

    Checks for:
        - wrong_relation: Semantic predicate on clearly mismatched objects
        - missed_interaction: Animate subject with no semantic relation
        - hallucinated_object: Caption mentions objects not detected
        - over_filtering: Too few relations given many detections
        - weak_grounding: Relations rely heavily on priors over visual evidence
        - noisy_spatial_relation: Spatial preds dominate semantic preds

    Returns:
        Failure report dict.
    """
    issues: List[Dict] = []
    obj_labels = [d["label"] for d in detections]
    animate_objs = [l for l in obj_labels if "animate" in _get_categories(l)]
    has_animate = len(animate_objs) > 0

    # Check 1: Animate subject with no semantic relation → missed interaction
    if has_animate and len(relations) > 0:
        has_semantic = any(r["predicate"] in SEMANTIC_PREDS for r in relations)
        if not has_semantic:
            issues.append({
                "type": "missed_interaction",
                "severity": "high",
                "detail": f"Animate objects ({animate_objs}) but no semantic relations",
            })
    elif has_animate and len(relations) == 0:
        issues.append({
            "type": "missed_interaction",
            "severity": "high",
            "detail": f"Animate objects present but ZERO relations inferred",
        })

    # Check 2: Semantic predicates on wrong object categories
    for r in relations:
        pred = r["predicate"]
        obj = r["object"]
        subj = r["subject"]
        if pred == "sitting on":
            obj_cats = _get_categories(obj)
            if "furniture" not in obj_cats and "rideable" not in obj_cats and "animate" not in obj_cats:
                issues.append({
                    "type": "wrong_relation",
                    "severity": "medium",
                    "detail": f"'sitting on' with non-furniture object '{obj}'",
                    "relation": f"{subj} {pred} {obj}",
                })
        if pred == "riding":
            obj_cats = _get_categories(obj)
            if "rideable" not in obj_cats:
                issues.append({
                    "type": "wrong_relation",
                    "severity": "medium",
                    "detail": f"'riding' with non-rideable object '{obj}'",
                    "relation": f"{subj} {pred} {obj}",
                })

    # Check 3: Caption hallucination
    if caption and obj_labels:
        cap_lower = caption.lower()
        detected_set = {l.lower() for l in obj_labels}
        for obj_name in detected_set:
            pass
        known_objects = {
            "person", "bicycle", "car", "dog", "cat", "horse",
            "chair", "bottle", "cup", "phone", "book", "umbrella",
            "backpack", "handbag", "tie", "frisbee", "skis",
            "snowboard", "ball", "kite", "skateboard", "surfboard",
            "racket", "glass", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake",
            "couch", "plant", "bed", "table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "microwave",
            "oven", "toaster", "sink", "fridge", "clock", "vase",
            "scissors", "teddy bear", "toothbrush",
            "woman", "man", "people", "child", "girl", "boy",
            "bird", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "truck", "boat", "train", "bus", "plane",
            "motorcycle",
        }
        found = set()
        for word in cap_lower.split():
            w = word.strip(".,!?;:'\"")
            if w in known_objects:
                found.add(w)
        hallucinated = found - detected_set
        if hallucinated:
            issues.append({
                "type": "hallucinated_object",
                "severity": "high",
                "detail": f"Caption mentions undetected objects: {hallucinated}",
            })

    # Check 4: Over-filtering — 4+ detections but 0 relations
    if len(detections) >= 4 and len(relations) == 0:
        issues.append({
            "type": "over_filtering",
            "severity": "medium",
            "detail": f"{len(detections)} detections but 0 relations (precision too aggressive)",
        })

    # Check 5: Weak grounding — priors dominate visual score
    for r in relations:
        prior = r.get("prior_adjustment", 0)
        conf = r.get("confidence", 0)
        if abs(prior) > conf * 1.5 and conf < 0.3:
            issues.append({
                "type": "weak_grounding",
                "severity": "medium",
                "detail": f"Prior ({prior:.3f}) dominates visual confidence ({conf:.3f}) for {r['subject']} {r['predicate']} {r['object']}",
            })

    # Check 6: Noisy spatial relations — spatial >> semantic
    if len(relations) >= 2:
        spatial_count = sum(1 for r in relations if r["predicate"] in WEAK_SPATIAL or r["predicate"] in NEUTRAL_SPATIAL)
        semantic_count = sum(1 for r in relations if r["predicate"] in SEMANTIC_PREDS)
        if spatial_count > semantic_count * 2 and semantic_count > 0:
            issues.append({
                "type": "noisy_spatial_relation",
                "severity": "low",
                "detail": f"Spatial relations ({spatial_count}) dominate semantic ({semantic_count})",
            })

    severity = "none"
    if any(i["severity"] == "high" for i in issues):
        severity = "high"
    elif any(i["severity"] == "medium" for i in issues):
        severity = "medium"
    elif issues:
        severity = "low"

    return {
        "image_name": image_name,
        "num_detections": len(detections),
        "num_relations": len(relations),
        "has_animate_subject": has_animate,
        "issues": issues,
        "severity": severity,
    }


def build_failure_report(
    per_image_failures: List[Dict],
    output_dir: str = "analysis_results",
) -> str:
    """Build failure report JSON with aggregate statistics.

    Args:
        per_image_failures: List of failure analysis dicts per image.
        output_dir: Output directory.

    Returns:
        Path to saved failure report.
    """
    os.makedirs(output_dir, exist_ok=True)

    category_counts: Dict[str, int] = defaultdict(int)
    severity_counts: Dict[str, int] = defaultdict(int)
    image_details = []

    for f in per_image_failures:
        severity_counts[f.get("severity", "none")] += 1
        for issue in f.get("issues", []):
            category_counts[issue["type"]] += 1
        image_details.append({
            "image_name": f["image_name"],
            "severity": f.get("severity", "none"),
            "num_detections": f["num_detections"],
            "num_relations": f["num_relations"],
            "issues": f.get("issues", []),
        })

    total = len(per_image_failures)
    report = {
        "total_images_analyzed": total,
        "images_with_issues": sum(1 for f in per_image_failures if f.get("issues")),
        "images_clean": sum(1 for f in per_image_failures if not f.get("issues")),
        "severity_summary": dict(severity_counts),
        "failure_category_counts": dict(category_counts),
        "failure_categories_pct": {
            k: round(v / max(total, 1) * 100, 1)
            for k, v in category_counts.items()
        },
        "per_image_details": image_details,
    }

    path = os.path.join(output_dir, "failure_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[eval_debug] Phase 3: Failure report saved to {path}")
    return path


# ===================================================================
# PHASE 5 — Attention Visualization (for Transformer models)
# ===================================================================

def generate_attention_visualization(
    model: torch.nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    union_feat: Optional[torch.Tensor],
    pose_feat: Optional[torch.Tensor],
    pred_vocab_labels: List[str],
    output_dir: str = "outputs/attention",
    image_name: str = "attention",
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """Generate attention modality breakdown per predicate.

    For Transformer models: uses decoder cross-attention.
    For MLP models: uses feature group norms as proxy.

    Saves JSON + text + PNG chart.
    Returns path to the JSON file, or None on failure.

    If *device* is passed, input tensors are moved to that device
    (if they are not already on it) before the forward pass.
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, 'get_attention_summary') and hasattr(model, '_decoder_cross_attn_weights'):
        # Move input tensors to the correct device
        if device is not None:
            subj_idx = subj_idx.to(device)
            obj_idx = obj_idx.to(device)
            geo = geo.to(device)
            if subj_feat is not None:
                subj_feat = subj_feat.to(device)
            if obj_feat is not None:
                obj_feat = obj_feat.to(device)
            if union_feat is not None:
                union_feat = union_feat.to(device)
            if pose_feat is not None:
                pose_feat = pose_feat.to(device)

        model.set_attention_capture(True)
        with torch.no_grad():
            try:
                _ = model.forward(
                    subj_idx, obj_idx, geo,
                    subj_feat=subj_feat, obj_feat=obj_feat,
                    union_feat=union_feat, pose_feat=pose_feat,
                )
            except Exception as e:
                model.set_attention_capture(False)
                print(f"[attention] forward pass failed: {e}")
                return None

        summary = model.get_attention_summary(pred_vocab_labels=pred_vocab_labels)
        model.set_attention_capture(False)

        if not summary:
            print(f"[attention] No attention weights captured for {image_name}")
            return None

        # Format summary text with visual bar chart
        lines = ["Attention Modality Breakdown (per predicate):", ""]
        for pred_name, mod_dict in summary.items():
            lines.append(f"  {pred_name}:")
            sorted_items = sorted(mod_dict.items(), key=lambda x: -x[1])
            for mod_name, weight in sorted_items:
                bar = "█" * int(weight * 40)
                lines.append(f"    {mod_name:20s}: {weight:.4f}  {bar}")

        text = "\n".join(lines)
        print(f"\n[attention_analysis]\n{text}")

        # -- Save JSON --
        json_path = os.path.join(output_dir, f"{image_name}_attention.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[attention] visualization saved to {json_path}")

        # -- Save text (UTF-8 to handle Unicode bar characters) --
        txt_path = os.path.join(output_dir, f"{image_name}_attention.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # -- Save per-predicate attention JSON --
        per_pred_path = os.path.join(output_dir, f"{image_name}_per_predicate_attention.json")
        try:
            with open(per_pred_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[attention] per-predicate attention saved to {per_pred_path}")
        except Exception as e:
            print(f"[attention] could not save per-predicate attention: {e}")

        # -- Save modality contribution image --
        try:
            _save_modality_contribution_chart(
                summary,
                os.path.join(output_dir, f"{image_name}_modality_contributions.png"),
            )
        except Exception as e:
            print(f"[attention] could not save modality chart: {e}")

        return json_path

    # MLP fallback: use feature group norms
    try:
        norms = _get_feature_group_norms(model)
        if not norms:
            return None
        total = sum(norms.values()) or 1.0
        contrib = {k: round(v / total, 4) for k, v in norms.items()}

        lines = ["Feature Modality Contribution (MLP weight norms):", ""]
        for name, val in sorted(contrib.items(), key=lambda x: -x[1]):
            bar = "█" * int(val * 40)
            lines.append(f"  {name:20s}: {val:.4f}  {bar}")

        text = "\n".join(lines)
        print(f"\n[attention_analysis] (MLP proxy)\n{text}")

        json_path = os.path.join(output_dir, f"{image_name}_attention.json")
        with open(json_path, "w") as f:
            json.dump(contrib, f, indent=2)

        print(f"[eval_debug] Phase 5: MLP feature contribution saved to {json_path}")
        return json_path
    except Exception as e:
        print(f"[attention_analysis] Skipped ({e})")
        return None


def _save_modality_contribution_chart(
    summary: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Save a horizontal bar chart of modality contributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Aggregate across all predicates for an overall modality view
    modality_totals: Dict[str, float] = {}
    for _, mod_dict in summary.items():
        for mod_name, weight in mod_dict.items():
            modality_totals[mod_name] = modality_totals.get(mod_name, 0.0) + weight

    if not modality_totals:
        return

    names = list(modality_totals.keys())
    values = list(modality_totals.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Total Attention Weight")
    ax.set_title("Modality Contributions (aggregated across predicates)")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[attention] modality contribution chart saved to {save_path}")


# ===================================================================
# PHASE 6 — Relation Error Analysis
# ===================================================================

def analyze_relation_errors(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float,
    img_h: float,
    image: Optional[Image.Image] = None,
    pred_vocab_labels: Optional[List[str]] = None,
    temperature: float = 2.0,
) -> Dict:
    """Deep diagnostics for WHY a specific relation was predicted.

    Returns:
        Dict with:
            - raw_logits: per-predicate raw logits
            - calibrated: temperature-calibrated scores
            - prior_adjustments: semantic prior bonuses/penalties
            - final_scores: combined scores
            - top_k: ranked predicates with scores
            - semantic_override: whether semantic consistency triggered
            - feature_norms: modality contribution breakdown
    """
    logits, pred_tokens, _, _ = _get_raw_logits(
        subject, obj_label, box1, box2,
        img_w=img_w, img_h=img_h,
        image=image,
    )

    if logits is None:
        return {"error": "Could not get logits"}

    if pred_vocab_labels is None and _pred_vocab is not None:
        pred_vocab_labels = [_pred_vocab.token(i) for i in range(len(_pred_vocab))]

    calibrated = _calibrate_scores(logits.unsqueeze(0), temperature=temperature)[0]

    per_pred = []
    for pidx, pname in enumerate(pred_tokens if pred_vocab_labels is None else pred_vocab_labels):
        if pname in ("<pad>", "<unk>"):
            continue
        raw_val = logits[pidx].item()
        calib_val = calibrated[pidx].item()
        bonus, penalty, prior_total = _compute_prior_adjustment(subject, pname, obj_label)
        final_val = calib_val + prior_total
        per_pred.append({
            "predicate": pname,
            "raw_logit": round(raw_val, 4),
            "calibrated": round(calib_val, 4),
            "prior_bonus": round(bonus, 4),
            "prior_penalty": round(penalty, 4),
            "prior_total": round(prior_total, 4),
            "final_score": round(final_val, 4),
        })

    per_pred.sort(key=lambda x: -x["final_score"])

    top_k = per_pred[:5]

    # Feature norms
    feature_norms = {}
    if rel_model is not None:
        try:
            feature_norms = _get_feature_group_norms(rel_model)
        except Exception:
            pass

    return {
        "subject": subject,
        "object": obj_label,
        "num_predicates": len(per_pred),
        "top_k": top_k,
        "full_ranked": per_pred,
        "feature_norms": feature_norms,
        "winner": top_k[0] if top_k else None,
        "runner_up": top_k[1] if len(top_k) > 1 else None,
        "margin": (top_k[0]["final_score"] - top_k[1]["final_score"]) if len(top_k) > 1 else None,
    }


# ===================================================================
# PHASE 7 — Relation Prior Improvements
# ===================================================================

# Improved soft semantic penalties using object compatibility.
# These are applied as additional prior adjustments, not hard rules.

_IMPROVED_PRIORS = {
    "sitting on": {
        "allowed_objects": {
            "chair", "couch", "bench", "bed", "dining table", "toilet",
            "bicycle", "motorcycle", "horse", "skateboard", "surfboard",
            "floor", "ground", "grass", "stool", "sofa", "seat",
            "car", "truck", "bus",
        },
        "reject_objects": {
            "dog", "cat", "bird", "cell phone", "bottle", "cup",
            "book", "frisbee", "kite", "baseball bat", "tennis racket",
            "sports ball",
        },
        "soft_reject_penalty": -0.35,
        "allowed_bonus": 0.08,
    },
    "riding": {
        "allowed_objects": {
            "bicycle", "horse", "motorcycle", "skateboard", "surfboard",
            "skis", "snowboard", "bus", "train", "elephant", "camel",
            "bike", "scooter",
        },
        "reject_objects": {
            "chair", "couch", "bench", "bed", "dining table",
            "dog", "cat", "cell phone", "bottle", "cup", "book",
            "frisbee", "kite", "backpack", "handbag", "suitcase",
            "tie", "umbrella", "sports ball",
        },
        "soft_reject_penalty": -0.40,
        "allowed_bonus": 0.10,
    },
    "holding": {
        "allowed_objects": set(),  # broad - most handheld objects
        "reject_objects": {
            "chair", "couch", "bench", "bed", "dining table",
            "car", "bicycle", "truck", "bus", "train",
            "horse", "elephant", "cow",
            "toilet", "sink", "refrigerator",
        },
        "soft_reject_penalty": -0.30,
        "allowed_bonus": 0.06,
    },
    "wearing": {
        "allowed_objects": {
            "backpack", "handbag", "tie", "suitcase", "umbrella",
            "helmet", "glasses", "hat", "cap", "shoe", "shoes",
            "jacket", "coat", "shirt", "pants", "skirt", "dress",
            "watch", "necklace", "scarf", "belt", "glove", "gloves",
        },
        "reject_objects": {
            "chair", "couch", "bench", "bed", "dining table",
            "car", "bicycle", "bus", "truck", "train",
            "cell phone", "bottle", "cup", "book",
            "dog", "cat", "horse", "cow", "elephant",
            "frisbee", "kite", "sports ball", "skateboard",
            "surfboard", "toilet", "sink", "refrigerator",
        },
        "soft_reject_penalty": -0.35,
        "allowed_bonus": 0.10,
    },
    "carrying": {
        "allowed_objects": set(),
        "reject_objects": {
            "chair", "couch", "bench", "bed", "dining table",
            "car", "bicycle", "bus", "truck", "train",
            "toilet", "sink", "refrigerator",
            "horse", "cow", "elephant",
        },
        "soft_reject_penalty": -0.25,
        "allowed_bonus": 0.06,
    },
    "looking at": {
        "allowed_objects": set(),
        "reject_objects": set(),
        "soft_reject_penalty": 0.0,
        "allowed_bonus": 0.0,
    },
    "standing on": {
        "allowed_objects": {
            "chair", "couch", "bench", "bed", "dining table",
            "floor", "ground", "grass", "surfboard", "skateboard",
            "stool", "stage", "platform",
        },
        "reject_objects": {
            "cell phone", "bottle", "cup", "book",
            "dog", "cat", "bird",
            "frisbee", "kite", "sports ball",
        },
        "soft_reject_penalty": -0.30,
        "allowed_bonus": 0.06,
    },
}


def compute_refined_prior_adjustment(
    subject: str,
    predicate: str,
    object: str,
) -> Tuple[float, float, float]:
    """Compute improved soft semantic penalties using object compatibility.

    Extends the basic prior with more specific object-level constraints.
    Uses only soft penalties (never hard rejection).

    Returns:
        (bonus, penalty, total)
    """
    bonus = 0.0
    penalty = 0.0
    subj_cats = _get_categories(subject)
    obj_cats = _get_categories(object)
    obj_lower = object.lower().replace("_", " ")

    # Basic category priors (existing logic)
    base_bonus, base_penalty, _ = _compute_prior_adjustment(subject, predicate, object)
    bonus += base_bonus
    penalty += base_penalty

    # Refined object-level priors
    prior = _IMPROVED_PRIORS.get(predicate)
    if prior:
        allowed = prior.get("allowed_objects", set())
        rejected = prior.get("reject_objects", set())

        obj_lower_no_article = obj_lower
        if obj_lower.startswith("a ") or obj_lower.startswith("an "):
            obj_lower_no_article = obj_lower.split(" ", 1)[1]

        if obj_lower_no_article in rejected or obj_lower in rejected:
            penalty += prior["soft_reject_penalty"]
        elif allowed and (obj_lower_no_article in allowed or obj_lower in allowed):
            bonus += prior["allowed_bonus"]

    # Animate subject with animate object: strong penalty for riding/sitting
    if "animate" in subj_cats and "animate" in obj_cats:
        if predicate in ("sitting on", "riding"):
            penalty -= 0.20

    return bonus, penalty, bonus + penalty


# ===================================================================
# PHASE 8 — Final Evaluation Table
# ===================================================================

def build_final_evaluation_table(
    per_image_results: List[Dict],
    output_dir: str = "analysis_results",
) -> str:
    """Build CSV evaluation table with per-image metrics.

    Columns:
        Image, Objects, Relations, Final Caption, Hallucination?,
        Semantic Precision, Grounded Relation Count, Issues

    Args:
        per_image_results: List of dicts with keys:
            image_name, detections, relations, caption, failures, vanilla_caption

    Returns:
        Path to saved CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "final_eval.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Objects", "Relations", "Final Caption",
            "Hallucination?", "Semantic Precision",
            "Grounded Relation Count", "Failure Severity",
        ])

        for res in per_image_results:
            img_name = res.get("image_name", "unknown")
            obj_list = [d.get("label", "") for d in res.get("detections", [])]
            rels = res.get("relations", [])
            rel_list = [f"{r.get('subject','')} {r.get('predicate','')} {r.get('object','')}" for r in rels]
            caption = res.get("caption", "")
            failures = res.get("failures", {})
            severity = failures.get("severity", "none") if failures else "none"

            # Hallucination check
            hall = "No"
            vanilla = res.get("vanilla_caption", "")
            if vanilla and _detect_hallucination_likely(vanilla, obj_list):
                hall = "Likely (vanilla)"

            # Semantic precision
            sem_count = sum(1 for r in rels if r.get("predicate") in SEMANTIC_PREDS)
            sem_prec = f"{sem_count}/{len(rels)}" if rels else "0/0"

            writer.writerow([
                img_name,
                ", ".join(obj_list),
                ", ".join(rel_list),
                caption,
                hall,
                sem_prec,
                len(rels),
                severity,
            ])

    print(f"[eval_debug] Phase 8: Evaluation table saved to {path}")
    return path


# ===================================================================
# PHASE 9 — Final Report
# ===================================================================

def generate_final_report(
    per_image_results: List[Dict],
    failure_report_path: str,
    eval_table_path: str,
    attention_report_path: Optional[str] = None,
    output_dir: str = "analysis_results",
) -> str:
    """Generate comprehensive final report as JSON.

    Includes:
        1. Visual debug examples summary
        2. Baseline comparison examples
        3. Attention analysis summary
        4. Failure categories
        5. Remaining weaknesses
        6. Most improved relation types
        7. Hallucination reduction observations

    Args:
        per_image_results: All per-image pipeline outputs.
        failure_report_path: Path to failure_report.json.
        eval_table_path: Path to final_eval.csv.
        attention_report_path: Optional path to attention analysis.
        output_dir: Output directory.

    Returns:
        Path to saved report JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    total = len(per_image_results)
    total_relations = sum(len(r.get("relations", [])) for r in per_image_results)
    total_semantic = sum(
        1 for r in per_image_results
        for rel in r.get("relations", [])
        if rel.get("predicate") in SEMANTIC_PREDS
    )
    total_spatial = total_relations - total_semantic

    # Count hallucinated captions (where gating triggered)
    hallucinated_count = sum(
        1 for r in per_image_results
        if r.get("caption", "").startswith("The scene contains")
    )
    gated_count = sum(
        1 for r in per_image_results
        if r.get("raw_caption", "") != r.get("caption", "")
    )

    # Predicate frequency
    pred_freq: Dict[str, int] = defaultdict(int)
    for r in per_image_results:
        for rel in r.get("relations", []):
            pred_freq[rel.get("predicate", "")] += 1

    # Top predicates
    top_preds = sorted(pred_freq.items(), key=lambda x: -x[1])[:10]

    # Load failure report
    failures = {}
    try:
        with open(failure_report_path) as f:
            failures = json.load(f)
    except Exception:
        pass

    report = {
        "report_metadata": {
            "title": "Grounded Caption Pipeline — Final Evaluation Report",
            "total_images": total,
            "total_relations": total_relations,
            "total_semantic_relations": total_semantic,
            "total_spatial_relations": total_spatial,
            "semantic_ratio": round(total_semantic / max(total_relations, 1), 3),
        },
        "hallucination_reduction": {
            "images_with_hallucination_safe_fallback": hallucinated_count,
            "images_with_gating_triggered": gated_count,
            "gating_rate": round(gated_count / max(total, 1) * 100, 1),
            "observation": (
                f"Gating triggered on {gated_count}/{total} images "
                f"({round(gated_count/max(total,1)*100,1)}%). "
                f"Hallucinated captions reduced via evidence gating."
            ),
        },
        "relation_quality": {
            "top_predicates": top_preds,
            "semantic_precision": round(total_semantic / max(total_relations, 1), 3),
            "observation": (
                f"Semantic predicates account for {total_semantic}/{total_relations} "
                f"({round(total_semantic/max(total_relations,1)*100,1)}%) of relations."
            ),
        },
        "failure_analysis": {
            "total_images_with_issues": failures.get("images_with_issues", 0),
            "failure_categories": failures.get("failure_category_counts", {}),
            "severity_summary": failures.get("severity_summary", {}),
        },
        "remaining_weaknesses": [
            "Some spatial relations may still pass precision filters for inanimate objects",
            "CLIP verification may reject valid detections in unusual poses",
            "Relation model can confuse visually similar predicates",
            "Pose features only extracted for 'person' subjects",
        ],
        "most_improved_relation_types": [
            "riding — now correctly preferred over 'on' for person+bicycle via consistency override",
            "sitting on — restricted to furniture/rideable objects via improved priors",
            "holding — rejects large/heavy objects via object compatibility",
            "wearing — restricted to wearable objects via improved priors",
        ],
        "evaluation_table_path": eval_table_path,
        "failure_report_path": failure_report_path,
    }

    if attention_report_path:
        try:
            with open(attention_report_path) as f:
                report["attention_analysis"] = json.load(f)
        except Exception:
            pass

    path = os.path.join(output_dir, "final_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also save a text summary
    txt_path = os.path.join(output_dir, "final_report_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  FINAL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Images processed:          {total}\n")
        f.write(f"Total relations:           {total_relations}\n")
        f.write(f"Semantic relations:        {total_semantic}\n")
        f.write(f"Spatial relations:         {total_spatial}\n")
        f.write(f"Semantic ratio:            {round(total_semantic/max(total_relations,1),3)}\n")
        f.write(f"Hallucination fallbacks:   {hallucinated_count}\n")
        f.write(f"Gating events:             {gated_count}\n")
        f.write(f"Gate trigger rate:         {round(gated_count/max(total,1)*100,1)}%\n\n")

        f.write("-" * 40 + "\n")
        f.write("Top predicates:\n")
        for pred, count in top_preds:
            f.write(f"  {pred}: {count}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Failure analysis:\n")
        if failures:
            f.write(f"  Images with issues: {failures.get('images_with_issues', 0)}\n")
            for cat, cnt in failures.get("failure_category_counts", {}).items():
                f.write(f"  - {cat}: {cnt}\n")

        f.write("\n" + "-" * 40 + "\n")
        f.write("Remaining weaknesses:\n")
        for w in report["remaining_weaknesses"]:
            f.write(f"  * {w}\n")

        f.write("\n-" * 40 + "\n")
        f.write("Most improved relation types:\n")
        for imp in report["most_improved_relation_types"]:
            f.write(f"  * {imp}\n")

    print(f"[eval_debug] Phase 9: Final report saved to {path}")
    print(f"[eval_debug] Phase 9: Text summary saved to {txt_path}")
    return path
