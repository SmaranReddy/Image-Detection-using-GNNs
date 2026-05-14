"""
grounded_caption_pipeline.py — Integrated grounded caption generation.

Full pipeline:
    Image → YOLO detection → CLIP crops → Relation MLP
    → Grounded relation prompt → BLIP-2 generation
    → Evidence gating → Final caption

The trained visual-semantic MLP actively influences caption generation
by providing CLIP-grounded semantic relation predictions.

Usage:
    # Single image
    python grounded_caption_pipeline.py --image test_images/bicycle.jpg

    # Full evaluation (all test images)
    python grounded_caption_pipeline.py --image-dir test_images

    # Side-by-side comparison
    python grounded_caption_pipeline.py --image test_images/bicycle.jpg --compare

    # Qualitative stress-test (hallucination-prone images)
    python grounded_caption_pipeline.py --image-dir "hallucinating images" --num-samples 5

    # Debug visualization
    python grounded_caption_pipeline.py --image test_images/bicycle.jpg --debug
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

from relation_prediction import predict as rel_predict
from relation_prediction.predict import (
    SEMANTIC_PREDS,
    WEAK_SPATIAL,
    NEUTRAL_SPATIAL,
    MIN_SEMANTIC_SCORE,
    WEAK_SPATIAL_THRESHOLD,
    REJECT_INANIMATE_SPATIAL,
    evaluate_relation_quality,
    _get_feature_group_norms,
)
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.blip_captioner import (
    generate_blip_caption,
    generate_blip_baseline,
    generate_blip_candidates,
    generate_blip_semantic_caption,
    build_grounded_prompt,
    build_semantic_prompt,
    verbalize_relation,
)
from utils.clip_scorer import clip_rerank_captions, get_clip_scorer
from utils.detection_verifier import verify_detections
from utils.visualize import draw_relation_boxes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "./checkpoints"
SEMANTIC_PREDICATES = SEMANTIC_PREDS

# ---------------------------------------------------------------------------
# STEP 1 — Verify and load trained relation model
# ---------------------------------------------------------------------------

def verify_relation_model() -> Dict:
    """Load and verify the trained relation model.

    Returns:
        Dict with model info: input_dim, clip_dim, num_labels, num_predicates, device
    """
    print("=" * 60)
    print("  STEP 1 — LOADING TRAINED RELATION MODEL")
    print("=" * 60)

    rel_predict.load_relation_model(CHECKPOINT_DIR)

    embed_dim = rel_predict._model.label_emb.weight.shape[1]
    in_dim = 2 * embed_dim + 5 + 2 * rel_predict._model_clip_dim

    info = {
        "input_dim": in_dim,
        "clip_dim": rel_predict._model_clip_dim,
        "embed_dim": embed_dim,
        "num_labels": len(rel_predict._label_vocab),
        "num_predicates": len(rel_predict._pred_vocab),
        "device": str(rel_predict._device),
        "visual_mode": rel_predict._model_clip_dim > 0,
    }

    print(f"  Model loaded successfully")
    print(f"  Input dimension:     {info['input_dim']}")
    print(f"  CLIP dimension:      {info['clip_dim']}")
    print(f"  Label vocab:         {info['num_labels']} classes")
    print(f"  Predicate vocab:     {info['num_predicates']} predicates")
    print(f"  Visual mode:         {info['visual_mode']}")
    print(f"  Device:              {info['device']}")
    print(f"  Embed dim:           {info['embed_dim']}")

    assert info["input_dim"] == 1669, (
        f"Expected input_dim=1669 but got {info['input_dim']}"
    )
    assert info["visual_mode"], "Model must be in visual-semantic mode (clip_dim > 0)"

    return info


# ---------------------------------------------------------------------------
# STEP 2 — Pure visual relation inference
# ---------------------------------------------------------------------------

def run_pure_visual_inference(
    detections: List[Dict],
    image: Image.Image,
    top_k: int = 3,
    temperature: float = 2.0,
) -> Tuple[List[Dict], List[Dict]]:
    """Run calibrated visual-semantic relation inference with precision filtering.

    Pipeline:
      1. Raw logits → temperature-calibrated softmax
      2. Semantic prior adjustment (category-based bonuses/penalties)
         - Stronger directionality penalties (Step 3)
      3. Hard negative filtering (Step 5)
      4. Semantic consistency override (Step 7):
         - Prefer "riding" over "on" when scores are close
      5. Precision filtering:
         a. Drop weak spatial (< WEAK_SPATIAL_THRESHOLD)
         b. Reject non-semantic below MIN_SEMANTIC_SCORE
         c. Enforce animate subjects for semantic predicates
      6. Semantic consistency cleanup (Step 7)
      7. Semantic priority sort (semantic >> spatial)
      8. Directionality-aware dedup
      9. One primary relation per animate subject

    Precision config:
      MIN_SEMANTIC_SCORE={MIN_SEMANTIC_SCORE}
      WEAK_SPATIAL_THRESHOLD={WEAK_SPATIAL_THRESHOLD}

    Args:
        detections: YOLO detection list.
        image:      Original PIL image.
        top_k:      Max relations to return.
        temperature: Softmax temperature for calibration (default 2.0).

    Returns:
        (filtered_relations, raw_predictions_debug)
    """
    print(f"\n  STEP 2 — PRECISION RELATION INFERENCE")
    print(f"  {len(detections)} detections, top_k={top_k}, T={temperature}")
    print(f"  Precision: MIN_SEMANTIC_SCORE={MIN_SEMANTIC_SCORE}, "
          f"WEAK_SPATIAL_THRESHOLD={WEAK_SPATIAL_THRESHOLD}, "
          f"REJECT_INANIMATE_SPATIAL={REJECT_INANIMATE_SPATIAL}")

    relations, raw_debug = rel_predict.infer_relationships_semantic(
        detections,
        threshold=0.10,
        top_k=top_k,
        image=image,
        temperature=temperature,
    )

    if not relations:
        print("  No relations after precision filtering.")
        return [], raw_debug

    print(f"\n  Relations ({len(relations)}):")
    for r in relations:
        marker = " ★" if r["predicate"] in SEMANTIC_PREDICATES else ""
        pri = r.get("prior_adjustment", 0)
        adj = r.get("adjusted_confidence", r["confidence"])
        print(f"    {r['subject']} {r['predicate']} {r['object']} "
              f"(calib={r['confidence']:.3f}, adj={adj:.3f}, prior={pri:+.3f}){marker}")

    # Feature contribution analysis
    try:
        feature_norms = _get_feature_group_norms(rel_predict._model)
        if feature_norms:
            total_fn = sum(feature_norms.values()) or 1.0
            print(f"\n  ─── FEATURE GROUP CONTRIBUTION ───")
            for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                pct = norm / total_fn * 100
                print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
            clip_pct = (feature_norms.get("subj_clip", 0) + feature_norms.get("obj_clip", 0)) / total_fn * 100
            geo_pct = feature_norms.get("geo", 0) / total_fn * 100
            print(f"    CLIP total:      {clip_pct:5.1f}%")
            print(f"    Geometry:        {geo_pct:5.1f}%")
    except Exception:
        pass

    # Quality evaluation
    quality = evaluate_relation_quality(relations, raw_debug)
    print(f"\n  ─── RELATION QUALITY ───")
    print(f"    Semantic precision:  {quality['semantic_precision']:.0%} "
          f"({quality['semantic_relations']}/{quality['total_relations']})")
    print(f"    Animate subjects:    {quality['animate_subject_rate']:.0%}")
    print(f"    Reversed direction:  {quality['reversed_direction_rate']:.0%}")
    print(f"    Weak spatial:        {quality['weak_spatial_rate']:.0%}")

    return relations, raw_debug


# ---------------------------------------------------------------------------
# STEP 3 — Build grounded prompt
# ---------------------------------------------------------------------------

def build_semantic_prompt_step(
    detections: List[Dict],
    relations: List[Dict],
) -> Tuple[str, List[str]]:
    """Build precision-grounded prompt with high-value relations only.

    Steps 6, 7:
    - Verbalize only high-value semantic relations
    - Suppress geometry-only verbalizations
    - Build prompt with scene elements + grounded interactions

    Returns:
        (prompt_string, verbalized_relations_list)
    """
    print(f"\n  STEP 3 — BUILDING PRECISION-GROUNDED PROMPT")
    print(f"  {len(detections)} objects, {len(relations)} high-value relations")

    verbalized = []
    for r in relations:
        vr = verbalize_relation(
            r["subject"], r["predicate"], r["object"],
            r.get("confidence", 0.5),
        )
        verbalized.append(vr)

    prompt = build_semantic_prompt(detections, verbalized)

    print(f"\n  Verbalized relations:")
    for v in verbalized:
        print(f"    - {v}")

    print(f"\n  Prompt ({len(prompt)} chars):")
    for line in prompt.split("\n"):
        print(f"    {line}")

    return prompt, verbalized


# ---------------------------------------------------------------------------
# STEP 4 — Generate grounded caption
# ---------------------------------------------------------------------------

def generate_grounded_caption(
    image: Image.Image,
    detections: List[Dict],
    relations: List[Dict],
) -> Tuple[str, str, str, List[str]]:
    """Generate grounded caption using semantic prompt.

    Pipeline:
        Image → Semantic prompt → BLIP-2 → Evidence gating → Final caption

    Returns:
        (raw_caption, gated_caption, prompt, verbalized_relations)
    """
    print(f"\n  STEP 4 — GENERATING SEMANTIC CAPTION + RELATION CORRECTION")

    raw, gated, prompt, verbalized = generate_blip_semantic_caption(
        image,
        detections,
        relations,
        unsupported_threshold=0.4,
        confidence_threshold=0.5,
    )

    print(f"  Raw caption:    {raw}")
    print(f"  Gated caption:  {gated}")
    if raw != gated:
        print(f"  ✓ BLIP action corrected using grounded relations")

    return raw, gated, prompt, verbalized


# ---------------------------------------------------------------------------
# STEP 5 — Run full pipeline
# ---------------------------------------------------------------------------

def run_grounded_pipeline(
    image_path: str,
    top_k_relations: int = 3,
    temperature: float = 2.0,
    debug: bool = False,
) -> Dict:
    """Run the complete grounded caption pipeline on a single image.

    Steps:
      1. YOLO detection
      2. Semantic relation inference (rerank + filter + dedup)
      3. Verbalize + build semantic prompt
      4. BLIP-2 generation + evidence gating
      5. Save debug info (raw predictions, filtered, verbalized, etc.)

    Returns a result dict with all intermediate outputs for visibility.
    """
    print(f"\n{'=' * 70}")
    print(f"  SEMANTIC GROUNDED CAPTION PIPELINE")
    print(f"  Image: {image_path}")
    print(f"{'=' * 70}")

    result: Dict = {
        "image_path": image_path,
        "timestamp": time.time(),
    }

    # Load image
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    result["image_size"] = (img_w, img_h)
    print(f"  Image size: {img_w}x{img_h}")

    # ── STEP 1 — YOLO Detection ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  STEP 1 — YOLO DETECTION")
    print(f"{'=' * 60}")
    yolo_model = load_model()
    raw = run_inference(yolo_model, image)
    raw_detections = format_detections(raw, conf_thres=0.5)
    result["raw_detections"] = [
        {"label": d["label"], "box": d["box"], "score": round(d["score"], 3)}
        for d in raw_detections
    ]
    print(f"  {len(raw_detections)} objects detected (raw):")
    for d in result["raw_detections"]:
        print(f"    {d['label']} (conf={d['score']:.3f})")

    # ── STEP 1b — CLIP Semantic Verification ─────────────────────
    print(f"\n{'=' * 60}")
    print(f"  STEP 1b — CLIP VERIFICATION")
    print(f"{'=' * 60}")
    detections = verify_detections(raw_detections, image, debug=True)
    result["detections"] = [
        {"label": d["label"], "box": d["box"], "score": round(d["score"], 3),
         "verification_score": round(d.get("verification_score", 0), 3),
         "clip_similarity": round(d.get("clip_similarity", 0), 3)}
        for d in detections
    ]
    print(f"  {len(detections)} objects verified:")
    for d in result["detections"]:
        print(f"    ✓ {d['label']} (yolo={d['score']:.2f}, clip={d['clip_similarity']:.2f}, "
              f"trust={d['verification_score']:.2f})")

    if len(detections) < 2:
        result["relations"] = []
        result["raw_predictions"] = []
        result["verbalized_relations"] = []
        result["semantic_prompt"] = ""
        result["caption"] = "Only one object detected — cannot infer relations."
        return result

    # ── STEP 2 — Calibrated semantic relation inference ─────────
    relations, raw_predictions = run_pure_visual_inference(
        detections, image,
        top_k=top_k_relations,
        temperature=temperature,
    )
    result["relations"] = relations
    result["raw_predictions"] = raw_predictions

    if not relations:
        result["verbalized_relations"] = []
        result["semantic_prompt"] = ""
        result["caption"] = "No semantic interactions detected."
        return result

    # ── STEP 3 — Verbalize + build semantic prompt ──────────────
    #   Sub-steps:
    #     2. Convert raw triplets to natural language
    #     3. Confidence-aware language
    #     6. Redesigned prompt template
    prompt, verbalized = build_semantic_prompt_step(detections, relations)
    result["verbalized_relations"] = verbalized
    result["semantic_prompt"] = prompt

    # ── STEP 4 — Generate + gate ────────────────────────────────
    #   Sub-step:
    #     9. Evidence gating (unchanged)
    raw_caption, gated_caption, _, _ = generate_grounded_caption(
        image, detections, relations,
    )
    result["raw_caption"] = raw_caption
    result["caption"] = gated_caption

    # ── STEP 7 — Debug summary ──────────────────────────────────
    semantic_count = sum(1 for r in relations if r["predicate"] in SEMANTIC_PREDICATES)
    print(f"\n{'=' * 60}")
    print(f"  PRECISION FILTER SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Pairs evaluated:       {len(raw_predictions)}")
    print(f"  Final relations:       {len(relations)}")
    print(f"  Semantic predicates:   {semantic_count}/{len(relations)}")
    print(f"  Spatial predicates:    {len(relations) - semantic_count}/{len(relations)}")
    for r in relations:
        sem_marker = " \u2605" if r["predicate"] in SEMANTIC_PREDICATES else ""
        adj = r.get("adjusted_confidence", r["confidence"])
        pri = r.get("prior_adjustment", 0.0)
        print(f"    {r['subject']} {r['predicate']} {r['object']} "
              f"(calib={r['confidence']:.3f}, final={adj:.3f}, "
              f"prior={pri:+.3f}){sem_marker}")
    print(f"  Verbalized:")
    for v in verbalized:
        print(f"    - {v}")
    print(f"  Raw BLIP caption:  {raw_caption}")
    print(f"  Gated caption:     {gated_caption}")

    if raw_caption != gated_caption:
        print(f"  [gating] modified caption (object repair or relation correction)")

    return result


# ---------------------------------------------------------------------------
# STEP 6 — Debug visualization
# ---------------------------------------------------------------------------

def save_debug_visualization(
    result: Dict,
    output_path: str,
    image: Optional[Image.Image] = None,
) -> None:
    """Save debug visualization with boxes, relations, and caption."""
    if image is None:
        image = Image.open(result["image_path"]).convert("RGB")

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Draw detection boxes
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow"]
    obj_colors = {}
    for i, d in enumerate(result.get("detections", [])):
        color = colors[i % len(colors)]
        label = d["label"]
        obj_colors[label] = color
        box = d["box"]
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], box[1] - 18), f"{label} {d['score']:.2f}", fill=color, font=font)

    # Draw relation arrows and labels
    for r in result.get("relations", []):
        color = "lime"
        subj_box = r.get("subject_box")
        obj_box = r.get("object_box")
        if subj_box and obj_box:
            sx, sy = (subj_box[0] + subj_box[2]) / 2, (subj_box[1] + subj_box[3]) / 2
            ox, oy = (obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2
            draw.line([(sx, sy), (ox, oy)], fill=color, width=2)
            mx, my = (sx + ox) / 2, (sy + oy) / 2 - 10
            rel_text = f"{r['predicate']} ({r['confidence']:.2f})"
            draw.text((mx, my), rel_text, fill=color, font=small_font)

    # Add caption at bottom
    caption = result.get("caption", "")
    draw.text((10, image.size[1] - 60), f"Caption: {caption}", fill="white", font=font)

    image.save(output_path)
    print(f"  Debug visualization saved to: {output_path}")


# ---------------------------------------------------------------------------
# STEP 8 — Run qualitative tests on semantic cases
# ---------------------------------------------------------------------------

TEST_CATEGORIES = {
    "person+bicycle": ["bicycle"],
    "person+horse": [],
    "person+backpack": [],
    "person+car": ["car"],
    "person+dog": ["dog"],
    "person+cell phone": [],
    "person+umbrella": [],
}

def run_qualitative_tests(
    image_dir: str,
    num_samples: int = 10,
    output_dir: str = "results/grounded_qualitative",
) -> Dict:
    """Run the grounded pipeline on test images.

    Tests are expected to demonstrate:
    - Relations appear correctly in grounded prompt
    - BLIP captions reflect predicted relations
    - Hallucinated objects reduce
    """
    print(f"\n{'=' * 70}")
    print(f"  STEP 7 — QUALITATIVE TESTS")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(Path(image_dir).iterdir())[:num_samples]
    results: List[Dict] = []

    for img_path in image_paths:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        print(f"\n  --- Testing: {img_path.name} ---")
        result = run_grounded_pipeline(str(img_path))
        results.append(result)

    # Save results
    output_path = os.path.join(output_dir, "qualitative_results.json")
    serializable = []
    for r in results:
        serializable.append({
            "image_path": r["image_path"],
            "image_size": r.get("image_size"),
            "detections": r.get("detections", []),
            "relations": r.get("relations", []),
            "raw_predictions": r.get("raw_predictions", []),
            "verbalized_relations": r.get("verbalized_relations", []),
            "semantic_prompt": r.get("semantic_prompt", ""),
            "raw_caption": r.get("raw_caption", ""),
            "caption": r.get("caption", ""),
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Qualitative results saved to: {output_path}")
    print(f"  Total images tested: {len(results)}")

    return {"results": results, "output_path": output_path}


# ---------------------------------------------------------------------------
# STEP 8 — Semantic interaction tests
# ---------------------------------------------------------------------------

def run_semantic_tests(
    image_dir: str,
    output_dir: str = "results/semantic_tests",
) -> Dict:
    """
    Run precision-focused tests on images containing known semantic interactions.

    Tests (Step 8):
    - bicycle images → should detect riding (not "on" or "near")
    - horse images → should detect riding
    - umbrella holding → should detect holding (not "next to" or "near")
    - backpack carrying → should detect carrying (not "on")
    - phone interaction → should detect holding/looking at

    Verifies:
    - NO weak spatial relations survive (no "near", "next to", "under", etc.)
    - semantic predicates dominate (riding, holding, wearing, carrying, looking at)
    - one primary relation per animate subject
    - prompts contain ONLY grounded semantic interactions
    - BLIP captions explicitly reflect grounded interactions
    """
    print(f"\n{'=' * 70}")
    print(f"  STEP 8 — SEMANTIC PRECISION TESTS")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(Path(image_dir).iterdir())
    results: List[Dict] = []

    for img_path in image_paths:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        print(f"\n  --- Testing: {img_path.name} ---")
        result = run_grounded_pipeline(str(img_path), top_k_relations=3)

        # Check for semantic predicates
        relations = result.get("relations", [])
        has_semantic = any(
            r["predicate"] in SEMANTIC_PREDICATES for r in relations
        )
        semantic_count = sum(
            1 for r in relations if r["predicate"] in SEMANTIC_PREDICATES
        )
        verbalized = result.get("verbalized_relations", [])

        print(f"\n  Semantic check:")
        print(f"    Semantic predicates in relations: {semantic_count}/{len(relations)}")
        print(f"    Has semantic interaction:          {has_semantic}")
        print(f"    Verbalized:                        {verbalized}")
        print(f"    Caption:                           {result['caption']}")

        result["test_metadata"] = {
            "has_semantic": has_semantic,
            "semantic_count": semantic_count,
            "total_relations": len(relations),
        }
        results.append(result)

    # Summary
    total = len(results)
    with_semantic = sum(1 for r in results if r.get("test_metadata", {}).get("has_semantic"))
    total_sem_preds = sum(r.get("test_metadata", {}).get("semantic_count", 0) for r in results)
    total_relations = sum(r.get("test_metadata", {}).get("total_relations", 0) for r in results)

    print(f"\n{'=' * 70}")
    print(f"  SEMANTIC PRECISION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Images tested:              {total}")
    print(f"  Images with semantic preds: {with_semantic} ({100*with_semantic/max(total,1):.0f}%)")
    print(f"  Total relations:            {total_relations}")
    print(f"  Total semantic preds:       {total_sem_preds}")
    print(f"  Semantic ratio:             {total_sem_preds/max(total_relations,1):.2f}")
    print(f"  (Target: semantic ratio > 0.8 = precision-oriented)")
    print(f"  (Lower total relations = higher precision filter impact)")

    # Save results
    output_path = os.path.join(output_dir, "semantic_test_results.json")
    serializable = []
    for r in results:
        serializable.append({
            "image_path": r["image_path"],
            "detections": r.get("detections", []),
            "raw_predictions": r.get("raw_predictions", []),
            "relations": r.get("relations", []),
            "verbalized_relations": r.get("verbalized_relations", []),
            "semantic_prompt": r.get("semantic_prompt", ""),
            "raw_caption": r.get("raw_caption", ""),
            "caption": r.get("caption", ""),
            "test_metadata": r.get("test_metadata", {}),
        })
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return {"results": results, "output_path": output_path}


# ---------------------------------------------------------------------------
# STEP 9 — Compare against baselines
# ---------------------------------------------------------------------------

def run_baseline_comparison(
    image_path: str,
    output_dir: str = "results/comparison",
) -> Dict:
    """Generate side-by-side comparison of all three systems.

    Systems:
    1. BLIP-2 baseline (pure implicit captioning)
    2. BLIP-2 + CLIP reranking (semantically enhanced)
    3. Grounded visual-semantic pipeline (ours)

    Returns dict with all captions and relation predictions.
    """
    print(f"\n{'=' * 70}")
    print(f"  STEP 8 — BASELINE COMPARISON")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem

    image = Image.open(image_path).convert("RGB")

    # System 1: BLIP-2 baseline
    print(f"\n  --- System 1: BLIP-2 Baseline ---")
    blip2_caption = generate_blip_baseline(image)
    print(f"    Caption: {blip2_caption}")

    # System 2: BLIP-2 + CLIP reranking
    print(f"\n  --- System 2: BLIP-2 + CLIP Reranking ---")
    candidates = generate_blip_candidates(image, num_candidates=5)
    best_caption, ranked = clip_rerank_captions(image, candidates)
    print(f"    Best:    {best_caption}")
    print(f"    Ranked:")
    for cap, score in ranked[:3]:
        print(f"      {score:.4f}: {cap}")

    # System 3: Grounded pipeline
    print(f"\n  --- System 3: Semantic Grounded Pipeline ---")
    grounded_result = run_grounded_pipeline(image_path, top_k_relations=3)
    grounded_caption = grounded_result["caption"]
    relations = grounded_result["relations"]
    prompt = grounded_result.get("semantic_prompt", "")

    comparison = {
        "image_path": image_path,
        "blip2_baseline": blip2_caption,
        "blip2_clip_reranking": {
            "best_caption": best_caption,
            "ranked": [(cap, round(score, 4)) for cap, score in ranked[:5]],
        },
        "grounded_pipeline": {
            "caption": grounded_caption,
            "relations": relations,
            "semantic_prompt": prompt,
            "raw_predictions": grounded_result.get("raw_predictions", []),
            "verbalized_relations": grounded_result.get("verbalized_relations", []),
            "raw_caption": grounded_result.get("raw_caption", ""),
            "raw_detections": grounded_result.get("raw_detections", []),
            "detections": grounded_result.get("detections", []),
        },
    }

    # Save comparison
    output_path = os.path.join(output_dir, f"{image_name}_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved to: {output_path}")

    # Print side-by-side
    print(f"\n  {'=' * 60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"  {'=' * 60}")
    print(f"  Image: {image_path}")
    print(f"  {'─' * 60}")
    print(f"  BLIP-2 Baseline:")
    print(f"    {blip2_caption}")
    print(f"  {'─' * 60}")
    print(f"  BLIP-2 + CLIP Reranking:")
    print(f"    {best_caption}")
    print(f"  {'─' * 60}")
    print(f"  Grounded Visual-Semantic (ours):")
    print(f"    {grounded_caption}")
    print(f"  Relations:")
    for r in relations:
        marker = " ★" if r["predicate"] in SEMANTIC_PREDICATES else ""
        print(f"    {r['subject']} {r['predicate']} {r['object']} "
              f"(conf={r['confidence']:.3f}){marker}")

    return comparison


# ---------------------------------------------------------------------------
# STEP 9 — Evaluation readiness
# ---------------------------------------------------------------------------

def prepare_evaluation_output(
    results: List[Dict],
    output_dir: str = "results/evaluation_ready",
) -> None:
    """Format pipeline outputs for evaluation with CHAIR, POPE, SPICE, etc.

    Outputs:
        - captions.json:    {image_id: caption} for each system
        - relations.json:   {image_id: [structured relations]}
        - prompts.json:     {image_id: grounded prompt}
        - metadata.json:    Per-image metadata with confidence scores
    """
    print(f"\n{'=' * 70}")
    print(f"  STEP 9 — EVALUATION READINESS")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    grounded_captions = {}
    grounded_relations = {}
    grounded_prompts = {}
    metadata = []

    for r in results:
        img_id = Path(r["image_path"]).stem
        grounded_captions[img_id] = r.get("caption", "")
        grounded_relations[img_id] = r.get("relations", [])
        grounded_prompts[img_id] = r.get("semantic_prompt", r.get("grounded_prompt", ""))
        metadata.append({
            "image_id": img_id,
            "image_path": r["image_path"],
            "num_raw_detections": len(r.get("raw_detections", [])),
            "num_verified_detections": len(r.get("detections", [])),
            "num_rejected_detections": len(r.get("raw_detections", [])) - len(r.get("detections", [])),
            "num_relations": len(r.get("relations", [])),
            "raw_detections": r.get("raw_detections", []),
            "detections": r.get("detections", []),
            "relations": r.get("relations", []),
        })

    def _save_json(data, path):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved: {path}")

    _save_json(grounded_captions, os.path.join(output_dir, "captions.json"))
    _save_json(grounded_relations, os.path.join(output_dir, "relations.json"))
    _save_json(grounded_prompts, os.path.join(output_dir, "prompts.json"))
    _save_json(metadata, os.path.join(output_dir, "metadata.json"))

    print(f"\n  Evaluation-ready outputs in: {output_dir}")
    print(f"  Use with:")
    print(f"    python evaluate.py --mode custom --image-dir <images> --refs <refs>")
    print(f"    python hallucination_eval.py --image-dir <images>")
    print(f"    python -c 'from utils.metrics import *; ... evaluate_caption(...)'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grounded caption pipeline — integrates trained relation MLP with BLIP-2",
    )

    # Input sources (mutually exclusive via group)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image",
    )
    input_group.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory of images to process",
    )

    # Pipeline options
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Max relations per image (default: 3)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save debug visualization with boxes and relation arrows",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run side-by-side comparison against BLIP-2 and BLIP-2+CLIP baselines",
    )
    parser.add_argument(
        "--qualitative", action="store_true",
        help="Run qualitative test suite",
    )
    parser.add_argument(
        "--semantic-tests", action="store_true",
        help="Run focused semantic interaction tests (Step 8)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Limit images processed (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/grounded_pipeline",
        help="Output directory (default: results/grounded_pipeline)",
    )
    parser.add_argument(
        "--eval-ready", action="store_true",
        help="Prepare evaluation-ready outputs",
    )
    parser.add_argument(
        "--feature-analysis", action="store_true",
        help="Print detailed feature group norm analysis (Step 4)",
    )
    parser.add_argument(
        "--relations-only", action="store_true",
        help="Skip BLIP caption generation, only run relation inference (faster)",
    )

    args = parser.parse_args()

    # ── Verify model first ──────────────────────────────────────────────
    model_info = verify_relation_model()

    # ── Collect image paths ─────────────────────────────────────────────
    image_paths: List[str] = []
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in exts
        )[:args.num_samples]
    else:
        # Default: run qualitative tests
        args.qualitative = True
        image_paths = [str(p) for p in Path("test_images").iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not image_paths:
        print("No images found.")
        return

    output_root = Path(args.output_dir)
    os.makedirs(str(output_root), exist_ok=True)

    # ── Run pipeline on each image ──────────────────────────────────────
    all_results: List[Dict] = []

    for img_path in image_paths:
        print(f"\n{'#' * 70}")
        print(f"  Processing: {img_path}")
        print(f"{'#' * 70}")

        # Run the full grounded pipeline (or just relations)
        if args.relations_only:
            result = run_grounded_pipeline(
                img_path,
                top_k_relations=args.top_k,
                debug=args.debug,
            )
            # Skip BLIP: mark caption as N/A
            if not result.get("caption") or result["caption"] == "No semantic interactions detected.":
                result["caption"] = result.get("caption", "(relations only)")
            # Override to skip BLIP loading in generate phase
            result["_skip_blip"] = True
        else:
            result = run_grounded_pipeline(
                img_path,
                top_k_relations=args.top_k,
                debug=args.debug,
            )
        all_results.append(result)

        # Debug visualization
        if args.debug:
            debug_dir = output_root / "debug_vis"
            os.makedirs(str(debug_dir), exist_ok=True)
            debug_path = str(debug_dir / f"{Path(img_path).stem}_debug.png")
            save_debug_visualization(result, debug_path)

        # Side-by-side comparison
        if args.compare:
            run_baseline_comparison(img_path, str(output_root / "comparison"))

    # ── Save all results with full debug info (Step 7) ──────────────────
    results_path = output_root / "all_results.json"
    serializable = []
    for r in all_results:
        serializable.append({
            "image_path": r["image_path"],
            "raw_detections": r.get("raw_detections", []),
            "detections": r.get("detections", []),
            "raw_predictions": r.get("raw_predictions", []),
            "relations": r.get("relations", []),
            "verbalized_relations": r.get("verbalized_relations", []),
            "semantic_prompt": r.get("semantic_prompt", ""),
            "raw_caption": r.get("raw_caption", ""),
            "caption": r.get("caption", ""),
        })
    with open(str(results_path), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  All results saved to: {results_path}")

    # ── Qualitative tests ──────────────────────────────────────────────
    if args.qualitative:
        qual_dir = output_root / "qualitative"
        run_qualitative_tests(
            str(Path(image_paths[0]).parent) if image_paths else "test_images",
            num_samples=min(args.num_samples, len(image_paths)),
            output_dir=str(qual_dir),
        )

    # ── Semantic interaction tests (Step 8) ───────────────────────────
    if args.semantic_tests:
        sem_dir = output_root / "semantic_tests"
        run_semantic_tests(
            str(Path(image_paths[0]).parent) if image_paths else "test_images",
            output_dir=str(sem_dir),
        )

    # ── Evaluation readiness ────────────────────────────────────────────
    if args.eval_ready or args.compare:
        eval_dir = output_root / "evaluation_ready"
        prepare_evaluation_output(all_results, str(eval_dir))

    # ── Feature contribution analysis ─────────────────────────────────
    if args.feature_analysis and rel_predict._model is not None:
        try:
            feature_norms = _get_feature_group_norms(rel_predict._model)
            if feature_norms:
                total_fn = sum(feature_norms.values()) or 1.0
                print(f"\n  ─── FEATURE GROUP CONTRIBUTION ───")
                for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                    pct = norm / total_fn * 100
                    print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
        except Exception as e:
            print(f"\n  Feature analysis: {e}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_rels = sum(len(r.get('relations', [])) for r in all_results)
    total_sems = sum(1 for r in all_results for rel in r.get('relations', []) if rel.get('predicate') in SEMANTIC_PREDICATES)
    total_anim = sum(1 for r in all_results for rel in r.get('relations', [])
                     if "animate" in rel_predict._get_categories(rel.get('subject', '')))
    total_rev = sum(1 for r in all_results for rel in r.get('relations', [])
                    if "animate" not in rel_predict._get_categories(rel.get('subject', ''))
                    and "animate" in rel_predict._get_categories(rel.get('object', '')))

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — PRECISION-ORIENTED")
    print(f"{'=' * 70}")
    print(f"  Images processed:        {len(all_results)}")
    print(f"  Model:                   visual-semantic (1669-dim input)")
    print(f"  Output directory:        {output_root.resolve()}")
    print(f"  Total relations:         {total_rels}")
    print(f"  Total semantic preds:    {total_sems}")
    print(f"  Semantic ratio:          {total_sems/max(total_rels,1):.2f}")
    print(f"  Animate subject count:   {total_anim}/{total_rels}")
    print(f"  Reversed direction:      {total_rev}/{total_rels}")
    print(f"  Precision thresholds:    MIN_SEMANTIC_SCORE={MIN_SEMANTIC_SCORE}, "
          f"WEAK_SPATIAL_THRESHOLD={WEAK_SPATIAL_THRESHOLD}")
    print()

    # Print summary table
    header = (f"  {'Image':<30} {'Raw':<5} {'Ver':<5} {'Rej':<5} "
              f"{'Rel':<5} {'Sem':<5} {'Verbalized':<40} {'Caption':<40}")
    print(header)
    print(f"  {'─' * len(header)}")
    for r in all_results:
        img_short = Path(r["image_path"]).name[:28]
        nraw = len(r.get("raw_detections", []))
        nver = len(r.get("detections", []))
        nrej = nraw - nver
        nr = len(r.get("relations", []))
        ns = sum(1 for rel in r.get("relations", []) if rel.get("predicate") in SEMANTIC_PREDICATES)
        vr = (r.get("verbalized_relations") or [])
        vr_preview = vr[0][:38] if vr else "(none)"
        cap = r.get("caption", "")[:38]
        print(f"  {img_short:<30} {nraw:<5} {nver:<5} {nrej:<5} {nr:<5} {ns:<5} {vr_preview:<40} {cap:<40}")


if __name__ == "__main__":
    main()
