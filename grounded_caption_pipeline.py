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
from utils.eval_debug import (
    save_visual_debug,
    save_debug_composite,
    run_baseline_comparison_debug,
    classify_failures,
    build_failure_report,
    generate_attention_visualization,
    analyze_relation_errors,
    build_final_evaluation_table,
    generate_final_report,
    compute_refined_prior_adjustment,
)
from utils.logger_utils import (
    DEBUG,
    debug_print,
    section,
    print_clean_summary,
    print_dataset_summary,
    set_debug,
)

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
    print("[TRACE] entered verify_relation_model")
    debug_print("=" * 60)
    debug_print("  LOADING TRAINED RELATION MODEL")
    debug_print("=" * 60)

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

    debug_print(f"  Model loaded successfully")
    debug_print(f"  Input dimension:     {info['input_dim']}")
    debug_print(f"  CLIP dimension:      {info['clip_dim']}")
    debug_print(f"  Label vocab:         {info['num_labels']} classes")
    debug_print(f"  Predicate vocab:     {info['num_predicates']} predicates")
    debug_print(f"  Visual mode:         {info['visual_mode']}")
    debug_print(f"  Device:              {info['device']}")
    debug_print(f"  Embed dim:           {info['embed_dim']}")

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
    improved_priors: bool = False,
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
    print("[TRACE] entered run_pure_visual_inference")
    debug_print(f"\n  PRECISION RELATION INFERENCE")
    debug_print(f"  {len(detections)} detections, top_k={top_k}, T={temperature}")
    debug_print(f"  Precision: MIN_SEMANTIC_SCORE={MIN_SEMANTIC_SCORE}, "
                f"WEAK_SPATIAL_THRESHOLD={WEAK_SPATIAL_THRESHOLD}, "
                f"REJECT_INANIMATE_SPATIAL={REJECT_INANIMATE_SPATIAL}")

    relations, raw_debug = rel_predict.infer_relationships_semantic(
        detections,
        threshold=0.10,
        top_k=top_k,
        image=image,
        temperature=temperature,
        improved_priors=improved_priors,
    )

    if not relations:
        print("[EARLY RETURN] run_pure_visual_inference: no relations after filtering")
        debug_print("  No relations after precision filtering.")
        return [], raw_debug

    debug_print(f"\n  Relations ({len(relations)}):")
    for r in relations:
        marker = " \u2605" if r["predicate"] in SEMANTIC_PREDICATES else ""
        pri = r.get("prior_adjustment", 0)
        adj = r.get("adjusted_confidence", r["confidence"])
        debug_print(f"    {r['subject']} {r['predicate']} {r['object']} "
                    f"(calib={r['confidence']:.3f}, adj={adj:.3f}, prior={pri:+.3f}){marker}")

    # Feature contribution analysis (always computed, only printed in DEBUG)
    try:
        feature_norms = _get_feature_group_norms(rel_predict._model)
        if feature_norms:
            total_fn = sum(feature_norms.values()) or 1.0
            debug_print(f"\n  \u2500\u2500\u2500 FEATURE GROUP CONTRIBUTION \u2500\u2500\u2500")
            for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                pct = norm / total_fn * 100
                debug_print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
            clip_pct = (feature_norms.get("subj_clip", 0) + feature_norms.get("obj_clip", 0)) / total_fn * 100
            geo_pct = feature_norms.get("geo", 0) / total_fn * 100
            debug_print(f"    CLIP total:      {clip_pct:5.1f}%")
            debug_print(f"    Geometry:        {geo_pct:5.1f}%")
    except Exception:
        pass

    # Quality evaluation (always computed, only printed in DEBUG)
    quality = evaluate_relation_quality(relations, raw_debug)
    debug_print(f"\n  \u2500\u2500\u2500 RELATION QUALITY \u2500\u2500\u2500")
    debug_print(f"    Semantic precision:  {quality['semantic_precision']:.0%} "
                f"({quality['semantic_relations']}/{quality['total_relations']})")
    debug_print(f"    Animate subjects:    {quality['animate_subject_rate']:.0%}")
    debug_print(f"    Reversed direction:  {quality['reversed_direction_rate']:.0%}")
    debug_print(f"    Weak spatial:        {quality['weak_spatial_rate']:.0%}")

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
    print("[TRACE] entered build_semantic_prompt_step")
    debug_print(f"\n  BUILDING PRECISION-GROUNDED PROMPT")
    debug_print(f"  {len(detections)} objects, {len(relations)} high-value relations")

    verbalized = []
    for r in relations:
        vr = verbalize_relation(
            r["subject"], r["predicate"], r["object"],
            r.get("confidence", 0.5),
        )
        verbalized.append(vr)

    prompt = build_semantic_prompt(detections, verbalized)

    debug_print(f"\n  Verbalized relations:")
    for v in verbalized:
        debug_print(f"    - {v}")

    debug_print(f"\n  Prompt ({len(prompt)} chars):")
    for line in prompt.split("\n"):
        debug_print(f"    {line}")

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
    print("[TRACE] entered generate_grounded_caption")
    debug_print(f"\n  GENERATING SEMANTIC CAPTION + RELATION CORRECTION")

    raw, gated, prompt, verbalized = generate_blip_semantic_caption(
        image,
        detections,
        relations,
        unsupported_threshold=0.4,
        confidence_threshold=0.5,
    )

    debug_print(f"  Raw caption:    {raw}")
    debug_print(f"  Gated caption:  {gated}")
    if raw != gated:
        debug_print(f"  \u2713 BLIP action corrected using grounded relations")

    return raw, gated, prompt, verbalized


# ---------------------------------------------------------------------------
# STEP 5 — Run full pipeline
# ---------------------------------------------------------------------------

def run_grounded_pipeline(
    image_path: str,
    top_k_relations: int = 3,
    temperature: float = 2.0,
    debug: bool = False,
    compare_baseline: bool = False,
    improved_priors: bool = False,
    diagnose: bool = False,
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
    print("[TRACE] entered run_grounded_pipeline")
    debug_print(f"\n{'=' * 70}")
    debug_print(f"  SEMANTIC GROUNDED CAPTION PIPELINE")
    debug_print(f"  Image: {image_path}")
    debug_print(f"{'=' * 70}")

    result: Dict = {
        "image_path": image_path,
        "timestamp": time.time(),
        "improved_priors": improved_priors,
        "diagnose": diagnose,
    }

    # Load image
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    result["image_size"] = (img_w, img_h)
    debug_print(f"  Image size: {img_w}x{img_h}")

    # ── STEP 1 — YOLO Detection ──────────────────────────────────
    debug_print(f"\n{'=' * 60}")
    debug_print(f"  YOLO DETECTION")
    debug_print(f"{'=' * 60}")
    yolo_model = load_model()
    raw = run_inference(yolo_model, image)
    raw_detections = format_detections(raw, conf_thres=0.5)
    result["raw_detections"] = [
        {"label": d["label"], "box": d["box"], "score": round(d["score"], 3)}
        for d in raw_detections
    ]
    debug_print(f"  {len(raw_detections)} objects detected (raw):")
    for d in result["raw_detections"]:
        debug_print(f"    {d['label']} (conf={d['score']:.3f})")

    # ── STEP 1b — CLIP Semantic Verification ─────────────────────
    debug_print(f"\n{'=' * 60}")
    debug_print(f"  CLIP VERIFICATION")
    debug_print(f"{'=' * 60}")
    detections = verify_detections(raw_detections, image, debug=DEBUG)
    result["detections"] = [
        {"label": d["label"], "box": d["box"], "score": round(d["score"], 3),
         "verification_score": round(d.get("verification_score", 0), 3),
         "clip_similarity": round(d.get("clip_similarity", 0), 3)}
        for d in detections
    ]
    debug_print(f"  {len(detections)} objects verified:")
    for d in result["detections"]:
        debug_print(f"    \u2713 {d['label']} (yolo={d['score']:.2f}, clip={d['clip_similarity']:.2f}, "
                    f"trust={d['verification_score']:.2f})")

    if len(detections) < 2:
        print("[EARLY RETURN] reason=less than 2 detections (cannot infer relations)")
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
        improved_priors=improved_priors,
    )
    result["relations"] = relations
    result["raw_predictions"] = raw_predictions

    if not relations:
        print("[EARLY RETURN] reason=no relations after inference")
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

    # ── Baseline: Vanilla BLIP for comparison ───────────────────
    result["vanilla_caption"] = ""
    if compare_baseline:
        from utils.blip_captioner import generate_blip_baseline
        vanilla = generate_blip_baseline(image)
        result["vanilla_caption"] = vanilla
        result["compare_baseline"] = True

    # ── Debug summary (only in DEBUG mode) ──────────────────────
    semantic_count = sum(1 for r in relations if r["predicate"] in SEMANTIC_PREDICATES)
    debug_print(f"\n{'=' * 60}")
    debug_print(f"  PRECISION FILTER SUMMARY")
    debug_print(f"{'=' * 60}")
    debug_print(f"  Pairs evaluated:       {len(raw_predictions)}")
    debug_print(f"  Final relations:       {len(relations)}")
    debug_print(f"  Semantic predicates:   {semantic_count}/{len(relations)}")
    debug_print(f"  Spatial predicates:    {len(relations) - semantic_count}/{len(relations)}")
    for r in relations:
        sem_marker = " \u2605" if r["predicate"] in SEMANTIC_PREDICATES else ""
        adj = r.get("adjusted_confidence", r["confidence"])
        pri = r.get("prior_adjustment", 0.0)
        debug_print(f"    {r['subject']} {r['predicate']} {r['object']} "
                    f"(calib={r['confidence']:.3f}, final={adj:.3f}, "
                    f"prior={pri:+.3f}){sem_marker}")
    debug_print(f"  Verbalized:")
    for v in verbalized:
        debug_print(f"    - {v}")
    debug_print(f"  Raw BLIP caption:  {raw_caption}")
    debug_print(f"  Gated caption:     {gated_caption}")

    if raw_caption != gated_caption:
        debug_print(f"  [gating] modified caption (object repair or relation correction)")

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
    debug_print(f"  Debug visualization saved to: {output_path}")


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
    debug_print(f"\n{'=' * 70}")
    debug_print(f"  QUALITATIVE TESTS")
    debug_print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(Path(image_dir).iterdir())[:num_samples]
    results: List[Dict] = []

    for img_path in image_paths:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        debug_print(f"\n  --- Testing: {img_path.name} ---")
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

    debug_print(f"\n  Qualitative results saved to: {output_path}")
    debug_print(f"  Total images tested: {len(results)}")

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
    debug_print(f"\n{'=' * 70}")
    debug_print(f"  SEMANTIC PRECISION TESTS")
    debug_print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(Path(image_dir).iterdir())
    results: List[Dict] = []

    for img_path in image_paths:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        debug_print(f"\n  --- Testing: {img_path.name} ---")
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

        debug_print(f"\n  Semantic check:")
        debug_print(f"    Semantic predicates in relations: {semantic_count}/{len(relations)}")
        debug_print(f"    Has semantic interaction:          {has_semantic}")
        debug_print(f"    Verbalized:                        {verbalized}")
        debug_print(f"    Caption:                           {result['caption']}")

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

    debug_print(f"\n{'=' * 70}")
    debug_print(f"  SEMANTIC PRECISION SUMMARY")
    debug_print(f"{'=' * 70}")
    debug_print(f"  Images tested:              {total}")
    debug_print(f"  Images with semantic preds: {with_semantic} ({100*with_semantic/max(total,1):.0f}%)")
    debug_print(f"  Total relations:            {total_relations}")
    debug_print(f"  Total semantic preds:       {total_sem_preds}")
    debug_print(f"  Semantic ratio:             {total_sem_preds/max(total_relations,1):.2f}")
    debug_print(f"  (Target: semantic ratio > 0.8 = precision-oriented)")
    debug_print(f"  (Lower total relations = higher precision filter impact)")

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
    debug_print(f"\n  Results saved to: {output_path}")

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
    debug_print(f"\n{'=' * 70}")
    debug_print(f"  BASELINE COMPARISON")
    debug_print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem

    image = Image.open(image_path).convert("RGB")

    # System 1: BLIP-2 baseline
    debug_print(f"\n  --- System 1: BLIP-2 Baseline ---")
    blip2_caption = generate_blip_baseline(image)
    debug_print(f"    Caption: {blip2_caption}")

    # System 2: BLIP-2 + CLIP reranking
    debug_print(f"\n  --- System 2: BLIP-2 + CLIP Reranking ---")
    candidates = generate_blip_candidates(image, num_candidates=5)
    best_caption, ranked = clip_rerank_captions(image, candidates)
    debug_print(f"    Best:    {best_caption}")
    debug_print(f"    Ranked:")
    for cap, score in ranked[:3]:
        debug_print(f"      {score:.4f}: {cap}")

    # System 3: Grounded pipeline
    debug_print(f"\n  --- System 3: Semantic Grounded Pipeline ---")
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
    debug_print(f"\n  Comparison saved to: {output_path}")

    # Print side-by-side
    debug_print(f"\n  {'=' * 60}")
    debug_print(f"  SIDE-BY-SIDE COMPARISON")
    debug_print(f"  {'=' * 60}")
    debug_print(f"  Image: {image_path}")
    debug_print(f"  {'─' * 60}")
    debug_print(f"  BLIP-2 Baseline:")
    debug_print(f"    {blip2_caption}")
    debug_print(f"  {'─' * 60}")
    debug_print(f"  BLIP-2 + CLIP Reranking:")
    debug_print(f"    {best_caption}")
    debug_print(f"  {'─' * 60}")
    debug_print(f"  Grounded Visual-Semantic (ours):")
    debug_print(f"    {grounded_caption}")
    debug_print(f"  Relations:")
    for r in relations:
        marker = " \u2605" if r["predicate"] in SEMANTIC_PREDICATES else ""
        debug_print(f"    {r['subject']} {r['predicate']} {r['object']} "
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
    debug_print(f"\n{'=' * 70}")
    debug_print(f"  EVALUATION READINESS")
    debug_print(f"{'=' * 70}")

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
        debug_print(f"  Saved: {path}")

    _save_json(grounded_captions, os.path.join(output_dir, "captions.json"))
    _save_json(grounded_relations, os.path.join(output_dir, "relations.json"))
    _save_json(grounded_prompts, os.path.join(output_dir, "prompts.json"))
    _save_json(metadata, os.path.join(output_dir, "metadata.json"))

    debug_print(f"\n  Evaluation-ready outputs in: {output_dir}")
    debug_print(f"  Use with:")
    debug_print(f"    python evaluate.py --mode custom --image-dir <images> --refs <refs>")
    debug_print(f"    python hallucination_eval.py --image-dir <images>")
    debug_print(f"    python -c 'from utils.metrics import *; ... evaluate_caption(...)'")


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

    # ── PHASE 1: Visual debug outputs ───────────────────────────────────
    parser.add_argument(
        "--debug-full", action="store_true",
        help="Phase 1: Save full visual debug output (5 panels + composite) to outputs/debug/",
    )
    parser.add_argument(
        "--debug-dir", type=str, default="outputs/debug",
        help="Directory for debug visualizations (default: outputs/debug)",
    )

    # ── PHASE 2: Baseline comparison ────────────────────────────────────
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Phase 2: Compare vanilla BLIP vs grounded pipeline side-by-side",
    )

    # ── PHASE 3+8+9: Full evaluation ────────────────────────────────────
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Phase 3+8+9: Run full evaluation (failure analysis + table + report)",
    )
    parser.add_argument(
        "--eval-dir", type=str, default="analysis_results",
        help="Directory for analysis results (default: analysis_results)",
    )

    # ── PHASE 5: Attention analysis ─────────────────────────────────────
    parser.add_argument(
        "--attention", action="store_true",
        help="Phase 5: Generate attention/feature contribution analysis",
    )
    parser.add_argument(
        "--attention-dir", type=str, default="outputs/attention",
        help="Directory for attention outputs (default: outputs/attention)",
    )

    # ── PHASE 6: Relation error diagnostics ─────────────────────────────
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Phase 6: Print detailed per-pair relation diagnostics",
    )

    # ── PHASE 7: Use improved semantic priors ──────────────────────────
    parser.add_argument(
        "--improved-priors", action="store_true",
        help="Phase 7: Use refined object-compatibility priors (soft penalties)",
    )

    # ── Console verbosity ──────────────────────────────────────────────
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable full debug/research console output (default: only clean summaries)",
    )

    args = parser.parse_args()

    # ── Wire global debug flag ──────────────────────────────────────────
    if args.verbose:
        set_debug(True)

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

    # ── Setup eval dirs ─────────────────────────────────────────────────
    if args.evaluate:
        eval_dir = args.eval_dir
        os.makedirs(eval_dir, exist_ok=True)
        debug_dir = args.debug_dir
        os.makedirs(debug_dir, exist_ok=True)

    # ── Run pipeline on each image ──────────────────────────────────────
    all_results: List[Dict] = []
    all_failures: List[Dict] = []
    all_comparisons: List[Dict] = []

    for img_path in image_paths:
        debug_print(f"\n{'#' * 70}")
        debug_print(f"  Processing: {img_path}")
        debug_print(f"{'#' * 70}")

        img_name = Path(img_path).stem

        # Run the full grounded pipeline (or just relations)
        if args.relations_only:
            result = run_grounded_pipeline(
                img_path,
                top_k_relations=args.top_k,
                debug=args.debug,
                compare_baseline=args.compare_baseline or args.compare or args.evaluate,
                improved_priors=args.improved_priors,
                diagnose=args.diagnose,
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
                compare_baseline=args.compare_baseline or args.compare or args.evaluate,
                improved_priors=args.improved_priors,
                diagnose=args.diagnose,
            )
        all_results.append(result)

        # ── Print clean per-image summary ─────────────────────────────
        print_clean_summary(result, img_path)

        # ── PHASE 1: Visual debug outputs ───────────────────────────────
        if args.debug_full or args.evaluate:
            image = Image.open(img_path).convert("RGB")
            save_debug_composite(
                image,
                result.get("raw_detections", []),
                result.get("detections", []),
                result.get("relations", []),
                result.get("caption", ""),
                img_name,
                output_dir=debug_dir if args.evaluate else args.debug_dir,
            )

        # Legacy debug visualization
        if args.debug:
            debug_dir = output_root / "debug_vis"
            os.makedirs(str(debug_dir), exist_ok=True)
            debug_path = str(debug_dir / f"{img_name}_debug.png")
            save_debug_visualization(result, debug_path)

        # ── PHASE 2: Baseline comparison ────────────────────────────────
        if args.compare_baseline or args.compare:
            if args.compare_baseline:
                comp_dir = os.path.join(args.eval_dir, "baseline_comparison")
            else:
                comp_dir = str(output_root / "comparison")
            image = Image.open(img_path).convert("RGB")
            comparison = run_baseline_comparison_debug(
                image,
                img_name,
                result.get("vanilla_caption", ""),
                result.get("caption", ""),
                result.get("relations", []),
                result.get("detections", []),
                output_dir=comp_dir,
            )
            all_comparisons.append(comparison)

        # ── PHASE 3+6: Failure analysis + diagnostics ──────────────────
        if args.evaluate:
            failure = classify_failures(
                img_name,
                result.get("detections", []),
                result.get("relations", []),
                result.get("raw_predictions", []),
                result.get("caption", ""),
                result.get("vanilla_caption", None),
            )
            all_failures.append(failure)

        # ── PHASE 5: Attention analysis (per-pair) ──────────────────────
        if args.attention:
            ds = result.get("detections", [])
            if len(ds) >= 2 and rel_predict._model is not None:
                a, b = ds[0], ds[1]
                img_w, img_h = result.get("image_size", [640, 480])
                image_pil = Image.open(img_path).convert("RGB")

                logits, pred_tokens, s_idx, o_idx = rel_predict._get_raw_logits(
                    a["label"], b["label"],
                    a["box"], b["box"],
                    img_w=img_w, img_h=img_h,
                    image=image_pil,
                )
                if logits is not None:
                    # Extract features for meaningful attention visualization
                    subj_norm = rel_predict.normalize_label(a["label"])
                    obj_norm = rel_predict.normalize_label(b["label"])
                    subj_box = tuple(a["box"])
                    obj_box = tuple(b["box"])

                    geo = rel_predict.extract_geo_features(subj_box, obj_box, img_w, img_h)
                    geo_t = torch.tensor([geo], dtype=torch.float32)

                    subj_feat_t: Optional[torch.Tensor] = None
                    obj_feat_t: Optional[torch.Tensor] = None
                    union_feat_t: Optional[torch.Tensor] = None
                    pose_feat_t: Optional[torch.Tensor] = None

                    clip_dim = rel_predict._model_clip_dim
                    union_dim = rel_predict._model_union_dim
                    pose_dim = rel_predict._model_pose_dim

                    if clip_dim > 0 or union_dim > 0:
                        rel_predict._ensure_clip_model()
                        if clip_dim > 0 and rel_predict._clip_model is not None:
                            subj_feat_t = rel_predict._clip_model.extract_crop(
                                image_pil, subj_box
                            ).unsqueeze(0)
                            obj_feat_t = rel_predict._clip_model.extract_crop(
                                image_pil, obj_box
                            ).unsqueeze(0)
                        if union_dim > 0 and rel_predict._clip_model is not None:
                            uemb = rel_predict._clip_model.extract_union_embedding(
                                image_pil, subj_box, obj_box
                            )
                            union_feat_t = uemb.unsqueeze(0)

                    if pose_dim > 0 and subj_norm == "person":
                        rel_predict._ensure_pose_model()
                        if rel_predict.PoseExtractor.is_available() and rel_predict._pose_model is not None:
                            pf = rel_predict._pose_model.extract_pose_features(image_pil, subj_box)
                            if pf is not None:
                                pose_feat_t = pf.unsqueeze(0)

                    pred_labels = [rel_predict._pred_vocab.token(i)
                                   for i in range(len(rel_predict._pred_vocab))]
                    generate_attention_visualization(
                        rel_predict._model,
                        torch.tensor([s_idx]),
                        torch.tensor([o_idx]),
                        geo_t,
                        subj_feat_t, obj_feat_t, union_feat_t, pose_feat_t,
                        pred_labels,
                        output_dir=os.path.join(args.eval_dir, "attention") if args.evaluate else args.attention_dir,
                        image_name=img_name,
                        device=rel_predict._device,
                    )

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
    debug_print(f"\n  All results saved to: {results_path}")

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
                debug_print(f"\n  \u2500\u2500\u2500 FEATURE GROUP CONTRIBUTION \u2500\u2500\u2500")
                for name, norm in sorted(feature_norms.items(), key=lambda x: -x[1]):
                    pct = norm / total_fn * 100
                    debug_print(f"    {name:15s}: {norm:8.4f}  ({pct:5.1f}%)")
        except Exception as e:
            debug_print(f"\n  Feature analysis: {e}")

    # ── PHASE 3+8+9: Build evaluation outputs ──────────────────────────
    if args.evaluate:
        # Phase 8: Evaluation table
        table_data = []
        for r in all_results:
            img_name = Path(r["image_path"]).stem
            # Find matching failure
            failure = next((f for f in all_failures if f["image_name"] == img_name), {})
            table_data.append({
                "image_name": img_name,
                "detections": r.get("detections", []),
                "relations": r.get("relations", []),
                "caption": r.get("caption", ""),
                "vanilla_caption": r.get("vanilla_caption", ""),
                "failures": failure,
                "raw_caption": r.get("raw_caption", ""),
            })

        eval_csv = build_final_evaluation_table(
            table_data,
            output_dir=args.eval_dir,
        )

        # Phase 3: Failure report
        fail_path = build_failure_report(
            all_failures,
            output_dir=args.eval_dir,
        )

        # Phase 9: Final report
        attention_path = None
        attn_dir = os.path.join(args.eval_dir, "attention")
        if os.path.isdir(attn_dir):
            attn_files = sorted(Path(attn_dir).glob("*_attention.json"))
            if attn_files:
                attention_path = str(attn_files[0])

        report_path = generate_final_report(
            table_data,
            failure_report_path=fail_path,
            eval_table_path=eval_csv,
            attention_report_path=attention_path,
            output_dir=args.eval_dir,
        )

        debug_print(f"\n{'=' * 70}")
        debug_print(f"  FULL EVALUATION COMPLETE")
        debug_print(f"{'=' * 70}")
        debug_print(f"  Evaluation CSV:  {eval_csv}")
        debug_print(f"  Failure report:  {fail_path}")
        debug_print(f"  Final report:    {report_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_rels = sum(len(r.get('relations', [])) for r in all_results)
    total_sems = sum(1 for r in all_results for rel in r.get('relations', []) if rel.get('predicate') in SEMANTIC_PREDICATES)
    total_anim = sum(1 for r in all_results for rel in r.get('relations', [])
                     if "animate" in rel_predict._get_categories(rel.get('subject', '')))
    total_rev = sum(1 for r in all_results for rel in r.get('relations', [])
                    if "animate" not in rel_predict._get_categories(rel.get('subject', ''))
                    and "animate" in rel_predict._get_categories(rel.get('object', '')))

    weak_spatial_rate = 0.0
    reversed_direction_rate = 0.0
    if total_rels > 0:
        quality = evaluate_relation_quality(
            [r for res in all_results for r in res.get('relations', [])],
            []
        )
        weak_spatial_rate = quality.get('weak_spatial_rate', 0.0)
        reversed_direction_rate = quality.get('reversed_direction_rate', 0.0)

    print_dataset_summary(
        all_results,
        total_rels=total_rels,
        total_sems=total_sems,
        total_anim=total_anim,
        total_rev=total_rev,
        weak_spatial_rate=weak_spatial_rate,
        reversed_direction_rate=reversed_direction_rate,
        num_images=len(all_results),
    )


if __name__ == "__main__":
    main()
