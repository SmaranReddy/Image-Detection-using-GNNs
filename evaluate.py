"""
evaluate.py — Unified evaluation pipeline for grounded captioning.

Compares three systems on the SAME images with the SAME references:
    1. BLIP-2 baseline (pure implicit captioning)
    2. BLIP-2 + CLIP reranking (semantically enhanced implicit captioning)
    3. Grounded relational pipeline (explicit grounded reasoning)

Scientific progression:
    - BLIP-2:          implicit captioning (pixels → caption)
    - BLIP-2 + CLIP:   implicit + semantic visual alignment (reranking)
    - Grounded:        explicit object detection → relation prediction → grounded captioning

Metrics: BLEU, METEOR, BERTScore, SPICE, CHAIR, POPE

    Groundedness evaluation suite centers on:
        SPICE  → relation grounding
        CHAIR  → hallucinated caption objects
        POPE   → object hallucination probing behavior

Usage:
    # COCO validation set evaluation
    python evaluate.py --mode coco --coco-root ./data/coco --num-samples 100

    # Custom image directory (requires reference captions JSON)
    python evaluate.py --mode custom --image-dir test_images --refs captions.json

    # Quick test on test_images (qualitative only, no references)
    python evaluate.py --mode custom --image-dir test_images

    # Limit systems
    python evaluate.py --mode custom --image-dir test_images --systems blip2 blip2_clip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from utils.metrics import (
    evaluate_caption,
    evaluate_all,
    detect_coco_objects,
    COCO_80,
    COCO_80_ID_TO_NAME,
    tokenize,
)


# ---------------------------------------------------------------------------
# Lazy-loaded singletons for BLIP-2 + CLIP reranking
# ---------------------------------------------------------------------------

_blip2_candidates_fn = None
_clip_rerank_fn = None


def _ensure_blip2_clip():
    """Lazy load BLIP-2 candidate generation and CLIP reranking."""
    global _blip2_candidates_fn, _clip_rerank_fn
    if _blip2_candidates_fn is None:
        from utils.blip_captioner import generate_blip_candidates
        _blip2_candidates_fn = generate_blip_candidates
    if _clip_rerank_fn is None:
        from utils.clip_scorer import clip_rerank_captions
        _clip_rerank_fn = clip_rerank_captions


def run_blip2_candidates(
    image: Image.Image,
    num_candidates: int = 5,
) -> List[str]:
    """Generate multiple caption candidates from BLIP-2 for CLIP reranking."""
    _ensure_blip2_clip()
    return _blip2_candidates_fn(image, num_candidates=num_candidates)


def run_clip_rerank(
    image: Image.Image,
    captions: List[str],
) -> Tuple[str, List[Tuple[str, float]]]:
    """Rerank captions by CLIP image-text similarity.

    Returns:
        (best_caption, ranked_list) where ranked_list is [(caption, score), ...]
    """
    _ensure_blip2_clip()
    return _clip_rerank_fn(image, captions)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_image_paths(image_dir: str) -> List[Path]:
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    t = image.float()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    t = t.cpu().numpy()
    if t.max() <= 1.0:
        t = (t * 255).clip(0, 255).astype("uint8")
    else:
        t = t.clip(0, 255).astype("uint8")
    if t.shape[2] == 1:
        t = t[:, :, 0]
    return Image.fromarray(t).convert("RGB")


# ---------------------------------------------------------------------------
# COCO data loading
# ---------------------------------------------------------------------------

COCO_CAP_ANNO_FILENAME = "captions_val2017.json"
COCO_INST_ANNO_FILENAME = "instances_val2017.json"
COCO_IMAGE_DIR = "val2017"


def load_coco_data(
    coco_root: str,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Load COCO val images with captions and object annotations.

    Returns:
        List of dicts:
            image_id, file_path, references ([caption, ...]),
            gt_objects ({class_name, ...})
    """
    cap_path = os.path.join(coco_root, "annotations", COCO_CAP_ANNO_FILENAME)
    inst_path = os.path.join(coco_root, "annotations", COCO_INST_ANNO_FILENAME)
    img_dir = os.path.join(coco_root, COCO_IMAGE_DIR)

    for p, name in [(cap_path, "captions"), (inst_path, "instances"),
                      (img_dir, "images dir")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"COCO {name} not found at: {p}\n"
                f"  Expected COCO structure:\n"
                f"    {coco_root}/annotations/captions_val2017.json\n"
                f"    {coco_root}/annotations/instances_val2017.json\n"
                f"    {coco_root}/val2017/"
            )

    # Load annotations.
    with open(cap_path) as f:
        cap_data = json.load(f)
    with open(inst_path) as f:
        inst_data = json.load(f)

    # Build caption index.
    refs_by_image: Dict[int, List[str]] = {}
    for ann in cap_data["annotations"]:
        refs_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    # Build object index from instances.
    cat_id_to_name: Dict[int, str] = {
        cat["id"]: cat["name"] for cat in inst_data["categories"]
    }
    objs_by_image: Dict[int, Set[str]] = {}
    for ann in inst_data["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name and cat_name in COCO_80:
            objs_by_image.setdefault(img_id, set()).add(cat_name)

    # Build file name index.
    file_by_id: Dict[int, str] = {
        img["id"]: img["file_name"] for img in cap_data["images"]
    }

    # Build sample list.
    sample_ids = sorted(set(refs_by_image.keys()) & set(objs_by_image.keys()))
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]

    samples = []
    for img_id in sample_ids:
        file_name = file_by_id.get(img_id)
        if file_name is None:
            continue
        file_path = os.path.join(img_dir, file_name)
        if not os.path.isfile(file_path):
            continue
        samples.append({
            "image_id": img_id,
            "file_path": file_path,
            "references": refs_by_image[img_id],
            "gt_objects": objs_by_image.get(img_id, set()),
        })

    print(f"[evaluate] Loaded {len(samples)} COCO val samples"
          f" (from {len(sample_ids)} requested)")
    return samples


# ---------------------------------------------------------------------------
# System interfaces (lazy-loaded singletons)
# ---------------------------------------------------------------------------

# YOLO
_yolo_model = None


def _ensure_yolo():
    global _yolo_model
    if _yolo_model is None:
        from utils.yolo_detector import load_model
        print("[evaluate] Loading YOLO ...")
        _yolo_model = load_model()
        print("[evaluate] YOLO ready.")


def run_yolo(image: Image.Image) -> List[Dict]:
    _ensure_yolo()
    from utils.yolo_detector import run_inference, format_detections
    raw = run_inference(_yolo_model, image)
    return format_detections(raw)


# BLIP-2 captioner
def run_blip2(
    image: Image.Image,
    detections: List[Dict],
    relationships: List[Tuple[str, str, str]],
) -> str:
    from utils.blip_captioner import generate_blip_caption
    return generate_blip_caption(image, detections, relationships)


# Relation MLP
_relation_available: Optional[bool] = None


def _check_relation_model() -> bool:
    global _relation_available
    if _relation_available is not None:
        return _relation_available
    ckpt = os.environ.get("REL_CKPT_DIR", "./checkpoints")
    model_path = os.path.join(ckpt, "relation_mlp.pt")
    _relation_available = os.path.isfile(model_path)
    return _relation_available


def run_relations(
    detections: List[Dict],
    image: Image.Image,
) -> List[Tuple[str, str, str]]:
    if not _check_relation_model():
        print("[evaluate] WARNING: relation_mlp.pt not found - skipping relation prediction")
        return []
    from relation_prediction.predict import infer_relationships_learned
    return infer_relationships_learned(detections, image=image)


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------

_SYSTEM_NAMES = {
    "blip2": "BLIP-2 Baseline",
    "blip2_clip": "BLIP-2 + CLIP Reranking",
    "grounded": "Grounded Pipeline",
}


def run_system_blip2(image: Image.Image) -> str:
    """Pure BLIP-2 baseline — implicit captioning only.

    Pipeline: Image → BLIP-2 → Caption

    No grounding, no detection hints, no semantic reranking.
    Uses a clean ungrounded prompt for free-form caption generation.
    """
    from utils.blip_captioner import generate_blip_baseline
    return generate_blip_baseline(image)


def run_system_blip2_clip(
    image: Image.Image,
    num_candidates: int = 5,
) -> Tuple[str, List[str], List[Tuple[str, float]]]:
    """BLIP-2 + CLIP reranking — semantically enhanced implicit captioning.

    Pipeline:
        Image → BLIP-2 (N candidates) → CLIP similarity reranking → Best caption

    This is semantic visual alignment WITHOUT explicit grounding:
    - No YOLO detections in the prompt
    - No object hinting
    - No scene graphs
    - Only CLIP-based image-text similarity for reranking

    Scientific role: Isolates the contribution of semantic visual alignment
    without explicit relational reasoning.

    Args:
        image: PIL Image
        num_candidates: Number of BLIP-2 candidates to generate

    Returns:
        (best_caption, candidates_list, ranked_list_with_scores)
    """
    candidates = run_blip2_candidates(image, num_candidates=num_candidates)

    if not candidates:
        return "", [], []

    best_caption, ranked = run_clip_rerank(image, candidates)

    return best_caption, candidates, ranked


def run_system_grounded(
    image: Image.Image,
    detections: List[Dict],
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Full grounded pipeline — YOLO + CLIP verification + MLP relations + BLIP-2."""
    from utils.detection_verifier import verify_detections
    detections = verify_detections(detections, image, debug=False)
    relations = run_relations(detections, image)
    caption = run_blip2(image, detections, relations)
    return caption, relations


# ---------------------------------------------------------------------------
# CHAIR helper: get GT objects from YOLO when COCO annotations unavailable
# ---------------------------------------------------------------------------

def yolo_detections_to_objects(detections: List[Dict]) -> Set[str]:
    return {d["label"] for d in detections if d["label"] in COCO_80}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def run_evaluation(args):
    output_root = Path(args.output_dir)
    captions_dir = output_root / "captions"
    metrics_dir = output_root / "metrics"
    comparisons_dir = output_root / "comparisons"

    for d in [captions_dir, metrics_dir, comparisons_dir]:
        os.makedirs(str(d), exist_ok=True)

    # ── Hallucination mode defaults ──────────────────────────────────────────
    if args.mode == "hallucination":
        if args.image_dir == "test_images":
            args.image_dir = "hallucinating images"
        print(f"{'=' * 60}")
        print(f"  HALLUCINATION STRESS-TEST EVALUATION")
        print(f"{'=' * 60}")
        print(f"  Image set: {args.image_dir}/")
        print(f"  Purpose:   Stress-test hallucination behavior across 3 systems")
        print(f"  Priority:  CHAIR > POPE > qualitative analysis")
        print(f"  Note:      BLEU/METEOR/BERTScore/SPICE skipped (no references)")
        print(f"{'=' * 60}")
        print()

    # ── Load data ───────────────────────────────────────────────────────────
    if args.mode == "coco":
        samples = load_coco_data(args.coco_root, args.num_samples)
    else:
        image_paths = _list_image_paths(args.image_dir)
        if args.num_samples is not None:
            image_paths = image_paths[:args.num_samples]

        # Try loading reference captions from file if provided.
        refs_dict: Dict[str, List[str]] = {}
        if args.refs:
            with open(args.refs) as f:
                refs_dict = json.load(f)
            print(f"[evaluate] Loaded references for {len(refs_dict)} images")

        samples = []
        for p in image_paths:
            refs = refs_dict.get(p.stem, [])
            samples.append({
                "image_id": p.stem,
                "file_path": str(p),
                "references": refs,
                "gt_objects": set(),
            })
        print(f"[evaluate] Loaded {len(samples)} custom images from '{args.image_dir}'")

    if not samples:
        print("[evaluate] No samples found. Exiting.")
        return

    # ── Determine which systems to run ──────────────────────────────────────
    system_keys = args.systems if args.systems else ["blip2", "blip2_clip", "grounded"]
    for sk in system_keys:
        if sk not in _SYSTEM_NAMES:
            print(f"[evaluate] Unknown system: {sk} (choices: {list(_SYSTEM_NAMES.keys())})")
            return

    # ── Pre-run: YOLO once on all images for CHAIR ground truth ─────────────
    # This ensures ALL systems use the SAME YOLO detections for CHAIR metric.
    # Note: blip2 and blip2_clip don't use YOLO for generation, but CHAIR
    # still needs ground truth objects (from COCO or YOLO proxy).
    needs_yolo = "grounded" in system_keys
    needs_yolo_for_chair = any(
        not s.get("gt_objects") for s in samples
    )
    if needs_yolo or needs_yolo_for_chair:
        print(f"\n  Pre-running YOLO on {len(samples)} images for CHAIR ground truth...")
        for s in tqdm(samples, desc="YOLO pre-run"):
            try:
                image = Image.open(s["file_path"]).convert("RGB")
                dets = run_yolo(image)
                s["yolo_detections"] = dets
                s["yolo_objects"] = yolo_detections_to_objects(dets)
            except Exception as e:
                print(f"\n[skip YOLO] {s['image_id']}: {e}")

    # ── Run each system ──────────────────────────────────────────────────────
    all_results: Dict[str, Dict] = {}

    for sk in system_keys:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {_SYSTEM_NAMES[sk]}")
        print(f"{'=' * 60}")

        candidates: List[str] = []
        metadata: List[Dict] = []
        per_image_captions: Dict[str, str] = {}

        pbar = tqdm(samples, desc=_SYSTEM_NAMES[sk])
        for s in pbar:
            try:
                image = Image.open(s["file_path"]).convert("RGB")
            except Exception as e:
                print(f"\n[skip] {s['image_id']}: {e}")
                continue

            try:
                if sk == "blip2":
                    caption = run_system_blip2(image)
                elif sk == "blip2_clip":
                    caption, candidates, ranked = run_system_blip2_clip(image)
                    s["blip2_clip_candidates"] = candidates
                    s["blip2_clip_ranked"] = ranked
                elif sk == "grounded":
                    dets = s.get("yolo_detections")
                    if dets is None:
                        dets = run_yolo(image)
                        s["yolo_detections"] = dets
                        s["yolo_objects"] = yolo_detections_to_objects(dets)
                    caption, rels = run_system_grounded(image, dets)
                    s["relations"] = rels
                else:
                    caption = "[error: unknown system]"
            except Exception as e:
                caption = f"[error: {e}]"
                import traceback
                traceback.print_exc()

            candidates.append(caption)
            per_image_captions[str(s["image_id"])] = caption
            metadata.append({
                "image_id": s["image_id"],
                "file_path": s["file_path"],
                "candidate": caption,
            })

        # Save per-system captions.
        cap_path = captions_dir / f"{sk}.json"
        _save_json(per_image_captions, str(cap_path))
        print(f"[evaluate] Saved captions to {cap_path}")

        # ── Compute metrics ─────────────────────────────────────────────────
        references_list = [s["references"] for s in samples[:len(candidates)]]

        # Use consistent CHAIR ground truth across ALL systems.
        # Priority: COCO GT > YOLO detections > none.
        has_gt_objects = any(
            s.get("gt_objects") for s in samples[:len(candidates)]
        )
        yolo_has_objects = any(
            s.get("yolo_objects") for s in samples[:len(candidates)]
        )

        if has_gt_objects:
            objects_list = [s["gt_objects"] for s in samples[:len(candidates)]]
            chair_source = "COCO GT annotations"
        elif yolo_has_objects:
            objects_list = [
                s.get("yolo_objects", set()) for s in samples[:len(candidates)]
            ]
            chair_source = "YOLO detections (proxy)"
        else:
            objects_list = None
            chair_source = "none (CHAIR skipped)"

        if objects_list is not None:
            print(f"[evaluate] CHAIR ground truth: {chair_source}")

        result = evaluate_all(
            candidates,
            references_list,
            objects_list=objects_list,
            system_name=_SYSTEM_NAMES[sk],
        )
        result["metadata"] = metadata
        all_results[sk] = result

        # Save per-system metrics.
        metrics_path = metrics_dir / f"{sk}_metrics.json"
        _save_json(result, str(metrics_path))
        print(f"[evaluate] Saved metrics to {metrics_path}")

        # Print aggregate (skip reference-based metrics in hallucination mode).
        agg = result["aggregate"]
        is_hallucination = args.mode == "hallucination"
        print(f"\n  -- {_SYSTEM_NAMES[sk]} {'Hallucination Stress-Test' if is_hallucination else 'Aggregate'} --")
        for k, v in sorted(agg.items()):
            if is_hallucination and k.startswith("avg_") and any(
                ref in k for ref in ["bleu", "meteor", "bertscore", "spice"]
            ):
                continue
            print(f"    {k}: {v}")
        if result["hallucination_summary"]:
            hs = result["hallucination_summary"]
            print(f"  -- Groundedness Metrics --")
            print(f"    CHAIR_i:     {hs.get('avg_chair_i', 'N/A')}")
            print(f"    CHAIR_s:     {hs.get('avg_chair_s', 'N/A')}")
            print(f"    Hall. Rate:  {hs.get('hallucination_rate', 'N/A')}")
            print(f"    Hall. Imgs:  {hs.get('total_images_with_hallucination', 'N/A')}")
            print(f"    POPE F1:     {hs.get('avg_pope_f1', 'N/A')}")
            print(f"    POPE Prec:   {hs.get('avg_pope_precision', 'N/A')}")
            print(f"    POPE Rec:    {hs.get('avg_pope_recall', 'N/A')}")
            print(f"    POPE Acc:    {hs.get('avg_pope_accuracy', 'N/A')}")
            print(f"    Total Hal. Assertions: {hs.get('pope_total_hallucinated_assertions', 'N/A')}")
            print(f"    Total Missed Objects:  {hs.get('pope_total_missed_objects', 'N/A')}")
            if is_hallucination:
                print(f"    (CHAIR/POPE via YOLO proxy — no COCO GT annotations)")

    # ── Cross-system comparison ──────────────────────────────────────────────
    if len(all_results) >= 2:
        print(f"\n{'=' * 60}")
        print(f"  CROSS-SYSTEM COMPARISON")
        print(f"{'=' * 60}")

        comparison = {
            "num_samples": len(samples),
            "systems_compared": list(all_results.keys()),
            "metrics_comparison": {},
            "per_image": [],
        }

        # Aggregate comparison table.
        metric_keys = ["bleu", "meteor", "bertscore_f1", "spice", "spice_f",
                       "chair_i", "chair_s",
                       "pope_precision", "pope_recall", "pope_f1", "pope_accuracy"]
        table_data = {}
        for sk, result in all_results.items():
            agg = result["aggregate"]
            table_data[_SYSTEM_NAMES[sk]] = {
                mk: agg.get(f"avg_{mk}", "N/A")
                for mk in metric_keys
            }
            hs = result.get("hallucination_summary", {})
            table_data[_SYSTEM_NAMES[sk]]["hallucination_rate"] = hs.get(
                "hallucination_rate", "N/A"
            )

        comparison["metrics_comparison"] = table_data

        # Print comparison table.
        print(f"\n  {'Metric':<25}", end="")
        for sk in system_keys:
            print(f" {_SYSTEM_NAMES[sk]:<25}", end="")
        print()

        for mk in metric_keys + ["hallucination_rate"]:
            print(f"  {mk:<25}", end="")
            for sk in system_keys:
                val = table_data[_SYSTEM_NAMES[sk]].get(mk, "N/A")
                if isinstance(val, float):
                    print(f" {val:<25.4f}", end="")
                else:
                    print(f" {str(val):<25}", end="")
            print()

        # ── POPE qualitative diagnostics ─────────────────────────────────
        print(f"\n  -- POPE Probing Diagnostics (first {min(3, len(samples))} images) --")
        for i in range(min(3, len(samples))):
            s = samples[i]
            print(f"  Image: {s['image_id']}")
            print(f"    GT objects: {sorted(s.get('gt_objects', set()))}")
            if s.get("yolo_objects"):
                print(f"    YOLO objects: {sorted(s['yolo_objects'])}")
            for sk in system_keys:
                if i < len(all_results[sk].get("per_image", [])):
                    pi = all_results[sk]["per_image"][i]
                    cand = pi.get("candidate", "")
                    print(f"    [{_SYSTEM_NAMES[sk]}]")
                    print(f"      Caption: {cand[:100]}...")
                    print(f"      POPE precision={pi.get('pope_precision', 'N/A'):>8}"
                          f"  recall={pi.get('pope_recall', 'N/A'):>8}"
                          f"  F1={pi.get('pope_f1', 'N/A'):>8}")
                    if "pope_hallucinated_objects" in pi:
                        print(f"      Hallucinated: {pi['pope_hallucinated_objects']}")
                    if "pope_missed_objects" in pi:
                        print(f"      Missed: {pi['pope_missed_objects']}")
            print()

        # Per-image comparison for qualitative analysis.
        for i, s in enumerate(samples):
            if i >= (args.qualitative_max or 10):
                break
            entry = {
                "image_id": str(s["image_id"]),
                "file_path": s["file_path"],
                "references": s["references"][:3],  # limit to 3 refs
            }
            for sk in system_keys:
                if i < len(all_results[sk].get("per_image", [])):
                    pi = all_results[sk]["per_image"][i]
                    entry[f"{sk}_caption"] = pi.get("candidate", "")
                    for mk in metric_keys:
                        if mk in pi:
                            entry[f"{sk}_{mk}"] = pi[mk]

            # Add POPE probing diagnostics per system.
            for sk in system_keys:
                if i < len(all_results[sk].get("per_image", [])):
                    pi = all_results[sk]["per_image"][i]
                    if "pope_hallucinated_objects" in pi:
                        entry[f"{sk}_pope_hallucinated"] = pi.get("pope_hallucinated_objects", [])
                    if "pope_missed_objects" in pi:
                        entry[f"{sk}_pope_missed"] = pi.get("pope_missed_objects", [])
                    if "pope_num_positive_probes" in pi:
                        entry[f"{sk}_pope_pos_probes"] = pi["pope_num_positive_probes"]
                    if "pope_num_negative_probes" in pi:
                        entry[f"{sk}_pope_neg_probes"] = pi["pope_num_negative_probes"]

            # Add detection info if available.
            if s.get("yolo_detections"):
                entry["yolo_detections"] = [
                    {"label": d["label"], "score": round(d["score"], 3)}
                    for d in s["yolo_detections"][:5]
                ]
            if s.get("relations"):
                entry["relations"] = [
                    f"{r[0]} {r[1]} {r[2]}" for r in s["relations"]
                ]

            # Add BLIP-2 + CLIP reranking info for qualitative analysis.
            if s.get("blip2_clip_ranked"):
                entry["blip2_clip_reranking"] = [
                    {"caption": cap, "clip_similarity": round(score, 4)}
                    for cap, score in s["blip2_clip_ranked"]
                ]

            comparison["per_image"].append(entry)

        comp_path = comparisons_dir / "comparison.json"
        _save_json(comparison, str(comp_path))
        print(f"\n[evaluate] Saved comparison to {comp_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    is_hallucination = args.mode == "hallucination"
    print(f"\n{'=' * 60}")
    print(f"  {'HALLUCINATION STRESS-TEST COMPLETE' if is_hallucination else 'EVALUATION COMPLETE'}")
    print(f"{'=' * 60}")
    print(f"  Output directory: {output_root.resolve()}")
    print(f"  Samples evaluated: {len(samples)}")
    print(f"  Systems compared: {', '.join(_SYSTEM_NAMES[sk] for sk in system_keys)}")
    if is_hallucination:
        print(f"  Primary metrics: CHAIR, POPE (groundedness-focused)")
        print(f"  Reference metrics: SKIPPED (no references in stress-test set)")
        print(f"  For detailed per-image analysis, see:")
        print(f"    python hallucination_eval.py")
    else:
        print(f"  Metrics computed: BLEU, METEOR, BERTScore, SPICE, CHAIR, POPE")
    if args.mode == "custom" and not args.refs and not is_hallucination:
        print(f"  WARNING: No reference captions provided. Metrics will be incomplete.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation pipeline for grounded captioning.",
    )

    # Data mode
    parser.add_argument(
        "--mode", choices=["coco", "custom", "hallucination"], default="custom",
        help="Evaluation data source (default: custom, hallucination: stress-test suite)",
    )

    # COCO options
    parser.add_argument(
        "--coco-root", type=str, default="./data/coco",
        help="COCO dataset root (requires annotations/ and val2017/)",
    )

    # Custom options
    parser.add_argument(
        "--image-dir", type=str, default="test_images",
        help="Directory with test images (custom/hallucination mode)",
    )
    parser.add_argument(
        "--refs", type=str, default=None,
        help="JSON file mapping image filenames to reference captions",
    )

    # General options
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit number of images to evaluate",
    )
    parser.add_argument(
        "--systems", type=str, nargs="+",
        choices=list(_SYSTEM_NAMES.keys()), default=None,
        help="Systems to evaluate (default: all three)",
    )
    parser.add_argument(
        "--qualitative-max", type=int, default=10,
        help="Max images in qualitative analysis (default: 10; hallucination mode: all 250)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Hallucination mode defaults: evaluate all images, show all qualitative.
    if args.mode == "hallucination":
        if args.qualitative_max == 10:
            args.qualitative_max = 250  # show all in stress-test mode

    # Set seeds.
    from utils.seed import set_seed
    set_seed(args.seed)

    print(f"[evaluate] Starting evaluation (mode={args.mode})")
    print(f"[evaluate] Systems: {args.systems or 'all three'}")
    print(f"[evaluate] Output: {args.output_dir}")

    start = time.time()
    run_evaluation(args)
    elapsed = time.time() - start
    print(f"[evaluate] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
