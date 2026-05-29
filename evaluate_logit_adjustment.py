"""
evaluate_logit_adjustment.py — BEFORE vs AFTER comparison for logit adjustment.

Isolates the effect of predicate logit adjustment (Menon et al. ICLR 2021)
on semantic relation selection.

Usage:
    python evaluate_logit_adjustment.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

sys.path.insert(0, ".")

import relation_prediction.predict as predict_mod
from evaluate import run_yolo, _list_image_paths
from utils.logger_utils import debug_print

# ── Config ───────────────────────────────────────────────────────────────
IMAGE_DIRS = ["test_images", "hallucinating/hal30"]
OUTPUT_DIR = "results/logit_adjustment_eval"
TOP_K = 3

# Predicates of interest
SEMANTIC_PREDS = {"riding", "holding", "carrying", "wearing", "looking at", "sitting on", "standing on"}
GENERIC_SPATIAL = {"on", "in", "under", "above", "near", "next to", "behind", "in front of", "over", "inside", "attached to", "covering"}
INTEREST_PREDS = ["on", "near", "behind", "sitting on", "riding", "wearing", "holding", "carrying", "looking at", "standing on"]


def run_inference(
    image: Image.Image,
    detections: List[Dict],
    logit_adjustment: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Run semantic relation inference with or without logit adjustment."""
    predict_mod.ENABLE_LOGIT_ADJUSTMENT = logit_adjustment
    img_w, img_h = image.size
    relations, raw_debug = predict_mod.infer_relationships_semantic(
        detections,
        threshold=0.05,
        img_w=img_w,
        img_h=img_h,
        top_k=TOP_K,
        image=image,
        temperature=2.0,
        debug=False,
        improved_priors=False,
    )
    return relations, raw_debug


def extract_per_pair_logits(
    image: Image.Image,
    detections: List[Dict],
    logit_adjustment: bool,
) -> Dict[Tuple[str, str], Dict]:
    """Extract full predicate scores for each (subject, object) pair."""
    predict_mod.ENABLE_LOGIT_ADJUSTMENT = logit_adjustment
    img_w, img_h = image.size
    pair_data: Dict[Tuple[str, str], Dict] = {}
    n = len(detections)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = detections[i]
            b = detections[j]

            subj_norm = predict_mod.normalize_label(a["label"])
            obj_norm = predict_mod.normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            # Get raw logits (will apply adjustment based on flag)
            logits, pred_tokens, _, _ = predict_mod._get_raw_logits(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                image=image,
            )
            if logits is None:
                continue

            # Collect scores per predicate
            scores = {}
            for pidx, pname in enumerate(pred_tokens):
                if pname in predict_mod.Vocab.PAD or pname in predict_mod.Vocab.UNK:
                    continue
                if pname not in predict_mod.ALLOWED_PREDICATES:
                    continue
                scores[pname] = round(logits[pidx].item(), 4)

            if scores:
                pair_data[(subj_norm, obj_norm)] = {
                    "scores": scores,
                    "top1": max(scores, key=scores.get),
                    "top1_score": max(scores.values()),
                }

    return pair_data


def analyze_image(
    image_path: str,
    image_id: str,
    detections: List[Dict],
) -> Dict:
    """Full BEFORE vs AFTER analysis for one image."""
    image = Image.open(image_path).convert("RGB")

    # ── BEFORE: no logit adjustment ──
    before_relations, before_raw = run_inference(image, detections, logit_adjustment=False)
    before_pairs = extract_per_pair_logits(image, detections, logit_adjustment=False)

    # ── AFTER: with logit adjustment ──
    after_relations, after_raw = run_inference(image, detections, logit_adjustment=True)
    after_pairs = extract_per_pair_logits(image, detections, logit_adjustment=True)

    # ── Build predicate distribution counts ──
    before_pred_counts: Counter = Counter()
    after_pred_counts: Counter = Counter()
    for r in before_relations:
        before_pred_counts[r["predicate"]] += 1
    for r in after_relations:
        after_pred_counts[r["predicate"]] += 1

    # ── Per-pair top1 changes ──
    pair_changes = []
    all_pairs = set(before_pairs.keys()) | set(after_pairs.keys())
    for pair in sorted(all_pairs):
        before_info = before_pairs.get(pair, {})
        after_info = after_pairs.get(pair, {})
        before_top1 = before_info.get("top1", "N/A") if before_info else "N/A"
        after_top1 = after_info.get("top1", "N/A") if after_info else "N/A"
        before_score = before_info.get("top1_score", 0) if before_info else 0
        after_score = after_info.get("top1_score", 0) if after_info else 0

        if before_top1 != after_top1:
            pair_changes.append({
                "pair": f"{pair[0]}+{pair[1]}",
                "before": before_top1,
                "after": after_top1,
                "before_score": before_score,
                "after_score": after_score,
            })

    # ── Semantic predicate analysis ──
    before_semantic_count = sum(1 for r in before_relations if r["predicate"] in SEMANTIC_PREDS)
    after_semantic_count = sum(1 for r in after_relations if r["predicate"] in SEMANTIC_PREDS)
    before_generic_count = sum(1 for r in before_relations if r["predicate"] in GENERIC_SPATIAL)
    after_generic_count = sum(1 for r in after_relations if r["predicate"] in GENERIC_SPATIAL)

    return {
        "image_id": image_id,
        "file_path": image_path,
        "before_relations": [f"{r['subject']} {r['predicate']} {r['object']}" for r in before_relations],
        "after_relations": [f"{r['subject']} {r['predicate']} {r['object']}" for r in after_relations],
        "before_full": before_relations,
        "after_full": after_relations,
        "before_pred_distribution": dict(before_pred_counts),
        "after_pred_distribution": dict(after_pred_counts),
        "before_semantic_count": before_semantic_count,
        "after_semantic_count": after_semantic_count,
        "before_generic_count": before_generic_count,
        "after_generic_count": after_generic_count,
        "pair_changes": pair_changes,
        "detections": [{"label": d["label"], "score": round(d["score"], 3)} for d in detections],
    }


def aggregate_results(per_image: List[Dict]) -> Dict:
    """Aggregate results across all images."""
    before_total_counts: Counter = Counter()
    after_total_counts: Counter = Counter()
    total_pair_changes = []
    total_before_semantic = 0
    total_after_semantic = 0
    total_before_generic = 0
    total_after_generic = 0
    total_before_relations = 0
    total_after_relations = 0

    for img in per_image:
        before_total_counts.update(img.get("before_pred_distribution", {}))
        after_total_counts.update(img.get("after_pred_distribution", {}))
        total_pair_changes.extend(img.get("pair_changes", []))
        total_before_semantic += img.get("before_semantic_count", 0)
        total_after_semantic += img.get("after_semantic_count", 0)
        total_before_generic += img.get("before_generic_count", 0)
        total_after_generic += img.get("after_generic_count", 0)
        total_before_relations += len(img.get("before_relations", []))
        total_after_relations += len(img.get("after_relations", []))

    # ── Interest predicate comparison ──
    interest_comparison = {}
    for pred in INTEREST_PREDS:
        interest_comparison[pred] = {
            "before": before_total_counts.get(pred, 0),
            "after": after_total_counts.get(pred, 0),
        }

    # ── Semantic vs generic summary ──
    semantic_vs_generic = {
        "before_semantic": total_before_semantic,
        "after_semantic": total_after_semantic,
        "before_generic": total_before_generic,
        "after_generic": total_after_generic,
        "before_semantic_ratio": round(total_before_semantic / max(total_before_relations, 1), 4),
        "after_semantic_ratio": round(total_after_semantic / max(total_after_relations, 1), 4),
    }

    # ── Pair change summary ──
    change_summary = defaultdict(list)
    for change in total_pair_changes:
        key = f"{change['before']} → {change['after']}"
        change_summary[key].append(change["pair"])

    return {
        "num_images": len(per_image),
        "total_before_relations": total_before_relations,
        "total_after_relations": total_after_relations,
        "before_pred_distribution": dict(before_total_counts.most_common()),
        "after_pred_distribution": dict(after_total_counts.most_common()),
        "interest_predicates": interest_comparison,
        "semantic_vs_generic": semantic_vs_generic,
        "total_pair_changes": len(total_pair_changes),
        "pair_change_summary": dict(change_summary),
        "pair_changes": total_pair_changes,
    }


def print_report(agg: Dict, title: str):
    """Print a formatted comparison report."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Images: {agg['num_images']}")
    print(f"  Relations: BEFORE={agg['total_before_relations']}, AFTER={agg['total_after_relations']}")

    # ── 1. Predicate distribution ──
    print(f"\n  ── 1. Predicate Distribution ──")
    print(f"  {'Predicate':<20} {'BEFORE':<10} {'AFTER':<10} {'Δ':<10}")
    all_preds = sorted(set(list(agg["before_pred_distribution"].keys()) + list(agg["after_pred_distribution"].keys())))
    # Show interest predicates first, then others
    for pred in INTEREST_PREDS + [p for p in all_preds if p not in INTEREST_PREDS]:
        b = agg["before_pred_distribution"].get(pred, 0)
        a = agg["after_pred_distribution"].get(pred, 0)
        if b == 0 and a == 0:
            continue
        delta = a - b
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {pred:<20} {b:<10} {a:<10} {delta_str:<10}")

    # ── 2. Interest predicates ──
    print(f"\n  ── 2. Interest Predicates (Detailed) ──")
    for pred in INTEREST_PREDS:
        info = agg["interest_predicates"].get(pred, {"before": 0, "after": 0})
        delta = info["after"] - info["before"]
        arrow = "+" if delta > 0 else ""
        print(f"  {pred:<20}  BEFORE={info['before']:<3}  AFTER={info['after']:<3}  ({arrow}{delta})")

    # ── 3. Semantic vs Generic Spatial ──
    svg = agg["semantic_vs_generic"]
    print(f"\n  ── 3. Semantic vs Generic Spatial ──")
    print(f"  {'Metric':<30} {'BEFORE':<12} {'AFTER':<12}")
    print(f"  {'Semantic count':<30} {svg['before_semantic']:<12} {svg['after_semantic']:<12}")
    print(f"  {'Generic spatial count':<30} {svg['before_generic']:<12} {svg['after_generic']:<12}")
    print(f"  {'Semantic ratio':<30} {svg['before_semantic_ratio']:<12.4f} {svg['after_semantic_ratio']:<12.4f}")

    # ── 4. Top-1 Predicate Changes ──
    print(f"\n  ── 4. Top-1 Predicate Changes ({agg['total_pair_changes']} total) ──")
    for change_type, pairs in sorted(agg["pair_change_summary"].items(), key=lambda x: -len(x[1])):
        print(f"  {change_type:<30}  ({len(pairs)} pairs)")
        for pair in pairs[:5]:
            print(f"    {pair}")
        if len(pairs) > 5:
            print(f"    ... and {len(pairs) - 5} more")

    print()


def save_results(per_image: List[Dict], agg: Dict, output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    per_image_dir = os.path.join(output_dir, "per_image")
    os.makedirs(per_image_dir, exist_ok=True)
    for img in per_image:
        path = os.path.join(per_image_dir, f"{img['image_id']}.json")
        with open(path, "w") as f:
            json.dump(img, f, indent=2)

    agg_path = os.path.join(output_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"[save] Per-image results: {per_image_dir}/")
    print(f"[save] Aggregate results: {agg_path}")


def main():
    output_dir = OUTPUT_DIR

    for img_dir in IMAGE_DIRS:
        if not os.path.isdir(img_dir):
            print(f"[skip] Directory not found: {img_dir}")
            continue

        image_paths = _list_image_paths(img_dir)
        if not image_paths:
            print(f"[skip] No images in: {img_dir}")
            continue

        print(f"\n{'#' * 60}")
        print(f"  Evaluating: {img_dir}/ ({len(image_paths)} images)")
        print(f"{'#' * 60}")

        per_image = []
        for p in image_paths:
            image_id = p.stem
            print(f"\n  [{image_id}] Loading...")

            try:
                image = Image.open(str(p)).convert("RGB")
                detections = run_yolo(image)
                if len(detections) < 2:
                    print(f"  [{image_id}] Skipped: only {len(detections)} detection(s)")
                    continue

                result = analyze_image(str(p), image_id, detections)
                per_image.append(result)

                # Print quick summary
                before_str = ", ".join(result["before_relations"]) or "(none)"
                after_str = ", ".join(result["after_relations"]) or "(none)"
                print(f"  [{image_id}] BEFORE: {before_str}")
                print(f"  [{image_id}] AFTER:  {after_str}")

                changes = result.get("pair_changes", [])
                for c in changes:
                    print(f"  [{image_id}] CHANGE: {c['pair']}: {c['before']} → {c['after']}")

            except Exception as e:
                import traceback
                print(f"  [{image_id}] ERROR: {e}")
                traceback.print_exc()
                continue

        if per_image:
            agg = aggregate_results(per_image)
            print_report(agg, f"BEFORE vs AFTER Logit Adjustment — {img_dir}")
            save_results(per_image, agg, os.path.join(output_dir, Path(img_dir).name))

    print(f"\n{'=' * 60}")
    print(f"  DONE. Results in {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
