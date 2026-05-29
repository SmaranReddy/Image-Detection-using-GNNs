"""
evaluate_versions.py — V0 vs V1 vs V2 semantic evaluation.
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

CHECKPOINT_DIRS = {}
for v in ["V0", "V1", "V2"]:
    ckpt = f"./checkpoints" if v == "V0" else f"./checkpoints_{v.lower()}"
    if os.path.exists(os.path.join(ckpt, "relation_mlp.pt")):
        CHECKPOINT_DIRS[v] = ckpt
# Include sweep checkpoints
SWEEP_DIR = "./checkpoints_sweep"
if os.path.isdir(SWEEP_DIR):
    for entry in sorted(os.listdir(SWEEP_DIR)):
        sweep_ckpt = os.path.join(SWEEP_DIR, entry, "relation_mlp.pt")
        if os.path.exists(sweep_ckpt):
            vname = f"sweep_{entry}"
            CHECKPOINT_DIRS[vname] = os.path.join(SWEEP_DIR, entry)

SEMANTIC_PREDS = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})
WEAK_SPATIAL = frozenset({
    "under", "above", "over", "inside", "next to", "near",
    "attached to", "behind", "in front of", "covering",
})
NEUTRAL_SPATIAL = frozenset({"on", "in"})
ANIMATE = frozenset({
    "person", "dog", "horse", "cat", "bird",
    "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
})


def classify_predicate(pred: str) -> str:
    if pred in SEMANTIC_PREDS: return "semantic"
    if pred in WEAK_SPATIAL: return "weak_spatial"
    if pred in NEUTRAL_SPATIAL: return "neutral_spatial"
    return "other"


def _reset_relation_model():
    import relation_prediction.predict as rp
    rp._model = None
    rp._label_vocab = None
    rp._pred_vocab = None
    rp._device = None
    rp._clip_model = None
    rp._pose_model = None
    rp._model_clip_dim = 0
    rp._model_pose_dim = 0
    rp._model_pose_object_dim = 0
    rp._model_union_dim = 0
    rp._model_type = "mlp"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_version_on_image(
    checkpoint_dir: str, image_path: str, top_k: int = 5, temperature: float = 2.0,
) -> Dict:
    _reset_relation_model()
    import relation_prediction.predict as rp
    rp.load_relation_model(checkpoint_dir)

    from utils.yolo_detector import load_model, run_inference, format_detections
    from utils.detection_verifier import verify_detections
    from relation_prediction.predict import (
        infer_relationships_semantic, evaluate_relation_quality,
        _get_feature_group_norms,
    )

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    yolo_model = load_model()
    raw = run_inference(yolo_model, image)
    raw_detections = format_detections(raw, conf_thres=0.5)
    detections = verify_detections(raw_detections, image, debug=False)

    result = {
        "image_path": image_path,
        "image_size": (img_w, img_h),
        "detections": [{"label": d["label"], "box": d["box"], "score": round(d["score"], 3)} for d in detections],
        "relations": [], "raw_predictions": [], "predicate_counts": {}, "quality": {},
    }

    if len(detections) >= 2:
        relations, raw_predictions = infer_relationships_semantic(
            detections, threshold=0.05, top_k=top_k,
            image=image, temperature=temperature, debug=False,
            img_w=img_w, img_h=img_h,
        )
        result["relations"] = relations
        result["raw_predictions"] = raw_predictions

        counts = Counter()
        for r in relations:
            counts[r["predicate"]] += 1
        result["predicate_counts"] = dict(counts)

        quality = evaluate_relation_quality(relations, raw_predictions)
        result["quality"] = quality

        try:
            norms = _get_feature_group_norms(rp._model)
            if norms:
                result["feature_norms"] = {k: round(v, 4) for k, v in norms.items()}
        except Exception:
            pass

        # Capture rejected pairs for failure analysis
        rejected = []
        for rp_info in raw_predictions:
            status = rp_info.get("status", "")
            if "rejected" in status:
                rejected.append({
                    "subject": rp_info.get("subject", "?"),
                    "object": rp_info.get("object", "?"),
                    "status": status,
                    "predicate": rp_info.get("best_predicate", ""),
                    "reason": rp_info.get("reject_reason", rp_info.get("reason", "")),
                })
        result["rejected_pairs"] = rejected

    return result


def run_full_evaluation(checkpoint_dirs: Dict[str, str], image_dir: str, output_dir: str = "evaluation_report"):
    os.makedirs(output_dir, exist_ok=True)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in image_exts
    ])
    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Checkpoints available: {list(checkpoint_dirs.keys())}")

    # Shared detection
    from utils.yolo_detector import load_model, run_inference, format_detections
    from utils.detection_verifier import verify_detections
    yolo_model = load_model()
    image_metadata = {}
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        raw = run_inference(yolo_model, image)
        raw_detections = format_detections(raw, conf_thres=0.5)
        detections = verify_detections(raw_detections, image, debug=False)
        image_metadata[img_path] = {
            "detections": [d["label"] for d in detections],
            "num_raw": len(raw_detections),
            "num_verified": len(detections),
        }
        print(f"  {Path(img_path).name}: {len(raw_detections)} raw -> {len(detections)} verified [{', '.join(d['label'] for d in detections)}]")

    # Run each version
    all_results = {}
    for version_name, ckpt_dir in sorted(checkpoint_dirs.items()):
        print(f"\n{'='*60}")
        print(f"  EVALUATING: {version_name} ({ckpt_dir})")
        print(f"{'='*60}")

        version_results = {}
        for img_path in image_paths:
            img_name = Path(img_path).name
            print(f"  [{version_name}] {img_name}...", end=" ", flush=True)
            try:
                start = time.time()
                result = run_version_on_image(ckpt_dir, img_path)
                elapsed = time.time() - start
                rels = result["relations"]
                if rels:
                    rel_str = "; ".join(f"{r['subject']}->{r['predicate']}->{r['object']}" for r in rels)
                else:
                    rel_str = "(none)"
                print(f"{len(rels)} rels in {elapsed:.1f}s [{rel_str}]")
                version_results[img_path] = result
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
        all_results[version_name] = version_results

    if not all_results:
        print("No results collected. Exiting.")
        return

    # ── AGGREGATE STATISTICS ──
    print(f"\n{'='*60}")
    print(f"  AGGREGATE PREDICATE DISTRIBUTION")
    print(f"{'='*60}")

    aggregate = {}
    for vname, vresults in all_results.items():
        predicate_counts = Counter()
        semantic_count = weak_spatial_count = neutral_spatial_count = 0
        total_pairs = 0
        total_rejected = 0
        rejection_reasons = Counter()

        for img_path, result in vresults.items():
            for r in result.get("relations", []):
                predicate_counts[r["predicate"]] += 1
                cat = classify_predicate(r["predicate"])
                if cat == "semantic": semantic_count += 1
                elif cat == "weak_spatial": weak_spatial_count += 1
                elif cat == "neutral_spatial": neutral_spatial_count += 1
            total_pairs += len(result.get("raw_predictions", []))
            for rp in result.get("rejected_pairs", []):
                total_rejected += 1
                rejection_reasons[rp["status"]] += 1

        total_relations = sum(predicate_counts.values())
        aggregate[vname] = {
            "total_relations": total_relations,
            "total_images": len(vresults),
            "semantic_count": semantic_count,
            "weak_spatial_count": weak_spatial_count,
            "neutral_spatial_count": neutral_spatial_count,
            "predicate_distribution": dict(predicate_counts.most_common()),
            "total_pairs_evaluated": total_pairs,
            "total_rejected": total_rejected,
            "rejection_reasons": dict(rejection_reasons),
        }

        print(f"\n  --- {vname} ---")
        print(f"  Relations: {total_relations} (from {total_pairs} pairs, {total_rejected} rejected)")
        sem_pct = semantic_count / max(total_relations, 1) * 100
        print(f"  Semantic: {semantic_count} ({sem_pct:.0f}%)  |  Weak spatial: {weak_spatial_count}  |  Neutral spatial: {neutral_spatial_count}")
        print(f"  Top predicates:")
        for pred, count in predicate_counts.most_common(10):
            cat = classify_predicate(pred)
            markers = {"semantic": "[SEM]", "weak_spatial": "[W]", "neutral_spatial": "[N]"}
            m = markers.get(cat, "")
            print(f"    {pred:18s} {m}: {count:2d} ({count/max(total_relations,1)*100:.0f}%)")

    # ── QUALITATIVE TABLE ──
    print(f"\n{'='*60}")
    print(f"  QUALITATIVE COMPARISON")
    print(f"{'='*60}")

    vnames = sorted(all_results.keys())
    header = f"{'Image':20s}"
    for vn in vnames:
        header += f" {vn:30s}"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for img_path in image_paths:
        img_name = Path(img_path).stem
        meta = image_metadata.get(img_path, {})
        dets_str = ", ".join(meta.get("detections", []))

        row_data = {"image": img_name, "detections": dets_str, "per_version": {}}
        line = f"{img_name:20s}"

        for vn in vnames:
            if img_path in all_results.get(vn, {}):
                rels = all_results[vn][img_path].get("relations", [])
                if rels:
                    rel_strs = [f"{r['subject']}->{r['predicate']}->{r['object']}" for r in rels]
                else:
                    rel_strs = ["(none)"]
                row_data["per_version"][vn] = rel_strs[:3]
                display = "; ".join(rel_strs[:2])[:28]
            else:
                display = "(error)"
            line += f" {display:30s}"

        print(line)
        comparison_rows.append(row_data)

    # ── FEATURE UTILIZATION ──
    print(f"\n{'='*60}")
    print(f"  FEATURE UTILIZATION")
    print(f"{'='*60}")

    feature_summary = {}
    for vname in all_results:
        all_norms = []
        for img_path in image_paths:
            norms = all_results[vname].get(img_path, {}).get("feature_norms", {})
            if norms:
                all_norms.append(norms)
        if all_norms:
            avg_norms = {}
            for key in all_norms[0]:
                vals = [n.get(key, 0) for n in all_norms]
                avg_norms[key] = round(sum(vals) / len(vals), 4)
            total = sum(avg_norms.values()) or 1.0
            feature_summary[vname] = {
                "avg_norms": avg_norms,
                "pct": {k: round(v / total * 100, 1) for k, v in sorted(avg_norms.items(), key=lambda x: -x[1])},
            }
            print(f"\n  --- {vname} ---")
            for name, pct in feature_summary[vname]["pct"].items():
                print(f"    {name:18s}: {pct:5.1f}%")

    # ── FAILURE ANALYSIS ──
    print(f"\n{'='*60}")
    print(f"  FAILURE ANALYSIS")
    print(f"{'='*60}")

    for vname in all_results:
        all_rejected = []
        all_no_relation = []
        for img_path in image_paths:
            result = all_results[vname].get(img_path, {})
            dets = image_metadata.get(img_path, {}).get("detections", [])
            rels = result.get("relations", [])
            rejected = result.get("rejected_pairs", [])

            if len(dets) >= 2 and not rels:
                all_no_relation.append(Path(img_path).name)
            for r in rejected:
                all_rejected.append(f"{Path(img_path).name}: {r['subject']}+{r['object']} -> {r['status']} ({r.get('reason','')})")

        print(f"\n  --- {vname} ---")
        print(f"  Images with no relations (despite >=2 dets): {len(all_no_relation)}")
        for item in all_no_relation:
            print(f"    - {item}")
        if all_rejected:
            print(f"  Rejected pairs ({len(all_rejected)}):")
            for item in all_rejected[:10]:
                print(f"    - {item}")

    # ── PER-PREDICATE HEAT MAP ──
    print(f"\n{'='*60}")
    print(f"  SEMANTIC PREDICATE HEAT MAP")
    print(f"{'='*60}")

    all_predicates = sorted(set(
        p for v in aggregate.values()
        for p in v["predicate_distribution"]
    ))
    header = f"{'Predicate':20s}"
    for vn in vnames:
        header += f" {vn:>20s}"
    header += f" {'Type':10s}"
    print(header)
    print("-" * len(header))

    for pred in all_predicates:
        cat = classify_predicate(pred)
        cat_label = cat[:8]
        line = f"{pred:20s}"
        for vn in vnames:
            count = aggregate[vn]["predicate_distribution"].get(pred, 0)
            total = aggregate[vn]["total_relations"]
            pct = count / max(total, 1) * 100
            line += f" {count:3d}({pct:4.1f}%)"
        line += f" {cat_label:10s}"
        print(line)

    # ── DELTA ANALYSIS ──
    if len(vnames) >= 2:
        print(f"\n{'='*60}")
        print(f"  DELTA ANALYSIS")
        print(f"{'='*60}")

        v_first, v_second = vnames[0], vnames[-1]
        first_preds = aggregate[v_first]["predicate_distribution"]
        second_preds = aggregate[v_second]["predicate_distribution"]
        first_total = aggregate[v_first]["total_relations"]
        second_total = aggregate[v_second]["total_relations"]

        changes = []
        for pred in sorted(set(list(first_preds.keys()) + list(second_preds.keys()))):
            first_pct = first_preds.get(pred, 0) / max(first_total, 1) * 100
            second_pct = second_preds.get(pred, 0) / max(second_total, 1) * 100
            changes.append((second_pct - first_pct, pred, first_pct, second_pct))
        changes.sort(reverse=True)

        print(f"  {v_second} vs {v_first}:")
        print(f"  Increased:")
        for delta, pred, fp, sp in changes[:5]:
            arrow = "UP" if delta > 0 else "dn"
            print(f"    {arrow} {pred:18s}: {fp:5.1f}% -> {sp:5.1f}% (delta={delta:+.1f})")
        print(f"  Decreased:")
        for delta, pred, fp, sp in reversed(changes[-5:]):
            arrow = "UP" if delta > 0 else "dn"
            print(f"    {arrow} {pred:18s}: {fp:5.1f}% -> {sp:5.1f}% (delta={delta:+.1f})")

        first_sem = aggregate[v_first]["semantic_count"]
        second_sem = aggregate[v_second]["semantic_count"]
        print(f"\n  {v_second} semantic: {second_sem} (vs {v_first}: {first_sem})")
        if second_sem > first_sem:
            print(f"  [OK] {v_second} improved semantic predicate detection")
        elif second_sem < first_sem:
            print(f"  [REGRESSION] {v_second} semantic detection decreased")
        else:
            print(f"  [SAME] No change in semantic count")

    # ── SAVE RESULTS ──
    output = {
        "config": {"image_dir": image_dir, "checkpoint_dirs": dict(checkpoint_dirs)},
        "image_metadata": image_metadata,
        "aggregate": aggregate,
        "comparison_table": comparison_rows,
        "feature_summary": feature_summary,
    }
    json_path = os.path.join(output_dir, "version_comparison.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # ── MARKDOWN REPORT ──
    md_path = os.path.join(output_dir, "version_comparison.md")
    _save_md(output, md_path, comparison_rows, vnames)
    print(f"  Saved: {md_path}")
    return output


def _save_md(data, md_path, comparison_rows, vnames):
    lines = []
    def w(s=""): lines.append(s)

    w("# V0 vs V1 vs V2 Evaluation Report\n")
    w(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    w("## Configuration\n")
    for vn, ckpt in data["config"]["checkpoint_dirs"].items():
        w(f"- **{vn}**: `{ckpt}`")
    w(f"- **Image directory**: `{data['config']['image_dir']}`\n")

    w("## Aggregate Predicate Distribution\n")
    ag = data["aggregate"]
    w("| Metric | " + " | ".join(ag.keys()) + " |")
    w("|--------|" + "|".join("---" for _ in ag) + "|")
    for m in ["total_relations", "semantic_count", "weak_spatial_count", "neutral_spatial_count"]:
        w(f"| {m} | " + " | ".join(str(ag[v].get(m, 0)) for v in ag) + " |")
    w("")

    w("### Per-Predicate Distribution\n")
    all_predicates = sorted(set(p for v in ag.values() for p in v["predicate_distribution"]))
    w("| Predicate | " + " | ".join(v for v in ag) + " | Type |")
    w("|-----------|" + "|".join("---" for _ in ag) + "|------|")
    for pred in all_predicates:
        vals = []
        for vname in ag:
            c = ag[vname]["predicate_distribution"].get(pred, 0)
            t = ag[vname]["total_relations"]
            pct = c / max(t, 1) * 100
            vals.append(f"{c} ({pct:.1f}%)")
        w(f"| {pred} | {' | '.join(vals)} | {classify_predicate(pred)} |")
    w("")

    w("## Qualitative Comparison\n")
    w("| Image | Detections | " + " | ".join(vnames) + " |")
    w("|-------|-----------|" + "|".join("---" for _ in vnames) + "|")
    for row in comparison_rows:
        dets = row["detections"][:30]
        rel_strs = []
        for vn in vnames:
            rels = row["per_version"].get(vn, ["?"])
            preds = [r.split("->")[1].strip() if "->" in r else r for r in rels[:2]]
            rel_strs.append(", ".join(preds))
        w(f"| {row['image']} | {dets} | {' | '.join(rel_strs)} |")
    w("")

    w("## Detailed Per-Image Output\n")
    for vname in vnames:
        w(f"### {vname}\n")
        for img_path in sorted(data.get("_per_image", {}).get(vname, {})):
            pass  # detailed output omitted for brevity

    w("## Feature Utilization\n")
    fs = data.get("feature_summary", {})
    if fs:
        for vname, finfo in fs.items():
            w(f"### {vname}\n")
            for name, pct in finfo["pct"].items():
                w(f"- {name}: {pct}%\n")
            w("")

    if len(vnames) >= 2:
        w("## Version Delta\n")
        v_first, v_second = vnames[0], vnames[-1]
        fp = ag[v_first]["predicate_distribution"]
        sp = ag[v_second]["predicate_distribution"]
        ft = ag[v_first]["total_relations"]
        st = ag[v_second]["total_relations"]
        changes = []
        for pred in sorted(set(list(fp.keys()) + list(sp.keys()))):
            delta = (sp.get(pred, 0) / max(st, 1) * 100) - (fp.get(pred, 0) / max(ft, 1) * 100)
            changes.append((delta, pred))
        changes.sort(reverse=True)
        w(f"**{v_second} vs {v_first}:**\n")
        for delta, pred in changes[:5]:
            w(f"- {pred}: delta={delta:+.1f}%\n")

    with open(md_path, "w") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="V0 vs V1 vs V2 evaluation")
    parser.add_argument("--image-dir", default="test_images")
    parser.add_argument("--output-dir", default="evaluation_report")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    run_full_evaluation(CHECKPOINT_DIRS, args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
