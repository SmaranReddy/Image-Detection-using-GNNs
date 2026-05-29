"""
Comprehensive behavioral evaluation and analysis of V0 vs V2.

Produces:
- Predicate-specific analysis (precision, frequency, confidence)
- Confidence histograms (accepted vs rejected)
- Failure categorization (A/B/C/D)
- Feature utilization comparisons
- Qualitative comparison tables
- Calibration analysis
"""

import json
import math
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

# ── Predicate family classification (mirrors predict.py) ──────────────
PREDICATE_FAMILIES: Dict[str, Dict] = {
    "strong_semantic": {
        "predicates": {"riding", "wearing", "sitting on", "standing on", "holding", "carrying"},
        "base_conf": 0.27,
        "base_margin": 0.08,
        "relax_min": 0.50,
        "relax_max": 1.00,
    },
    "attentional": {
        "predicates": {"looking at"},
        "base_conf": 0.32,
        "base_margin": 0.15,
        "relax_min": 0.70,
        "relax_max": 1.00,
    },
    "neutral_spatial": {
        "predicates": {"on", "in"},
        "base_conf": 0.38,
        "base_margin": 0.14,
        "relax_min": 0.80,
        "relax_max": 1.00,
    },
    "weak_spatial": {
        "predicates": {"near", "behind", "in front of", "under", "above",
                       "next to", "over", "inside", "attached to", "covering"},
        "base_conf": 0.40,
        "base_margin": 0.18,
        "relax_min": 1.00,
        "relax_max": 1.00,
    },
}

_PREDICATE_TO_FAMILY: Dict[str, str] = {}
for _family_name, _config in PREDICATE_FAMILIES.items():
    for _p in _config["predicates"]:
        _PREDICATE_TO_FAMILY[_p] = _family_name


def get_predicate_family(pred: str) -> str:
    return _PREDICATE_TO_FAMILY.get(pred, "unknown")


# Confidence thresholds (mirrors predict.py — legacy defaults)
MIN_RELATION_CONFIDENCE = 0.38
MIN_RELATION_MARGIN = 0.12
WEAK_PREDICATE_EXTRA_MARGIN = 0.08
STRONG_PREDICATE_MIN_CONF = {"wearing": 0.26, "sitting on": 0.28, "riding": 0.27, "holding": 0.25}
STRONG_PREDICATE_MIN_MARGIN = {"wearing": 0.05, "sitting on": 0.06, "riding": 0.06, "holding": 0.05}


def classify_predicate(pred: str) -> str:
    if pred in SEMANTIC_PREDS: return "semantic"
    if pred in WEAK_SPATIAL: return "weak_spatial"
    if pred in NEUTRAL_SPATIAL: return "neutral_spatial"
    return "other"


def get_predicate_type(pred: str) -> str:
    """Categorize predicate by dominant evidence type."""
    geo_heavy = {"riding", "sitting on", "standing on", "on", "under", "above", "behind", "in front of", "near"}
    appearance_heavy = {"wearing"}
    pose_heavy = {"carrying", "holding"}
    if pred in geo_heavy: return "geometry-heavy"
    if pred in appearance_heavy: return "appearance-heavy"
    if pred in pose_heavy: return "pose-heavy"
    return "other"


class AnalysisCollector:
    """Collects detailed per-pair analysis across all images and versions."""

    def __init__(self):
        self.per_version = {}

    def run_evaluation(self, image_dir: str = "test_images", output_dir: str = "analysis_output"):
        os.makedirs(output_dir, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = sorted([
            str(p) for p in Path(image_dir).iterdir()
            if p.suffix.lower() in image_exts
        ])
        print(f"Found {len(image_paths)} images in {image_dir}")

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
                "detections": [{"label": d["label"], "box": d["box"], "score": d.get("score", 0)} for d in detections],
                "num_raw": len(raw_detections),
                "num_verified": len(detections),
            }

        # Run each version with DETAILED capture
        for version_name, ckpt_dir in sorted(CHECKPOINT_DIRS.items()):
            print(f"\n{'='*60}")
            print(f"  ANALYZING: {version_name} ({ckpt_dir})")
            print(f"{'='*60}")

            version_data = {
                "accepted": [],      # relations that passed all filters
                "rejected_calib": [], # relations rejected by calibration
                "rejected_plaus": [], # relations rejected by plausibility
                "rejected_nonsense": [], # relations rejected as nonsense
                "rejected_prior": [], # relations rejected by prior/threshold
                "per_predicate": defaultdict(list),
                "per_image": {},
                "feature_norms": [],
            }

            for img_path in image_paths:
                img_name = Path(img_path).name
                result = self._run_detailed(ckpt_dir, img_path, image_metadata[img_path])
                version_data["per_image"][img_name] = result

                # Collect accepted
                for r in result.get("relations", []):
                    version_data["accepted"].append({
                        "image": img_name,
                        "subject": r["subject"],
                        "predicate": r["predicate"],
                        "object": r["object"],
                        "confidence": r.get("confidence", 0),
                        "adjusted_confidence": r.get("adjusted_confidence", r.get("confidence", 0)),
                        "margin": r.get("margin", 0),
                        "top1": r.get("top1", r["predicate"]),
                        "top2": r.get("top2", ""),
                    })

                # Collect rejected with detailed status
                for rp in result.get("raw_predictions", []):
                    status = rp.get("status", "")
                    entry = {
                        "image": img_name,
                        "subject": rp.get("subject", "?"),
                        "object": rp.get("object", "?"),
                        "best_predicate": rp.get("best_predicate", ""),
                        "status": status,
                        "reason": rp.get("reject_reason", rp.get("reason", "")),
                        "top1": rp.get("top1", ""),
                        "top1_score": rp.get("top1_score", 0),
                        "top2": rp.get("top2", ""),
                        "top2_score": rp.get("top2_score", 0),
                        "margin": rp.get("margin", 0),
                    }
                    if "calibration" in status:
                        version_data["rejected_calib"].append(entry)
                    elif "plausibility" in status:
                        version_data["rejected_plaus"].append(entry)
                    elif "nonsense" in status:
                        version_data["rejected_nonsense"].append(entry)
                    else:
                        version_data["rejected_prior"].append(entry)

                # Collect feature norms
                norms = result.get("feature_norms", {})
                if norms:
                    version_data["feature_norms"].append(norms)

                # Collect per-predicate info from all pairs
                for rp in result.get("raw_predictions", []):
                    pred = rp.get("best_predicate", "")
                    if pred:
                        vp = version_data["per_predicate"][pred]
                        vp.append({
                            "image": img_name,
                            "subject": rp.get("subject", "?"),
                            "object": rp.get("object", "?"),
                            "top1": rp.get("top1", ""),
                            "top1_score": rp.get("top1_score", 0),
                            "top2": rp.get("top2", ""),
                            "top2_score": rp.get("top2_score", 0),
                            "margin": rp.get("margin", 0),
                            "status": rp.get("status", ""),
                            "rejected": "rejected" in rp.get("status", ""),
                        })

            self.per_version[version_name] = version_data

        # Generate all analyses
        report = self._generate_report(image_metadata, output_dir)
        return report

    def _run_detailed(self, checkpoint_dir: str, image_path: str,
                      meta: dict) -> Dict:
        """Run a single image with detailed capture, returning all metadata."""
        self._reset_relation_model()
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

        detections = meta["detections"]

        result = {
            "image_path": image_path,
            "image_size": (img_w, img_h),
            "detections": detections,
            "relations": [],
            "raw_predictions": [],
            "predicate_counts": {},
            "quality": {},
            "feature_norms": {},
        }

        if len(detections) >= 2:
            relations, raw_predictions = infer_relationships_semantic(
                detections, threshold=0.05, top_k=5,
                image=image, temperature=2.0, debug=False,
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

        return result

    def _reset_relation_model(self):
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
        rp._model_type = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_report(self, image_metadata: dict, output_dir: str) -> dict:
        report = {
            "config": {"image_dir": "test_images", "checkpoints": dict(CHECKPOINT_DIRS)},
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "image_metadata": {k: {"detections": [d["label"] for d in v["detections"]]}
                               for k, v in image_metadata.items()},
            "predicate_analysis": {},
            "confidence_histograms": {},
            "failure_analysis": {},
            "feature_utilization": {},
            "calibration_analysis": {},
            "qualitative_comparison": [],
        }

        vnames = sorted(self.per_version.keys())

        for vname in vnames:
            vd = self.per_version[vname]
            self._analyze_predicates(vname, vd, report)
            self._analyze_confidence_histograms(vname, vd, report)
            self._analyze_failures(vname, vd, report)
            self._analyze_features(vname, vd, report)
            self._analyze_calibration(vname, vd, report)

        self._build_qualitative_comparison(vnames, report, image_metadata)

        # Save report
        report_path = os.path.join(output_dir, "analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved analysis report: {report_path}")

        # Generate markdown
        self._save_markdown(report, output_dir, vnames)
        return report

    def _analyze_predicates(self, vname: str, vd: dict, report: dict):
        """Per-predicate frequency, confidence, rejection analysis."""
        pa = {}
        all_predicates = set()

        # Collect per-predicate stats
        pred_stats = defaultdict(lambda: {
            "accepted": 0, "rejected": 0, "confidences": [], "margins": [],
            "pairs_evaluated": 0,
        })

        for r in vd["accepted"]:
            p = r["predicate"]
            pred_stats[p]["accepted"] += 1
            pred_stats[p]["confidences"].append(r["adjusted_confidence"])
            pred_stats[p]["margins"].append(r.get("margin", 0))
            all_predicates.add(p)

        for entry in vd["rejected_calib"] + vd["rejected_plaus"] + vd["rejected_nonsense"] + vd["rejected_prior"]:
            p = entry.get("best_predicate", "")
            if p:
                pred_stats[p]["rejected"] += 1
                pred_stats[p]["confidences"].append(entry.get("top1_score", 0))
                all_predicates.add(p)

        for pred in sorted(all_predicates):
            s = pred_stats[pred]
            total = s["accepted"] + s["rejected"]
            confs = s["confidences"]
            margins = s["margins"]
            pa[pred] = {
                "type": classify_predicate(pred),
                "evidence_type": get_predicate_type(pred),
                "accepted": s["accepted"],
                "rejected": s["rejected"],
                "total_pairs": total,
                "accept_rate": round(s["accepted"] / max(total, 1), 4),
                "rejection_rate": round(s["rejected"] / max(total, 1), 4),
                "confidence_mean": round(sum(confs) / max(len(confs), 1), 4) if confs else 0,
                "confidence_min": round(min(confs), 4) if confs else 0,
                "confidence_max": round(max(confs), 4) if confs else 0,
                "margin_mean": round(sum(margins) / max(len(margins), 1), 4) if margins else 0,
            }

        total_accepted = len(vd["accepted"])
        total_rejected = len(vd["rejected_calib"]) + len(vd["rejected_plaus"]) + \
                         len(vd["rejected_nonsense"]) + len(vd["rejected_prior"])
        total_pairs = total_accepted + total_rejected

        report["predicate_analysis"][vname] = {
            "per_predicate": pa,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,
            "total_pairs_evaluated": total_pairs,
            "accept_rate": round(total_accepted / max(total_pairs, 1), 4),
            "semantic_count": sum(1 for r in vd["accepted"] if r["predicate"] in SEMANTIC_PREDS),
        }

    def _analyze_confidence_histograms(self, vname: str, vd: dict, report: dict):
        """Build confidence histograms for accepted and rejected pairs."""
        accepted_confs = [r["adjusted_confidence"] for r in vd["accepted"]]
        rejected_conf_calib = [e.get("top1_score", 0) for e in vd["rejected_calib"]]
        rejected_conf_all = (
            [e.get("top1_score", 0) for e in vd["rejected_calib"]] +
            [e.get("top1_score", 0) for e in vd["rejected_plaus"]] +
            [e.get("top1_score", 0) for e in vd["rejected_nonsense"]]
        )

        accepted_margins = [r.get("margin", 0) for r in vd["accepted"] if r.get("margin", 0) > 0]
        rejected_margins = [e.get("margin", 0) for e in vd["rejected_calib"] if e.get("margin", 0) > 0]

        def build_histogram(values, bins=10, range_min=0, range_max=1.0):
            if not values:
                return {"bins": [], "counts": [], "mean": 0, "n": 0}
            step = (range_max - range_min) / bins
            hist = [0] * bins
            bin_labels = []
            for i in range(bins):
                lo = range_min + i * step
                hi = lo + step
                bin_labels.append(f"{lo:.2f}-{hi:.2f}")
            for v in values:
                idx = min(int((v - range_min) / step), bins - 1)
                idx = max(0, idx)
                hist[idx] += 1
            return {
                "bins": bin_labels,
                "counts": hist,
                "mean": round(sum(values) / max(len(values), 1), 4),
                "median": round(sorted(values)[len(values) // 2], 4) if values else 0,
                "n": len(values),
                "min": round(min(values), 4) if values else 0,
                "max": round(max(values), 4) if values else 0,
            }

        report["confidence_histograms"][vname] = {
            "accepted_conf": build_histogram(accepted_confs, bins=10, range_min=0, range_max=1.0),
            "rejected_calib_conf": build_histogram(rejected_conf_calib, bins=10, range_min=0, range_max=1.0),
            "rejected_all_conf": build_histogram(rejected_conf_all, bins=10, range_min=0, range_max=1.0),
            "accepted_margins": build_histogram(accepted_margins, bins=10, range_min=0, range_max=0.5),
            "rejected_margins": build_histogram(rejected_margins, bins=10, range_min=0, range_max=0.5),
        }

    def _analyze_failures(self, vname: str, vd: dict, report: dict):
        """Categorize failures into A/B/C/D."""
        failures = []
        image_failures = defaultdict(list)

        # A: Geometry underuse - semantic replaced by unrelated semantic
        for r in vd["accepted"]:
            subj = r["subject"]
            obj = r["object"]
            pred = r["predicate"]
            if subj in ANIMATE and obj in {"bicycle", "horse", "motorcycle", "skateboard"} and pred in {"wearing", "holding", "carrying", "looking at"}:
                fails = {
                    "category": "A",
                    "label": "Geometry underuse",
                    "detail": f"{subj} {pred} {obj}: expected geometry-heavy interaction (riding/sitting on)",
                    "image": r["image"],
                    "subject": subj, "predicate": pred, "object": obj,
                }
                failures.append(fails)
                image_failures[r["image"]].append(fails)

        # B: Overconservative calibration - valid relation rejected near threshold
        for entry in vd["rejected_calib"]:
            reason = entry.get("reason", "")
            score = entry.get("top1_score", 0)
            pred = entry.get("best_predicate", "")
            subj = entry.get("subject", "")
            obj = entry.get("object", "")
            # Check if conf is close to threshold
            threshold = STRONG_PREDICATE_MIN_CONF.get(pred, MIN_RELATION_CONFIDENCE)
            if score >= threshold * 0.7 and score < threshold and subj in ANIMATE:
                fails = {
                    "category": "B",
                    "label": "Overconservative calibration",
                    "detail": f"{subj}->{pred}->{obj} conf={score:.2f} < threshold={threshold:.2f} (close call)",
                    "image": entry.get("image", ""),
                    "subject": subj, "predicate": pred, "object": obj,
                    "score": score, "threshold": threshold,
                }
                failures.append(fails)
                image_failures[entry.get("image", "")].append(fails)

        # Also check margin-based rejections
        for entry in vd["rejected_calib"]:
            reason = entry.get("reason", "")
            if "margin" in reason:
                score = entry.get("top1_score", 0)
                margin = entry.get("margin", 0)
                if margin >= 0.05:  # close to threshold (0.12 default)
                    fails = {
                        "category": "B",
                        "label": "Overconservative calibration (margin)",
                        "detail": f"{entry.get('subject','?')}->{entry.get('best_predicate','?')}->{entry.get('object','?')} "
                                  f"margin={margin:.2f} < threshold (close call, conf={score:.2f})",
                        "image": entry.get("image", ""),
                        "subject": entry.get("subject", ""),
                        "predicate": entry.get("best_predicate", ""),
                        "object": entry.get("object", ""),
                        "score": score, "margin": margin,
                    }
                    failures.append(fails)
                    image_failures[entry.get("image", "")].append(fails)

        # C: Weak semantic evidence - rejected with very low confidence
        for entry in vd["rejected_calib"]:
            score = entry.get("top1_score", 0)
            pred = entry.get("best_predicate", "")
            subj = entry.get("subject", "")
            if score < 0.15 and subj in ANIMATE:
                fails = {
                    "category": "C",
                    "label": "Weak semantic evidence",
                    "detail": f"{entry.get('subject','?')}->{pred}->{entry.get('object','?')} "
                              f"conf={score:.2f} (very low)",
                    "image": entry.get("image", ""),
                    "subject": subj, "predicate": pred,
                    "object": entry.get("object", ""),
                    "score": score,
                }
                failures.append(fails)
                image_failures[entry.get("image", "")].append(fails)

        # D: Hallucination suppression success - false interactions correctly removed
        for entry in vd["rejected_calib"] + vd["rejected_plaus"] + vd["rejected_nonsense"]:
            subj = entry.get("subject", "")
            obj = entry.get("object", "")
            pred = entry.get("best_predicate", "")
            if subj not in ANIMATE and obj not in ANIMATE:
                fails = {
                    "category": "D",
                    "label": "Hallucination suppression success",
                    "detail": f"{subj}->{pred}->{obj} correctly rejected (inanimate pair)",
                    "image": entry.get("image", ""),
                    "subject": subj, "predicate": pred, "object": obj,
                }
                failures.append(fails)
                image_failures[entry.get("image", "")].append(fails)

        # Check for rejected implausible semantic pairs
        for entry in vd["rejected_plaus"] + vd["rejected_nonsense"]:
            fails = {
                "category": "D",
                "label": "Hallucination suppression (semantic filter)",
                "detail": f"{entry.get('subject','?')}->{entry.get('best_predicate','?')}->{entry.get('object','?')} "
                          f"reason: {entry.get('reason','')}",
                "image": entry.get("image", ""),
                "subject": entry.get("subject", ""),
                "predicate": entry.get("best_predicate", ""),
                "object": entry.get("object", ""),
            }
            failures.append(fails)
            image_failures[entry.get("image", "")].append(fails)

        cat_counts = Counter(f["category"] for f in failures)
        report["failure_analysis"][vname] = {
            "total_failures": len(failures),
            "category_counts": dict(cat_counts),
            "category_labels": {
                "A": "Geometry underuse (semantic won over geometry-heavy)",
                "B": "Overconservative calibration (close to threshold)",
                "C": "Weak semantic evidence (very low confidence)",
                "D": "Hallucination suppression success (correct rejections)",
            },
            "failures": failures,
            "per_image_failure_counts": {
                img: len(fs) for img, fs in image_failures.items()
            },
            "images_with_failures": list(image_failures.keys()),
        }

    def _analyze_features(self, vname: str, vd: dict, report: dict):
        """Feature group utilization analysis."""
        all_norms = vd.get("feature_norms", [])
        if not all_norms:
            report["feature_utilization"][vname] = {}
            return

        avg_norms = {}
        for key in all_norms[0]:
            vals = [n.get(key, 0) for n in all_norms]
            avg_norms[key] = round(sum(vals) / len(vals), 4)
        total = sum(avg_norms.values()) or 1.0
        pct = {k: round(v / total * 100, 1) for k, v in sorted(avg_norms.items(), key=lambda x: -x[1])}

        report["feature_utilization"][vname] = {
            "avg_norms": avg_norms,
            "percentages": pct,
        }

    def _analyze_calibration(self, vname: str, vd: dict, report: dict):
        """Calibration analysis: per-family breakdown, adaptive margin stats."""
        reasons = Counter()
        for entry in vd["rejected_calib"]:
            reason = entry.get("reason", entry.get("reject_reason", "unknown"))
            if "conf" in reason.lower():
                reasons["confidence_below_threshold"] += 1
            elif "margin" in reason.lower():
                reasons["margin_below_threshold"] += 1
            else:
                reasons[reason] += 1

        # Analyze how far below threshold (legacy metrics)
        conf_shortfalls = []
        margin_shortfalls = []
        for entry in vd["rejected_calib"]:
            pred = entry.get("best_predicate", "")
            score = entry.get("top1_score", 0)
            threshold = STRONG_PREDICATE_MIN_CONF.get(pred, MIN_RELATION_CONFIDENCE)
            shortfall = threshold - score
            if shortfall > 0:
                conf_shortfalls.append(shortfall)

            margin = entry.get("margin", 0)
            margin_thresh = STRONG_PREDICATE_MIN_MARGIN.get(pred, MIN_RELATION_MARGIN)
            if pred in WEAK_SPATIAL:
                margin_thresh += 0.08
            mshort = margin_thresh - margin
            if mshort > 0:
                margin_shortfalls.append(mshort)

        # ── New: Per-family calibration analysis ────────────────────
        family_stats: Dict[str, Dict] = {}
        for fam_name in PREDICATE_FAMILIES:
            family_stats[fam_name] = {
                "rejected": 0, "accepted": 0,
                "conf_values": [], "margin_values": [],
                "effective_margins": [], "competitor_factors": [],
            }
        family_stats["unknown"] = {"rejected": 0, "accepted": 0, "conf_values": [],
                                   "margin_values": [], "effective_margins": [],
                                   "competitor_factors": []}

        # Accepted entries
        for r in vd["accepted"]:
            pred = r["predicate"]
            fam = get_predicate_family(pred)
            if fam not in family_stats:
                fam = "unknown"
            family_stats[fam]["accepted"] += 1
            family_stats[fam]["conf_values"].append(r.get("adjusted_confidence", r.get("confidence", 0)))
            family_stats[fam]["margin_values"].append(r.get("margin", 0))

        # Rejected entries — extract calib_debug if available
        for entry in vd["rejected_calib"]:
            pred = entry.get("best_predicate", entry.get("top1", ""))
            fam = get_predicate_family(pred)
            if fam not in family_stats:
                fam = "unknown"
            family_stats[fam]["rejected"] += 1
            family_stats[fam]["conf_values"].append(entry.get("top1_score", 0))
            family_stats[fam]["margin_values"].append(entry.get("margin", 0))

            # Extract adaptive calibration params
            calib_debug = entry.get("calib_debug", {})
            if calib_debug:
                eff_margin = calib_debug.get("effective_margin")
                if eff_margin is not None:
                    family_stats[fam]["effective_margins"].append(eff_margin)
                comp_factor = calib_debug.get("competitor_factor")
                if comp_factor is not None:
                    family_stats[fam]["competitor_factors"].append(comp_factor)

        # Compute per-family summaries
        family_summaries = {}
        for fam_name, stats in sorted(family_stats.items()):
            total = stats["accepted"] + stats["rejected"]
            confs = stats["conf_values"]
            margins = stats["margin_values"]
            eff_margins = stats["effective_margins"]
            comp_factors = stats["competitor_factors"]
            family_summaries[fam_name] = {
                "accepted": stats["accepted"],
                "rejected": stats["rejected"],
                "total": total,
                "accept_rate": round(stats["accepted"] / max(total, 1), 4),
                "conf_mean": round(sum(confs) / max(len(confs), 1), 4) if confs else 0,
                "margin_mean": round(sum(margins) / max(len(margins), 1), 4) if margins else 0,
                "effective_margin_mean": round(sum(eff_margins) / max(len(eff_margins), 1), 4) if eff_margins else None,
                "competitor_factor_mean": round(sum(comp_factors) / max(len(comp_factors), 1), 4) if comp_factors else None,
            }

        # ── Adaptive margin efficiency metric ──────────────────────
        # How many more acceptances does the adaptive system generate?
        # We estimate by checking how many margin-rejected entries had
        # margins above the adaptive effective threshold.
        recovered_by_adaptive = 0
        for entry in vd["rejected_calib"]:
            pred = entry.get("top1", entry.get("best_predicate", ""))
            score = entry.get("top1_score", 0)
            margin = entry.get("margin", 0)
            reason = entry.get("reason", entry.get("reject_reason", ""))
            if "margin" not in reason.lower():
                continue
            calib_debug = entry.get("calib_debug", {})
            if calib_debug:
                eff_margin = calib_debug.get("effective_margin")
                if eff_margin is not None and margin >= eff_margin:
                    recovered_by_adaptive += 1

        report["calibration_analysis"][vname] = {
            "rejection_reasons": dict(reasons),
            "conf_shortfall_mean": round(sum(conf_shortfalls) / max(len(conf_shortfalls), 1), 4) if conf_shortfalls else 0,
            "conf_shortfall_min": round(min(conf_shortfalls), 4) if conf_shortfalls else 0,
            "conf_shortfall_max": round(max(conf_shortfalls), 4) if conf_shortfalls else 0,
            "margin_shortfall_mean": round(sum(margin_shortfalls) / max(len(margin_shortfalls), 1), 4) if margin_shortfalls else 0,
            "rejected_by_conf": reasons.get("confidence_below_threshold", 0),
            "rejected_by_margin": reasons.get("margin_below_threshold", 0),
            "per_family": family_summaries,
            "recovered_by_adaptive": recovered_by_adaptive,
        }

    def _build_qualitative_comparison(self, vnames: list, report: dict, image_metadata: dict):
        """Build per-image comparison table."""
        for img_path, meta in image_metadata.items():
            img_name = Path(img_path).name
            row = {
                "image": img_name,
                "detections": [d["label"] for d in meta["detections"]],
                "per_version": {},
            }
            for vn in vnames:
                vd = self.per_version.get(vn, {})
                per_img = vd.get("per_image", {}).get(img_name, {})
                rels = per_img.get("relations", [])
                row["per_version"][vn] = [
                    {"subject": r["subject"], "predicate": r["predicate"],
                     "object": r["object"], "confidence": r.get("confidence", 0)}
                    for r in rels
                ]
            report["qualitative_comparison"].append(row)

    def _save_markdown(self, report: dict, output_dir: str, vnames: list):
        md_path = os.path.join(output_dir, "analysis_report.md")
        lines = []
        def w(s=""): lines.append(s)

        w("# V0 vs V2 Behavioral Analysis Report")
        w(f"\nGenerated: {report['timestamp']}\n")

        # Qualitative comparison
        w("## Qualitative Comparison\n")
        w("| Image | Detections | " + " | ".join(f"{v} Relations" for v in vnames) + " |")
        w("|-------|-----------|" + "|".join("---" for _ in vnames) + "|")
        for row in report["qualitative_comparison"]:
            dets = ", ".join(row["detections"])
            rel_strs = []
            for vn in vnames:
                rels = row["per_version"].get(vn, [])
                if rels:
                    s = "; ".join(f"{r['subject']}->{r['predicate']}->{r['object']}({r['confidence']:.2f})" for r in rels)
                else:
                    s = "(none)"
                rel_strs.append(s)
            w(f"| {row['image']} | {dets} | {' | '.join(rel_strs)} |")
        w("")

        # Predicate-specific analysis per version
        for vn in vnames:
            w(f"## Predicate-Specific Analysis: {vn}\n")
            pa = report["predicate_analysis"].get(vn, {})
            per_pred = pa.get("per_predicate", {})
            w("| Predicate | Type | Evidence | Accepted | Rejected | Accept% | Conf Mean | Margin Mean |")
            w("|-----------|------|----------|----------|----------|---------|-----------|-------------|")
            for pred in sorted(per_pred.keys()):
                info = per_pred[pred]
                w(f"| {pred} | {info['type']} | {info['evidence_type']} | "
                  f"{info['accepted']} | {info['rejected']} | "
                  f"{info['accept_rate']:.2%} | {info['confidence_mean']:.4f} | "
                  f"{info['margin_mean']:.4f} |")
            w(f"\nTotal accepted: {pa['total_accepted']} / {pa['total_pairs_evaluated']} "
              f"({pa['accept_rate']:.1%})\n")

        # Confidence histograms
        w("## Confidence Histograms\n")
        for vn in vnames:
            w(f"### {vn}\n")
            ch = report["confidence_histograms"].get(vn, {})
            for name, hist in [("Accepted", ch.get("accepted_conf", {})),
                               ("Rejected (calibration)", ch.get("rejected_calib_conf", {}))]:
                if hist.get("n", 0) == 0:
                    w(f"**{name}:** no data\n")
                    continue
                w(f"**{name}** (n={hist['n']}, mean={hist['mean']:.3f}, "
                  f"median={hist.get('median',0):.3f}, range=[{hist['min']:.3f}, {hist['max']:.3f}])\n")
                w("```")
                bins = hist.get("bins", [])
                counts = hist.get("counts", [])
                max_count = max(counts) if counts else 1
                scale = 40 / max_count
                for i, (bl, bc) in enumerate(zip(bins, counts)):
                    bar = "#" * int(bc * scale)
                    w(f"  {bl}: {bc:3d} {bar}")
                w("```\n")

        # Failure analysis
        w("## Failure Analysis\n")
        for vn in vnames:
            w(f"### {vn}\n")
            fa = report["failure_analysis"].get(vn, {})
            cats = fa.get("category_counts", {})
            w(f"Total failures: {fa.get('total_failures', 0)}\n")
            w("| Category | Count | Description |")
            w("|----------|-------|-------------|")
            labels = fa.get("category_labels", {})
            for cat in ["A", "B", "C", "D"]:
                count = cats.get(cat, 0)
                label = labels.get(cat, "")
                w(f"| {cat} | {count} | {label} |")
            w("")
            for f in fa.get("failures", [])[:20]:
                w(f"- [{f['category']}] {f['detail']}")

        # Feature utilization
        w("## Feature Utilization\n")
        for vn in vnames:
            w(f"### {vn}\n")
            fu = report["feature_utilization"].get(vn, {})
            pcts = fu.get("percentages", {})
            w("| Feature | Weight% |")
            w("|---------|---------|")
            for feat, pct in pcts.items():
                w(f"| {feat} | {pct}% |")
            w("")

        # Calibration analysis
        w("## Calibration Analysis\n")
        for vn in vnames:
            w(f"### {vn}\n")
            ca = report["calibration_analysis"].get(vn, {})
            w(f"Rejected by confidence: {ca.get('rejected_by_conf', 0)}")
            w(f"Rejected by margin: {ca.get('rejected_by_margin', 0)}")
            w(f"Conf shortfall mean: {ca.get('conf_shortfall_mean', 0):.4f}")
            w(f"Margin shortfall mean: {ca.get('margin_shortfall_mean', 0):.4f}")
            w(f"Recovered by adaptive calibration: {ca.get('recovered_by_adaptive', 0)}")
            w("")

            # Per-family breakdown
            per_family = ca.get("per_family", {})
            if per_family:
                w("| Predicate Family | Accepted | Rejected | Accept% | Conf Mean | Margin Mean | Eff Margin | Comp Factor |")
                w("|-----------------|----------|----------|---------|-----------|-------------|------------|-------------|")
                for fam_name, fstats in sorted(per_family.items()):
                    em_str = f"{fstats['effective_margin_mean']:.3f}" if fstats['effective_margin_mean'] is not None else "—"
                    cf_str = f"{fstats['competitor_factor_mean']:.2f}" if fstats['competitor_factor_mean'] is not None else "—"
                    w(f"| {fam_name} | {fstats['accepted']} | {fstats['rejected']} | "
                      f"{fstats['accept_rate']:.0%} | {fstats['conf_mean']:.4f} | "
                      f"{fstats['margin_mean']:.4f} | {em_str} | {cf_str} |")
                w("")

        with open(md_path, "w") as f:
            f.writelines(lines)
        print(f"Saved markdown report: {md_path}")


def print_terminal_summary(report: dict, vnames: list):
    """Print a compact terminal summary of the analysis."""
    print(f"\n{'='*70}")
    print(f"  BEHAVIORAL ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for vn in vnames:
        print(f"\n  --- {vn} ---")
        pa = report["predicate_analysis"].get(vn, {})
        print(f"  Accepted: {pa.get('total_accepted', 0)} / {pa.get('total_pairs_evaluated', 0)} "
              f"({pa.get('accept_rate', 0):.1%})")
        print(f"  Semantic: {pa.get('semantic_count', 0)}")

        per_pred = pa.get("per_predicate", {})
        print(f"\n  Per-predicate:")
        print(f"  {'Predicate':<18} {'Accepted':>8} {'Rejected':>8} {'Accept%':>8} {'Conf Mean':>10}")
        print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for pred in sorted(per_pred.keys()):
            info = per_pred[pred]
            print(f"  {pred:<18} {info['accepted']:>8} {info['rejected']:>8} "
                  f"{info['accept_rate']:.0%} {info['confidence_mean']:>10.4f}")

        fa = report["failure_analysis"].get(vn, {})
        cats = fa.get("category_counts", {})
        print(f"\n  Failures by category:")
        for cat in ["A", "B", "C", "D"]:
            count = cats.get(cat, 0)
            label = {"A": "Geometry underuse", "B": "Overconservative calib",
                     "C": "Weak evidence", "D": "Hallucination suppression"}.get(cat, "")
            print(f"    {cat}: {count} ({label})")

        fu = report["feature_utilization"].get(vn, {})
        pcts = fu.get("percentages", {})
        print(f"\n  Feature utilization:")
        for feat, pct in pcts.items():
            print(f"    {feat:<18}: {pct:5.1f}%")

        ch = report["confidence_histograms"].get(vn, {})
        for name, key in [("Accepted", "accepted_conf"), ("Rejected", "rejected_calib_conf")]:
            h = ch.get(key, {})
            if h.get("n", 0) > 0:
                print(f"\n  {name} conf: n={h['n']}, mean={h['mean']:.3f}, "
                      f"median={h.get('median',0):.3f}, range=[{h['min']:.3f}, {h['max']:.3f}]")

        ca = report["calibration_analysis"].get(vn, {})
        per_family = ca.get("per_family", {})
        if per_family:
            print(f"\n  Per-family calibration:")
            print(f"  {'Family':<20} {'Accepted':>8} {'Rejected':>8} {'Accept%':>8} {'Eff Margin':>10}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            for fam_name, fstats in sorted(per_family.items()):
                em = fstats.get('effective_margin_mean')
                em_str = f"{em:.3f}" if em is not None else "—"
                print(f"  {fam_name:<20} {fstats['accepted']:>8} {fstats['rejected']:>8} "
                      f"{fstats['accept_rate']:.0%} {em_str:>10}")
        recovered = ca.get("recovered_by_adaptive", 0)
        if recovered > 0:
            print(f"  Recovered by adaptive calibration: {recovered}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V0 vs V2 behavioral analysis")
    parser.add_argument("--image-dir", default="test_images")
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-evaluation, use existing results")
    args = parser.parse_args()

    collector = AnalysisCollector()

    if args.skip_eval and os.path.exists(os.path.join(args.output_dir, "analysis_report.json")):
        print("Loading existing analysis...")
        with open(os.path.join(args.output_dir, "analysis_report.json")) as f:
            report = json.load(f)
    else:
        report = collector.run_evaluation(args.image_dir, args.output_dir)

    vnames = sorted(CHECKPOINT_DIRS.keys())
    print_terminal_summary(report, vnames)
    print(f"\nFull report saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
