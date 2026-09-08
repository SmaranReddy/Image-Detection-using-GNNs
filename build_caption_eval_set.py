"""
build_caption_eval_set.py — independent ground truth for the caption experiment.

Why this exists
---------------
`hallucination_eval.py` scores CHAIR and POPE against `gt_objects`, and those
ground-truth objects come from YOLO itself (`yolo_detections_to_objects`). The
grounded system is CONDITIONED on the same YOLO detections and GATED against
the same labels, so it is being graded against its own input: it can only win.
Any hallucination reduction measured that way is circular and is not evidence
of anything.

This script produces an independent object ground truth from the Visual Genome
human annotations that the relation model is already trained and evaluated on:

    image_id -> {COCO-80 object labels a human annotated in that image}

using the repository's own `normalize_label` (the same COCO-80 space and the
same synonym map the relation model uses). Nothing is downloaded and no model
is run — it reads the `relationships.json` already required by E0.

It restricts the image set to the frozen E0 TEST split, so the caption
experiment runs on images the relation model has never been trained on, under
the same image-disjoint protocol as the relation numbers.

Usage:
    python build_caption_eval_set.py
    python build_caption_eval_set.py --split test --limit 250

Output (JSON):
    {
      "meta": {...},
      "images": {"<image_id>": {"objects": [...], "file": "<path or null>"}, ...}
    }

Feed it to the caption evaluation with:
    python hallucination_eval.py --gt-objects-json splits/e0_caption_gt_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from relation_prediction.vg_dataset import (  # noqa: E402
    COCO_LABELS,
    normalize_label,
    stream_relationship_records,
    _get_name,
)

DEFAULT_VG_ROOT = "./data/visual_genome"
DEFAULT_MANIFEST = "./splits/e0_image_split.json"
DEFAULT_OUTPUT = "./splits/e0_caption_gt_test.json"


def resolve_image_path(image_dir: Optional[str], image_id: int) -> Optional[str]:
    """Locate a VG image, honouring the VG_100K / VG_100K_2 layout."""
    if not image_dir:
        return None
    for subdir in ("", "VG_100K", "VG_100K_2"):
        for ext in (".jpg", ".png", ".jpeg"):
            path = (os.path.join(image_dir, subdir, f"{image_id}{ext}")
                    if subdir else os.path.join(image_dir, f"{image_id}{ext}"))
            if os.path.isfile(path):
                return path.replace("\\", "/")
    return None


def collect_objects(vg_root: str, wanted: Set[int]) -> Dict[int, Set[str]]:
    """Human-annotated COCO-80 objects per image, from VG relationship entities.

    Every subject and object of every annotated relationship is a human-drawn,
    human-named region. Mapping those names through `normalize_label` gives the
    COCO-80 objects a person confirmed are present — independent of YOLO.
    """
    rel_json = os.path.join(vg_root, "relationships.json")
    if not os.path.isfile(rel_json):
        raise FileNotFoundError(f"Required VG annotation file missing: {rel_json}")

    objects: Dict[int, Set[str]] = {}
    for record in stream_relationship_records(rel_json):
        iid = record.get("image_id")
        if iid is None:
            continue
        iid = int(iid)
        if iid not in wanted:
            continue
        found = objects.setdefault(iid, set())
        for rel in record.get("relationships", []):
            for side in ("subject", "object"):
                name = normalize_label(_get_name(rel.get(side, {})))
                if name in COCO_LABELS:
                    found.add(name)
    return objects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independent VG object ground truth for the caption "
                    "experiment (non-circular CHAIR / POPE).",
    )
    parser.add_argument("--vg-root", default=DEFAULT_VG_ROOT)
    parser.add_argument("--split-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which frozen split to draw images from "
                             "(default: test — the relation model never saw it)")
    parser.add_argument("--vg-image-dir", default=None,
                        help="VG image directory (default <vg-root>/images). "
                             "Images that are not present are still listed, "
                             "with file=null.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep at most N images (sampled deterministically "
                             "with --seed, after sorting by image_id)")
    parser.add_argument("--min-objects", type=int, default=2,
                        help="Drop images with fewer than N ground-truth "
                             "objects: CHAIR is uninformative on near-empty "
                             "annotations (default: 2)")
    parser.add_argument("--require-image", action="store_true",
                        help="Keep only images whose file is present on disk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists")
    args = parser.parse_args()

    if os.path.isfile(args.output) and not args.force:
        raise SystemExit(
            f"[caption-gt] Refusing to overwrite existing {args.output}. "
            "Pass --force or choose another --output."
        )

    with open(args.split_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    split_ids = {int(i) for i in manifest[f"{args.split}_ids"]}
    print(f"[caption-gt] frozen manifest : {args.split_manifest}")
    print(f"[caption-gt] split           : {args.split} ({len(split_ids):,} images)")

    image_dir = args.vg_image_dir or os.path.join(args.vg_root, "images")
    print(f"[caption-gt] scanning relationships.json …")
    objects = collect_objects(args.vg_root, split_ids)
    print(f"[caption-gt] images with annotations: {len(objects):,}")

    entries: Dict[str, Dict] = {}
    n_dropped_small = 0
    n_dropped_missing = 0
    for iid in sorted(objects):
        labels = sorted(objects[iid])
        if len(labels) < args.min_objects:
            n_dropped_small += 1
            continue
        path = resolve_image_path(image_dir, iid)
        if args.require_image and path is None:
            n_dropped_missing += 1
            continue
        entries[str(iid)] = {"objects": labels, "file": path}

    if args.limit is not None and len(entries) > args.limit:
        keys = sorted(entries, key=lambda k: int(k))
        rng = random.Random(args.seed)
        rng.shuffle(keys)
        keep = set(keys[:args.limit])
        entries = {k: v for k, v in entries.items() if k in keep}

    label_hist = Counter()
    for e in entries.values():
        label_hist.update(e["objects"])
    n_with_file = sum(1 for e in entries.values() if e["file"])

    report = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Visual Genome relationship entity names (human annotations)",
            "label_space": "COCO-80 via relation_prediction.vg_dataset.normalize_label",
            "split_manifest": os.path.abspath(args.split_manifest).replace("\\", "/"),
            "split": args.split,
            "seed": args.seed,
            "min_objects": args.min_objects,
            "limit": args.limit,
            "n_images": len(entries),
            "n_images_with_file_on_disk": n_with_file,
            "n_dropped_below_min_objects": n_dropped_small,
            "n_dropped_missing_image": n_dropped_missing,
            "mean_objects_per_image": round(
                sum(len(e["objects"]) for e in entries.values()) / max(len(entries), 1), 3),
            "label_histogram": dict(label_hist.most_common()),
            "why": (
                "Independent of YOLO. Scoring CHAIR/POPE against YOLO's own "
                "detections is circular because the grounded system is "
                "conditioned and gated on exactly those detections."
            ),
        },
        "images": entries,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[caption-gt] kept images     : {len(entries):,} "
          f"({n_with_file:,} with the image file present)")
    print(f"[caption-gt] mean objects    : {report['meta']['mean_objects_per_image']}")
    print(f"[caption-gt] dropped (<{args.min_objects} objects): {n_dropped_small:,}")
    print(f"[caption-gt] wrote {args.output}")


if __name__ == "__main__":
    main()
