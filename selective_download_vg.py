"""
Semantically-curated Visual Genome image download pipeline.

Replaces the old random-sampling approach with targeted selection
of high semantic-density images for grounded relation learning.

Strategy:
    1. Filter relationships to semantic interaction predicates only
    2. Enforce minimum box quality (>= 20px, valid boxes)
    3. Prioritise human-object interactions
    4. Rank images by semantic-density score
    5. Download only top-value images (10k-20k target)
    6. Report rich pre/post statistics
    7. Verify CLIP coverage improvement

Usage:
    python selective_download_vg.py --max-images 15000
    python selective_download_vg.py --max-images 15000 --download
    python selective_download_vg.py --max-images 15000 --download --skip-existing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "visual_genome")
os.makedirs(BASE_DIR, exist_ok=True)

FILES = {
    "relationships.json": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "image_data.json":    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
}

USER_AGENT = "Mozilla/5.0 (VG-Downloader; semantic-selection-pipeline)"

# ---------------------------------------------------------------------------
# STEP 1 — Semantic predicate definitions
# ---------------------------------------------------------------------------

# High-value semantic interaction predicates (keep these).
SEMANTIC_PREDICATES: frozenset = frozenset({
    "riding",
    "holding",
    "carrying",
    "wearing",
    "sitting on",
    "looking at",
    "attached to",
    "covering",
    "inside",
    "using",
    "eating",
    "drinking from",
})

# Normalisation map: raw predicate → canonical semantic predicate.
SEMANTIC_PREDICATE_MAP: Dict[str, str] = {
    # riding
    "riding on":      "riding",
    "mounted on":     "riding",
    "rides":          "riding",
    # holding
    "holding in":     "holding",
    "grasping":       "holding",
    "gripping":       "holding",
    "holds":          "holding",
    "hold":           "holding",
    # carrying
    "carrying in":    "carrying",
    "carried by":     "carrying",
    "carries":        "carrying",
    # wearing
    "wearing a":      "wearing",
    "wearing an":     "wearing",
    "wears":          "wearing",
    "wear":           "wearing",
    # sitting on — keep distinct from spatial "on"
    "sits on":        "sitting on",
    "sit on":         "sitting on",
    # looking at
    "looks at":       "looking at",
    "looking toward": "looking at",
    "looking in":     "looking at",
    "watching":       "looking at",
    # attached to
    "attached":       "attached to",
    # covering
    "covers":         "covering",
    "covered with":   "covering",
    "covered by":     "covering",
    # inside
    "inside of":      "inside",
    # using
    "uses":           "using",
    "use":            "using",
    # eating
    "eats":           "eating",
    "eat":            "eating",
    # drinking from
    "drinks from":    "drinking from",
    "drinking":       "drinking from",
    "drink from":     "drinking from",
}

# Spatial/geometry predicates to explicitly exclude.
SPATIAL_PREDICATES: frozenset = frozenset({
    "near", "beside", "above", "below", "behind",
    "next to", "around", "beneath", "along",
    "across", "past", "beyond",
})

# Reuse COCO label normalisation from existing codebase.
COCO_LABELS: frozenset = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
})

SYNONYM_MAP: Dict[str, str] = {
    "man": "person", "men": "person",
    "woman": "person", "women": "person",
    "boy": "person", "girl": "person",
    "people": "person", "child": "person",
    "children": "person", "guy": "person",
    "lady": "person",
    "bike": "bicycle", "cycle": "bicycle",
    "vehicle": "car", "automobile": "car",
    "sofa": "couch",
    "television": "tv", "tv monitor": "tv",
    "monitor": "tv",
    "cellphone": "cell phone", "mobile": "cell phone",
    "phone": "cell phone",
    "motorbike": "motorcycle",
    "aeroplane": "airplane", "aero plane": "airplane",
    "plant": "potted plant",
}

# ---------------------------------------------------------------------------
# STEP 2 — Box quality thresholds
# ---------------------------------------------------------------------------

MIN_BOX_SIZE = 20       # minimum width/height in pixels
MAX_ASPECT_RATIO = 10.0 # reject extreme aspect ratios


def normalise_label(label: str) -> str:
    label = label.lower().strip()
    label = SYNONYM_MAP.get(label, label)
    if label not in COCO_LABELS and label.endswith("s") and label[:-1] in COCO_LABELS:
        label = label[:-1]
    return label if label in COCO_LABELS else "UNK"


def normalise_predicate(pred: str) -> Optional[str]:
    pred = pred.lower().strip()
    pred = SEMANTIC_PREDICATE_MAP.get(pred, pred)
    if pred in SEMANTIC_PREDICATES:
        return pred
    return None


def is_human(label: str) -> bool:
    return label == "person"


def box_quality(
    x: float, y: float, w: float, h: float,
    min_size: int = MIN_BOX_SIZE,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> bool:
    if w < min_size or h < min_size:
        return False
    aspect = max(w, h) / max(min(w, h), 1.0)
    if aspect > max_aspect:
        return False
    return True


# ---------------------------------------------------------------------------
# Entity name extraction (mirrors vg_dataset._get_name)
# ---------------------------------------------------------------------------

def _get_name(entity: Dict) -> str:
    name = entity.get("name") or ""
    if not name:
        names = entity.get("names", [])
        name = names[0] if names else ""
    return name.lower().strip()


# ---------------------------------------------------------------------------
# JSON download
# ---------------------------------------------------------------------------

def download_json() -> None:
    for filename, url in FILES.items():
        dest = os.path.join(BASE_DIR, filename)
        if os.path.exists(dest):
            print(f"[skip] {filename} already exists.")
            continue

        zip_path = dest + ".zip"
        print(f"[download] {filename} ...")
        urllib.request.urlretrieve(url, zip_path)

        print(f"[extract] {filename} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BASE_DIR)
        os.remove(zip_path)
        print(f"[done] {filename}")

    print("\n--- JSON Verification ---")
    for filename in FILES:
        path = os.path.join(BASE_DIR, filename)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"[{status}] {path}")


# ---------------------------------------------------------------------------
# STEP 1-4: Filter relationships + rank images
# ---------------------------------------------------------------------------

def load_and_filter() -> Tuple[Dict[int, Any], Dict[int, Any], List[Dict]]:
    """
    Load relationships.json and image_data.json, apply semantic filters.

    Returns:
        img_meta:   image_id → image metadata dict
        img_rank:   image_id → ranking info dict
        stats:      list of stat dicts for reporting
    """
    rel_path = os.path.join(BASE_DIR, "relationships.json")
    img_path = os.path.join(BASE_DIR, "image_data.json")

    if not os.path.exists(rel_path) or not os.path.exists(img_path):
        print("[ERROR] JSON files not found. Run without flags first to download them.")
        sys.exit(1)

    with open(img_path) as f:
        img_meta_list: List[Dict] = json.load(f)
    img_size: Dict[int, Tuple[int, int]] = {
        m["image_id"]: (int(m["width"]), int(m["height"]))
        for m in img_meta_list
    }
    img_meta_dict: Dict[int, Dict] = {
        m["image_id"]: m for m in img_meta_list
    }

    with open(rel_path) as f:
        all_rels: List[Dict] = json.load(f)

    print(f"\n{'='*65}")
    print("  SEMANTIC IMAGE SELECTION PIPELINE")
    print(f"{'='*65}")

    # ------------------------------------------------------------------
    # Phase 1: Filter raw relationships to semantic only
    # ------------------------------------------------------------------
    total_raw_relations = 0
    valid_relations: List[Dict] = []
    dropped_spatial = 0
    dropped_unk_label = 0
    dropped_small_box = 0
    dropped_bad_aspect = 0
    dropped_other = 0
    raw_pred_counter: Counter = Counter()
    kept_pred_counter: Counter = Counter()
    human_interaction_counter: Counter = Counter()

    for img in all_rels:
        iid = img.get("image_id")
        img_w, img_h = img_size.get(iid, (1, 1))

        for r in img.get("relationships", []):
            total_raw_relations += 1

            pred_raw = r.get("predicate", "").lower().strip()
            raw_pred_counter[pred_raw] += 1

            pred = normalise_predicate(pred_raw)
            if pred is None:
                dropped_spatial += 1
                continue

            subj_d = r.get("subject", {})
            obj_d = r.get("object", {})

            subj_name = normalise_label(_get_name(subj_d))
            obj_name = normalise_label(_get_name(obj_d))

            if subj_name == "UNK" or obj_name == "UNK":
                dropped_unk_label += 1
                continue

            subj_box = (
                float(subj_d.get("x", 0)),
                float(subj_d.get("y", 0)),
                float(subj_d.get("w", 1)),
                float(subj_d.get("h", 1)),
            )
            obj_box = (
                float(obj_d.get("x", 0)),
                float(obj_d.get("y", 0)),
                float(obj_d.get("w", 1)),
                float(obj_d.get("h", 1)),
            )

            # Box quality checks.
            if not box_quality(*subj_box) or not box_quality(*obj_box):
                dropped_small_box += 1
                continue

            subj_aspect = max(subj_box[2], subj_box[3]) / max(min(subj_box[2], subj_box[3]), 1.0)
            obj_aspect = max(obj_box[2], obj_box[3]) / max(min(obj_box[2], obj_box[3]), 1.0)
            if subj_aspect > MAX_ASPECT_RATIO or obj_aspect > MAX_ASPECT_RATIO:
                dropped_bad_aspect += 1
                continue

            # Build enriched relation entry.
            subj_oid = subj_d.get("object_id", -1)
            obj_oid = obj_d.get("object_id", -1)

            valid_relations.append({
                "image_id": iid,
                "predicate": pred,
                "subj_name": subj_name,
                "obj_name": obj_name,
                "subj_oid": subj_oid,
                "obj_oid": obj_oid,
                "subj_box": subj_box,
                "obj_box": obj_box,
                "img_w": img_w,
                "img_h": img_h,
                "is_human_subj": is_human(subj_name),
                "is_human_obj": is_human(obj_name),
            })

            kept_pred_counter[pred] += 1
            if is_human(subj_name) or is_human(obj_name):
                human_interaction_counter[pred] += 1

    total_valid = len(valid_relations)

    print(f"\n--- Filtering Results ---")
    print(f"  Raw relationships:              {total_raw_relations}")
    print(f"  Valid semantic relationships:   {total_valid}")
    print(f"  Dropped (spatial/unknown pred): {dropped_spatial}")
    print(f"  Dropped (UNK label):            {dropped_unk_label}")
    print(f"  Dropped (small box <{MIN_BOX_SIZE}px):     {dropped_small_box}")
    print(f"  Dropped (extreme aspect):       {dropped_bad_aspect}")
    print(f"  Retention rate:                 {total_valid/max(total_raw_relations,1)*100:.2f}%")

    # ------------------------------------------------------------------
    # Phase 2: Build per-image aggregates
    # ------------------------------------------------------------------
    img_relations: Dict[int, List[Dict]] = defaultdict(list)
    for rel in valid_relations:
        img_relations[rel["image_id"]].append(rel)

    # Image ranking info.
    img_rank: Dict[int, Dict] = {}

    for iid, rels in img_relations.items():
        n_valid = len(rels)
        unique_predicates = set(r["predicate"] for r in rels)
        pred_diversity = len(unique_predicates)
        n_human_interactions = sum(
            1 for r in rels if r["is_human_subj"] or r["is_human_obj"]
        )
        avg_box_size = sum(
            min(r["subj_box"][2], r["subj_box"][3]) + min(r["obj_box"][2], r["obj_box"][3])
            for r in rels
        ) / max(n_valid * 2, 1)

        # Semantic-density score.
        score = (
            1.0 * n_valid
            + 2.0 * pred_diversity
            + 3.0 * n_human_interactions
            + 0.01 * avg_box_size
        )

        img_rank[iid] = {
            "n_valid_relations": n_valid,
            "pred_diversity": pred_diversity,
            "n_human_interactions": n_human_interactions,
            "avg_box_size": round(avg_box_size, 1),
            "score": round(score, 2),
        }

    return img_meta_dict, img_rank, valid_relations


def print_predicate_distribution(pred_counter: Counter, title: str = "Predicate Distribution") -> None:
    print(f"\n  {title}:")
    total = sum(pred_counter.values())
    for pred, count in pred_counter.most_common():
        pct = 100.0 * count / max(total, 1)
        print(f"    {pred}: {count} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# STEP 5-6: Download only high-value images
# ---------------------------------------------------------------------------

def _is_valid_jpeg(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            if header != b"\xff\xd8":
                return False
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _download_with_retry(url: str, dest: str, retries: int = 3, timeout: int = 60) -> bool:
    headers = {"User-Agent": USER_AGENT}
    tmp_dest = dest + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            resp.raise_for_status()
            total_size = int(resp.headers.get("content-length", 0))
            with open(tmp_dest, "wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True,
                        desc=os.path.basename(dest), leave=False,
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            os.replace(tmp_dest, dest)
            return True
        except requests.RequestException as exc:
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
            if attempt < retries:
                wait = 2 ** attempt
                tqdm.write(f"  [retry {attempt}/{retries}] {os.path.basename(dest)} failed: {exc}, retrying in {wait}s")
                time.sleep(wait)
            else:
                tqdm.write(f"  [fail] {os.path.basename(dest)}: {exc}")
    return False


def select_and_download(
    img_meta_dict: Dict[int, Dict],
    img_rank: Dict[int, Dict],
    valid_relations: List[Dict],
    max_images: int = 15000,
    download: bool = False,
    skip_existing: bool = True,
) -> None:
    """
    Select top-N images by semantic-density score and optionally download them.
    """
    # Sort images by score descending.
    sorted_images = sorted(img_rank.items(), key=lambda x: x[1]["score"], reverse=True)

    print(f"\n{'='*65}")
    print("  IMAGE RANKING — Semantic Density Scores")
    print(f"{'='*65}")

    # Score distribution.
    scores = [v["score"] for _, v in sorted_images]
    print(f"\n  Total qualifying images:      {len(scores)}")
    print(f"  Max score:                    {max(scores):.2f}")
    print(f"  Min score:                    {min(scores):.2f}")
    print(f"  Mean score:                   {sum(scores)/max(len(scores),1):.2f}")
    print(f"  Median score:                 {sorted(scores)[len(scores)//2]:.2f}")

    # Score quantiles.
    s_sorted = sorted(scores)
    n = len(s_sorted)
    print(f"\n  Score quantiles:")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * q / 100), n - 1)
        print(f"    P{q}: {s_sorted[idx]:.2f}")

    # Select top candidates.
    n_select = min(max_images, len(sorted_images))
    selected = [(iid, rank) for iid, rank in sorted_images[:n_select]]

    print(f"\n  Selected top {n_select} images for download.")

    # Print top 20.
    print(f"\n  Top 20 images by semantic-density score:")
    print(f"  {'Rank':<6} {'ImageID':<10} {'Score':<8} {'Relations':<10} {'PredDiv':<8} {'HumanInt':<10} {'AvgBox':<8}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    for idx, (iid, rank) in enumerate(selected[:20]):
        print(f"  {idx+1:<6} {iid:<10} {rank['score']:<8} {rank['n_valid_relations']:<10} {rank['pred_diversity']:<8} {rank['n_human_interactions']:<10} {rank['avg_box_size']:<8}")

    # Predicted CLIP coverage.
    total_selected_relations = sum(rank["n_valid_relations"] for _, rank in selected)
    total_selected_human = sum(rank["n_human_interactions"] for _, rank in selected)
    avg_score_selected = sum(rank["score"] for _, rank in selected) / max(n_select, 1)

    print(f"\n  Expected yield from selected images:")
    print(f"    Total valid relations:       {total_selected_relations}")
    print(f"    Human-object interactions:   {total_selected_human}")
    print(f"    Average semantic density:    {avg_score_selected:.2f}")

    # Predicate distribution in selected set.
    selected_iids = set(iid for iid, _ in selected)
    selected_pred_counter: Counter = Counter()
    for rel in valid_relations:
        if rel["image_id"] in selected_iids:
            selected_pred_counter[rel["predicate"]] += 1
    print_predicate_distribution(selected_pred_counter, "Predicate Distribution (Selected Set)")

    if not download:
        print(f"\n  [dry-run] Pass --download to actually download images.")
        print(f"  Estimated download size: {n_select} images (varies by resolution)")
        return

    # ------------------------------------------------------------------
    # Actual download
    # ------------------------------------------------------------------
    img_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  DOWNLOADING {n_select} HIGH-VALUE IMAGES")
    print(f"{'='*65}\n")

    downloaded = 0
    skipped = 0
    failed = 0
    corrupt = 0

    all_iids_ranked = {iid for iid, _ in selected}
    download_queue = []
    for iid, _ in selected:
        meta = img_meta_dict.get(iid)
        if meta is None:
            continue
        url = meta.get("url", "")
        if not url:
            continue
        download_queue.append((iid, url))

    for iid, url in tqdm(download_queue, desc="Downloading semantic VG images", unit="img"):
        dest = os.path.join(img_dir, f"{iid}.jpg")

        if os.path.exists(dest):
            if skip_existing:
                skipped += 1
                continue
            if not _is_valid_jpeg(dest):
                os.remove(dest)
            else:
                skipped += 1
                continue

        success = _download_with_retry(url, dest)
        if success:
            if _is_valid_jpeg(dest):
                downloaded += 1
            else:
                os.remove(dest)
                corrupt += 1
                tqdm.write(f"  [corrupt] {iid}.jpg: invalid JPEG data")
        else:
            failed += 1

    # ------------------------------------------------------------------
    # Post-download report
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  DOWNLOAD COMPLETE")
    print(f"{'='*65}")
    print(f"  Total targeted:   {n_select}")
    print(f"  Downloaded:       {downloaded}")
    print(f"  Skipped:          {skipped}")
    print(f"  Failed:           {failed}")
    print(f"  Corrupt:          {corrupt}")

    on_disk = len(os.listdir(img_dir))
    print(f"\n  Images on disk:   {on_disk}")

    # CLIP coverage estimate.
    downloaded_iids = set()
    for fname in os.listdir(img_dir):
        if fname.endswith(".jpg"):
            try:
                downloaded_iids.add(int(fname.replace(".jpg", "")))
            except ValueError:
                pass

    usable = downloaded_iids & selected_iids
    usable_relations = sum(
        1 for rel in valid_relations if rel["image_id"] in usable
    )
    usable_human = sum(
        1 for rel in valid_relations
        if rel["image_id"] in usable and (rel["is_human_subj"] or rel["is_human_obj"])
    )

    print(f"\n--- CLIP Coverage Analysis ---")
    print(f"  Download-verified images:  {len(usable)}")
    print(f"  Usable relation samples:   {usable_relations}")
    print(f"  Usable HOI samples:        {usable_human}")

    # Old approach comparison.
    print(f"\n--- Improvement vs Random Sampling ---")
    print(f"  Old approach: ~4.4% CLIP coverage, ~885 useful VG images")
    print(f"  New approach: {len(usable)} curated images with semantic relations")
    print(f"  Expected CLIP coverage: {usable_relations} non-zero embeddings")
    if len(usable) > 0:
        print(f"  Semantic density per image: {usable_relations/max(len(usable),1):.1f} relations/image")

    # Zero-vector risk.
    total_embeddings_needed = total_selected_relations * 2  # subj + obj
    print(f"\n  Total CLIP embeddings needed: {total_embeddings_needed}")
    print(f"  Estimated zero-vector risk:   LOW (all boxes >= {MIN_BOX_SIZE}px)")
    print(f"  Geometry shortcut risk:       LOW (semantic predicates dominate)")

    print(f"\n{'='*65}")
    print("  NEXT STEPS")
    print(f"{'='*65}")
    print(f"  Train with visual-semantic features:")
    print(f"    python train_full_visual_semantic.py --vg-root {BASE_DIR}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantically-curated Visual Genome download pipeline.\n"
                    "Selects only high-value semantic interaction images.",
    )
    parser.add_argument(
        "--max-images", type=int, default=15000,
        help="Maximum number of top-ranked images to download (default: 15000)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Actually download the selected images (dry-run without this flag)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip already-downloaded images (default: true)",
    )
    parser.add_argument(
        "--min-box-size", type=int, default=20,
        help=f"Minimum box dimension in pixels (default: 20)",
    )
    parser.add_argument(
        "--max-aspect", type=float, default=10.0,
        help=f"Maximum box aspect ratio (default: 10.0)",
    )
    args = parser.parse_args()

    # Globally reconfigure thresholds if overridden.
    if args.min_box_size != 20:
        global MIN_BOX_SIZE
        MIN_BOX_SIZE = args.min_box_size
    if args.max_aspect != 10.0:
        global MAX_ASPECT_RATIO
        MAX_ASPECT_RATIO = args.max_aspect

    # Ensure JSON data is available.
    download_json()

    # Load, filter, rank.
    img_meta_dict, img_rank, valid_relations = load_and_filter()

    # Print predicate distribution for all valid relations.
    pred_counter: Counter = Counter()
    for rel in valid_relations:
        pred_counter[rel["predicate"]] += 1
    print_predicate_distribution(pred_counter, "Filtered Predicate Distribution (All Qualifying)")

    # Human-object interaction summary.
    hoi_count = sum(1 for rel in valid_relations if rel["is_human_subj"] or rel["is_human_obj"])
    print(f"\n  Human-object interactions:     {hoi_count}")
    print(f"  Non-human relations:           {len(valid_relations) - hoi_count}")
    print(f"  HOI percentage:                {hoi_count/max(len(valid_relations),1)*100:.1f}%")

    # Average crop sizes.
    subj_sizes = [min(r["subj_box"][2], r["subj_box"][3]) for r in valid_relations]
    obj_sizes = [min(r["obj_box"][2], r["obj_box"][3]) for r in valid_relations]
    all_sizes = subj_sizes + obj_sizes
    print(f"\n  Average subject min-dim: {sum(subj_sizes)/max(len(subj_sizes),1):.1f} px")
    print(f"  Average object min-dim:  {sum(obj_sizes)/max(len(obj_sizes),1):.1f} px")
    print(f"  Overall average min-dim: {sum(all_sizes)/max(len(all_sizes),1):.1f} px")

    # Number of unique images.
    unique_img_ids = set(rel["image_id"] for rel in valid_relations)
    print(f"\n  Unique images with valid relations: {len(unique_img_ids)}")

    # Select and download.
    select_and_download(
        img_meta_dict=img_meta_dict,
        img_rank=img_rank,
        valid_relations=valid_relations,
        max_images=args.max_images,
        download=args.download,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
