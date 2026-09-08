"""Locate, download and validate the Visual Genome images the experiments need.

The relation experiments do NOT need all 108,077 VG images. After the repo's
filters (COCO-80 label space, 19-predicate allowlist, min box size 10px) only
27,145 images carry at least one usable relation, and the frozen split manifest
names exactly those. Downloading the full corpus would move ~25 GB to obtain
~7 GB of useful pixels.

    required = train_ids + val_ids + test_ids  from splits/e0_image_split.json

Typical use on a fresh machine:

    # 1. what do we actually need, and what is already here?
    python prepare_visual_genome.py --report

    # 2. fetch the missing ones (resumable, safe to re-run / interrupt)
    python prepare_visual_genome.py --download

    # 3. confirm every file is a readable image
    python prepare_visual_genome.py --validate

Resumability: a file that already exists on disk is never re-fetched, and each
download lands in a ``.part`` file that is renamed only after the bytes are
complete. Killing the process at any point and re-running resumes cleanly; a
truncated ``.part`` is discarded rather than mistaken for a finished image.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

DEFAULT_VG_ROOT = PROJ_ROOT / "data" / "visual_genome"
DEFAULT_MANIFEST = PROJ_ROOT / "splits" / "e0_image_split.json"

# image_data.json ships a per-image URL, which is what makes this reproducible:
# we never guess a filename pattern, we resolve each id through the metadata
# the dataset itself distributes.
IMAGE_DATA_URL = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip"
USER_AGENT = "Mozilla/5.0 (VG-prepare; relation-prediction-project)"

# Where _resolve_vg_image_path() looks. A flat images/ dir is what this script
# writes; the VG_100K / VG_100K_2 layout is what the official zips unpack to,
# and both are accepted so an existing download is not re-fetched.
SUBDIRS = ("", "VG_100K", "VG_100K_2")
EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def required_image_ids(manifest_path: Path) -> Tuple[Dict[str, List[int]], List[int]]:
    """Read the frozen split manifest and return per-split and combined ids."""
    if not manifest_path.is_file():
        raise SystemExit(
            f"[prepare] split manifest not found: {manifest_path}\n"
            "  This file defines which images the experiments need. It is\n"
            "  committed to the repo; if it is missing, regenerate it with\n"
            "    python eval_gt_relations.py --rebuild-split"
        )
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_split = {
        "train": [int(i) for i in m["train_ids"]],
        "val": [int(i) for i in m["val_ids"]],
        "test": [int(i) for i in m["test_ids"]],
    }
    combined = sorted({i for ids in per_split.values() for i in ids})
    return per_split, combined


def find_local(image_id: int, image_dir: Path) -> Optional[Path]:
    """Return the on-disk path for an image id, or None.

    Mirrors VGRelationshipDataset._resolve_vg_image_path so this script's idea
    of "present" is the same as the dataset's. If they disagreed, this tool
    would happily report full coverage for images training cannot open.
    """
    for sub in SUBDIRS:
        for ext in EXTS:
            p = image_dir / sub / f"{image_id}{ext}" if sub else image_dir / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


def load_url_map(vg_root: Path, needed: Set[int]) -> Dict[int, str]:
    """image_id -> download URL, from image_data.json."""
    meta = vg_root / "image_data.json"
    if not meta.is_file():
        raise SystemExit(
            f"[prepare] {meta} not found.\n"
            f"  It carries the per-image URLs. Get it with:\n"
            f"    python download_vg.py\n"
            f"  or download and unzip {IMAGE_DATA_URL}"
        )
    records = json.loads(meta.read_text(encoding="utf-8"))
    return {int(r["image_id"]): r["url"] for r in records if int(r["image_id"]) in needed}


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def fetch_one(image_id: int, url: str, image_dir: Path, timeout: float) -> Tuple[int, str]:
    """Download a single image. Returns (image_id, status)."""
    import requests

    dest = image_dir / f"{image_id}.jpg"
    if dest.is_file():
        return image_id, "skip"
    part = dest.with_suffix(".jpg.part")
    try:
        with requests.get(url, stream=True, timeout=timeout,
                          headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            return image_id, "empty"
        # Rename only once the body is fully written, so an interrupted run
        # never leaves a half image that later looks present.
        os.replace(part, dest)
        return image_id, "ok"
    except Exception as exc:  # noqa: BLE001 - report, never abort the batch
        part.unlink(missing_ok=True)
        return image_id, f"fail:{type(exc).__name__}"


def download_missing(missing: List[int], url_map: Dict[int, str], image_dir: Path,
                     workers: int, timeout: float, limit: Optional[int]) -> Dict[str, int]:
    image_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in missing if i in url_map]
    no_url = len(missing) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    print(f"[prepare] downloading {len(todo):,} images with {workers} workers "
          f"into {image_dir}")
    if no_url:
        print(f"[prepare]   WARNING: {no_url:,} required ids have no URL in image_data.json")

    counts: Dict[str, int] = {}
    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, i, url_map[i], image_dir, timeout) for i in todo]
        for fut in concurrent.futures.as_completed(futures):
            _, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 250 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [{done:,}/{len(todo):,}] {rate:.1f} img/s  "
                      f"eta {eta / 60:.1f} min  {counts}")
    counts["no_url"] = no_url
    return counts


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate(image_ids: List[int], image_dir: Path, workers: int) -> Tuple[List[int], List[int]]:
    """Open every present image; return (good_ids, corrupt_ids)."""
    from PIL import Image

    def check(iid: int):
        p = find_local(iid, image_dir)
        if p is None:
            return iid, None
        try:
            with Image.open(p) as im:
                im.verify()          # header/structure
            with Image.open(p) as im:
                im.convert("RGB").load()   # full decode, which is what training does
            return iid, True
        except Exception:
            return iid, False

    good, corrupt = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (iid, res) in enumerate(pool.map(check, image_ids), 1):
            if res is True:
                good.append(iid)
            elif res is False:
                corrupt.append(iid)
            if i % 2000 == 0:
                print(f"  validated {i:,}/{len(image_ids):,}")
    return good, corrupt


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def coverage_report(per_split: Dict[str, List[int]], image_dir: Path) -> Dict:
    report: Dict = {"image_dir": str(image_dir), "splits": {}}
    all_missing: List[int] = []
    for name, ids in per_split.items():
        present = [i for i in ids if find_local(i, image_dir) is not None]
        missing = [i for i in ids if find_local(i, image_dir) is None]
        all_missing.extend(missing)
        pct = 100.0 * len(present) / max(len(ids), 1)
        report["splits"][name] = {
            "required": len(ids), "present": len(present),
            "missing": len(missing), "coverage_pct": round(pct, 2),
        }
        status = "READY" if not missing else "INCOMPLETE"
        print(f"  {name:<6} required {len(ids):>6,}  present {len(present):>6,}  "
              f"missing {len(missing):>6,}  {pct:6.2f}%  {status}")
    total_req = sum(len(v) for v in per_split.values())
    total_present = total_req - len(all_missing)
    report["total"] = {
        "required": total_req, "present": total_present,
        "missing": len(all_missing),
        "coverage_pct": round(100.0 * total_present / max(total_req, 1), 2),
    }
    print(f"  {'TOTAL':<6} required {total_req:>6,}  present {total_present:>6,}  "
          f"missing {len(all_missing):>6,}  {report['total']['coverage_pct']:6.2f}%")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Locate / download / validate the VG images the experiments need.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--vg-root", type=Path, default=DEFAULT_VG_ROOT)
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Default: <vg-root>/images")
    ap.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--report", action="store_true",
                    help="Report coverage only (default when no action is given)")
    ap.add_argument("--download", action="store_true", help="Fetch missing images")
    ap.add_argument("--validate", action="store_true",
                    help="Decode every present image and report corrupt files")
    ap.add_argument("--delete-corrupt", action="store_true",
                    help="With --validate: remove files that fail to decode so a "
                         "later --download re-fetches them")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Download at most N images (for a smoke test)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the coverage report as JSON")
    args = ap.parse_args()

    image_dir = args.image_dir or (args.vg_root / "images")
    per_split, combined = required_image_ids(args.split_manifest)

    print("=" * 74)
    print("Visual Genome image preparation")
    print("=" * 74)
    print(f"  split manifest : {args.split_manifest}")
    print(f"  image dir      : {image_dir}")
    print(f"  required images: {len(combined):,} "
          f"(of 108,077 in VG; the rest carry no usable relation)")
    print()

    report = coverage_report(per_split, image_dir)

    if args.download:
        missing = [i for i in combined if find_local(i, image_dir) is None]
        if not missing:
            print("\n[prepare] nothing to download; all required images present.")
        else:
            print()
            url_map = load_url_map(args.vg_root, set(missing))
            counts = download_missing(missing, url_map, image_dir,
                                      args.workers, args.timeout, args.limit)
            print(f"\n[prepare] download summary: {counts}")
            print()
            report = coverage_report(per_split, image_dir)

    if args.validate:
        present = [i for i in combined if find_local(i, image_dir) is not None]
        print(f"\n[prepare] validating {len(present):,} present images …")
        good, corrupt = validate(present, image_dir, args.workers)
        print(f"[prepare]   decodable : {len(good):,}")
        print(f"[prepare]   corrupt   : {len(corrupt):,}")
        report["validation"] = {"checked": len(present), "good": len(good),
                                "corrupt": len(corrupt), "corrupt_ids": corrupt[:100]}
        if corrupt and args.delete_corrupt:
            for iid in corrupt:
                p = find_local(iid, image_dir)
                if p is not None:
                    p.unlink(missing_ok=True)
            print(f"[prepare]   deleted {len(corrupt):,} corrupt files; "
                  f"re-run with --download to refetch")
        elif corrupt:
            print("[prepare]   re-run with --validate --delete-corrupt to remove them")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[prepare] report written to {args.out}")

    missing_total = report["total"]["missing"]
    print()
    if missing_total == 0:
        print("[prepare] STATUS: READY — every required image is present.")
        return 0
    print(f"[prepare] STATUS: INCOMPLETE — {missing_total:,} images missing. "
          f"Run with --download.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
