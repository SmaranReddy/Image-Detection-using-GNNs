"""Precompute the CLIP feature cache for the relation experiments.

Why this exists as a separate step
----------------------------------
The dataset can build its own cache, but that path has three properties that
make it unusable for a 12-run experiment grid:

  * no resume - one 27k-image pass held entirely in memory, so a crash at
    image 26,000 loses everything;
  * union-region features were never written to disk at all. They were
    recomputed in-process on every run, one un-batched CLIP forward per
    sample, and held in a list indexed by sample *position* - which silently
    misaligns features and labels the moment the sample order changes;
  * crops are batched only within a single image, so the GPU idles on the
    (common) images that contain two or three objects.

This script does the pass once, batches crops across images, checkpoints to
shards as it goes, and writes both object and union features under stable keys:

    object : "{image_id}_obj_{object_id}"
    union  : "{image_id}_union_{subj_object_id}_{obj_object_id}"

Cache format
------------
A single file written by ClipCache.save() (torch.save), holding

    embeddings : FloatTensor (N, 768)   L2-normalised, one row per key
    index      : Dict[str, int]         key -> row
    metadata   : {"version": 2, "clip_dim": 768}

768 is CLIPVisionModel.pooler_output for openai/clip-vit-base-patch32 - the
pre-projection hidden state, NOT the 512-d joint image-text embedding.

Usage
-----
    python build_clip_cache.py --include-union            # GPU, resumable
    python build_clip_cache.py --report                   # coverage only
    python build_clip_cache.py --include-union --limit-images 200   # smoke test

Re-running after an interrupt resumes from the shards in <out>.shards/ and only
processes images that are not already covered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from relation_prediction.clip_cache import ClipCache  # noqa: E402
from relation_prediction.clip_extractor import CLIPExtractor, CLIP_DIM  # noqa: E402
from relation_prediction.vg_dataset import (  # noqa: E402
    PREDICATE_SCHEMES, VGRelationshipDataset,
)

DEFAULT_VG_ROOT = PROJ_ROOT / "data" / "visual_genome"


# ---------------------------------------------------------------------------
# what needs encoding
# ---------------------------------------------------------------------------

def enumerate_work(vg_root: Path, predicate_scheme: str, include_union: bool):
    """Return (per-image work plan, stats).

    The sample population is enumerated by instantiating the real dataset with
    use_visual=False. That is deliberate: the cache must cover exactly the
    samples training will ask for, and the only way to guarantee that is to run
    the same _load() the trainer runs rather than re-implementing its filters
    here and letting the two definitions drift.
    """
    print("[cache] enumerating samples via VGRelationshipDataset (use_visual=False) …")
    ds = VGRelationshipDataset(
        relationships_json=str(vg_root / "relationships.json"),
        image_data_json=str(vg_root / "image_data.json"),
        use_visual=False,
        predicate_scheme=predicate_scheme,
    )

    # object crops: every (image, object) that any retained sample references
    per_image: Dict[int, Dict[str, list]] = defaultdict(
        lambda: {"objects": [], "unions": []})
    wanted_objs = set()
    for subj_key, obj_key in ds.sample_keys:
        for key in (subj_key, obj_key):
            iid_s, oid_s = key.split("_obj_", 1)
            wanted_objs.add((int(iid_s), int(oid_s)))

    missing_boxes = 0
    for (iid, oid) in sorted(wanted_objs):
        box = ds._obj_box_map.get((iid, oid))
        if box is None:
            missing_boxes += 1
            continue
        per_image[iid]["objects"].append((f"{iid}_obj_{oid}", tuple(box)))

    n_union = 0
    if include_union:
        for subj_key, obj_key in ds.sample_keys:
            iid_s, subj_oid = subj_key.split("_obj_", 1)
            _, obj_oid = obj_key.split("_obj_", 1)
            iid = int(iid_s)
            sb = ds._obj_box_map.get((iid, int(subj_oid)))
            ob = ds._obj_box_map.get((iid, int(obj_oid)))
            if sb is None or ob is None:
                missing_boxes += 1
                continue
            ukey = CLIPExtractor.to_union_key(iid, subj_oid, obj_oid)
            union_box = (min(sb[0], ob[0]), min(sb[1], ob[1]),
                         max(sb[2], ob[2]), max(sb[3], ob[3]))
            per_image[iid]["unions"].append((ukey, union_box))
            n_union += 1

    # A pair can appear more than once (the same two objects related twice);
    # de-duplicate so the same region is not encoded repeatedly.
    for iid, work in per_image.items():
        seen = set()
        deduped = []
        for k, b in work["unions"]:
            if k not in seen:
                seen.add(k)
                deduped.append((k, b))
        work["unions"] = deduped

    stats = {
        "n_samples": len(ds.sample_keys),
        "n_images": len(per_image),
        "n_object_crops": sum(len(w["objects"]) for w in per_image.values()),
        "n_union_crops": sum(len(w["unions"]) for w in per_image.values()),
        "n_union_samples": n_union,
        "missing_boxes": missing_boxes,
        "predicate_scheme": predicate_scheme,
    }
    return per_image, stats


# ---------------------------------------------------------------------------
# shard-based resume
# ---------------------------------------------------------------------------

class ShardWriter:
    """Append-only shard store so an interrupted build resumes cheaply."""

    def __init__(self, shard_dir: Path, flush_every: int = 20000):
        self.shard_dir = shard_dir
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.keys: List[str] = []
        self.embs: List[torch.Tensor] = []
        self.done_images: set = set()
        self._pending_images: List[int] = []

    def load_existing(self) -> Tuple[Dict[str, torch.Tensor], set]:
        """Read shards already on disk. Returns (key->emb, covered image ids)."""
        found: Dict[str, torch.Tensor] = {}
        covered: set = set()
        shards = sorted(self.shard_dir.glob("shard_*.pt"))
        for sp in shards:
            try:
                blob = torch.load(sp, map_location="cpu", weights_only=False)
            except Exception as exc:  # a shard killed mid-write
                print(f"[cache]   discarding unreadable shard {sp.name}: {exc}")
                sp.unlink(missing_ok=True)
                continue
            for k, row in zip(blob["keys"], blob["embeddings"]):
                found[k] = row
            covered.update(int(i) for i in blob.get("images", []))
        if shards:
            print(f"[cache] resume: {len(shards)} shard(s), {len(found):,} embeddings, "
                  f"{len(covered):,} images already done")
        return found, covered

    def add(self, image_id: int, keys: Sequence[str], embs: torch.Tensor) -> None:
        self.keys.extend(keys)
        self.embs.append(embs)
        self._pending_images.append(image_id)
        if len(self.keys) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.keys:
            return
        idx = len(list(self.shard_dir.glob("shard_*.pt")))
        tmp = self.shard_dir / f"shard_{idx:05d}.pt.tmp"
        torch.save({"keys": self.keys,
                    "embeddings": torch.cat(self.embs, dim=0),
                    "images": self._pending_images}, tmp)
        os.replace(tmp, self.shard_dir / f"shard_{idx:05d}.pt")
        self.keys, self.embs, self._pending_images = [], [], []


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def resolve_image_path(image_dir: Path, iid: int) -> Optional[Path]:
    for sub in ("", "VG_100K", "VG_100K_2"):
        for ext in (".jpg", ".jpeg", ".png"):
            p = image_dir / sub / f"{iid}{ext}" if sub else image_dir / f"{iid}{ext}"
            if p.is_file():
                return p
    return None


def clamp(box, w, h):
    x1 = max(0.0, min(float(box[0]), float(w)))
    y1 = max(0.0, min(float(box[1]), float(h)))
    x2 = max(0.0, min(float(box[2]), float(w)))
    y2 = max(0.0, min(float(box[3]), float(h)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


def build(per_image, image_dir: Path, out_path: Path, batch_size: int,
          device: str, flush_every: int, limit_images: Optional[int]) -> Dict:
    shard_dir = Path(str(out_path) + ".shards")
    writer = ShardWriter(shard_dir, flush_every=flush_every)
    existing, covered = writer.load_existing()

    todo = [iid for iid in sorted(per_image) if iid not in covered]
    if limit_images is not None:
        todo = todo[:limit_images]

    print(f"[cache] {len(todo):,} images to process "
          f"({len(covered):,} already covered)")
    if not todo:
        print("[cache] nothing to do; merging shards.")

    extractor = CLIPExtractor(torch.device(device)) if todo else None

    stats = {"images_ok": 0, "images_missing": 0, "images_corrupt": 0,
             "crops_encoded": 0, "crops_degenerate": 0}

    # Cross-image crop buffer: keeps the GPU fed on images with few objects.
    buf_keys: List[str] = []
    buf_crops: List[Image.Image] = []
    buf_owner: List[int] = []

    def flush_buffer():
        if not buf_crops:
            return
        embs = extractor.encode_crops(buf_crops)
        stats["crops_encoded"] += len(buf_crops)
        # Attribute rows back to the image that produced them so a shard's
        # "images" list is honest about what it fully covers.
        by_img: Dict[int, List[int]] = defaultdict(list)
        for i, owner in enumerate(buf_owner):
            by_img[owner].append(i)
        for owner, rows in by_img.items():
            writer.add(owner, [buf_keys[i] for i in rows], embs[rows])
        buf_keys.clear()
        buf_crops.clear()
        buf_owner.clear()

    t0 = time.time()
    for n, iid in enumerate(todo, 1):
        path = resolve_image_path(image_dir, iid)
        if path is None:
            stats["images_missing"] += 1
            continue
        try:
            with Image.open(path) as im:
                pil = im.convert("RGB")
        except Exception:
            stats["images_corrupt"] += 1
            continue

        work = per_image[iid]
        for key, box in list(work["objects"]) + list(work["unions"]):
            cb = clamp(box, pil.width, pil.height)
            if cb is None:
                stats["crops_degenerate"] += 1
                continue
            buf_keys.append(key)
            buf_crops.append(pil.crop(cb))
            buf_owner.append(iid)
            if len(buf_crops) >= batch_size:
                flush_buffer()
        stats["images_ok"] += 1

        if n % 500 == 0 or n == len(todo):
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-6)
            print(f"  [{n:,}/{len(todo):,}] {rate:.1f} img/s  eta {eta/60:.1f} min  "
                  f"{stats['crops_encoded']:,} crops")

    flush_buffer()
    writer.flush()

    # ---- merge every shard into the final cache -------------------------
    merged, _ = writer.load_existing()
    merged.update(existing)
    if not merged:
        raise SystemExit("[cache] no embeddings produced - are the images present?")
    keys = sorted(merged)
    cache = ClipCache.build(keys, [merged[k] for k in keys])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache.save(str(out_path))
    print(f"\n[cache] wrote {len(keys):,} embeddings to {out_path}")
    print(f"[cache] shards kept in {shard_dir} (delete once the cache is verified)")
    stats["n_embeddings"] = len(keys)
    return stats


def report_coverage(per_image, out_path: Path) -> Dict:
    """Compare an existing cache against what the experiments require."""
    if not out_path.is_file():
        print(f"[cache] no cache at {out_path}")
        return {"exists": False}
    cache = ClipCache.load(str(out_path))
    need_obj, need_uni = [], []
    for work in per_image.values():
        need_obj += [k for k, _ in work["objects"]]
        need_uni += [k for k, _ in work["unions"]]
    have_obj = sum(1 for k in need_obj if k in cache)
    have_uni = sum(1 for k in need_uni if k in cache)
    rep = {
        "exists": True, "path": str(out_path), "n_in_cache": len(cache),
        "object": {"required": len(need_obj), "present": have_obj,
                   "coverage_pct": round(100.0 * have_obj / max(len(need_obj), 1), 2)},
        "union": {"required": len(need_uni), "present": have_uni,
                  "coverage_pct": round(100.0 * have_uni / max(len(need_uni), 1), 2)},
    }
    print(f"[cache] {out_path}  ({len(cache):,} embeddings)")
    print(f"  object crops  required {len(need_obj):>7,}  present {have_obj:>7,}  "
          f"{rep['object']['coverage_pct']:6.2f}%")
    print(f"  union  crops  required {len(need_uni):>7,}  present {have_uni:>7,}  "
          f"{rep['union']['coverage_pct']:6.2f}%")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Precompute the CLIP object/union feature cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--vg-root", type=Path, default=DEFAULT_VG_ROOT)
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="Default: <vg-root>/images")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <vg-root>/clip_cache_proper.pt")
    ap.add_argument("--include-union", action="store_true",
                    help="Also encode the union region of every related pair "
                         "(needed for the geometry+clip+union variant)")
    ap.add_argument("--predicate-scheme", type=str, default="v1",
                    choices=list(PREDICATE_SCHEMES))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None,
                    help="Default: cuda when available, else cpu")
    ap.add_argument("--flush-every", type=int, default=20000,
                    help="Write a resume shard after this many embeddings")
    ap.add_argument("--limit-images", type=int, default=None,
                    help="Process at most N images (smoke test)")
    ap.add_argument("--report", action="store_true",
                    help="Report coverage of an existing cache and exit")
    ap.add_argument("--stats-out", type=Path, default=None,
                    help="Write the build stats as JSON")
    args = ap.parse_args()

    image_dir = args.image_dir or (args.vg_root / "images")
    out_path = args.out or (args.vg_root / "clip_cache_proper.pt")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print("CLIP feature cache")
    print("=" * 74)
    print(f"  vg root    : {args.vg_root}")
    print(f"  image dir  : {image_dir}")
    print(f"  out        : {out_path}")
    print(f"  device     : {device}")
    print(f"  union      : {args.include_union}")
    print(f"  clip dim   : {CLIP_DIM} (CLIPVisionModel.pooler_output)")
    print()

    per_image, enum_stats = enumerate_work(args.vg_root, args.predicate_scheme,
                                           args.include_union)
    print(f"\n[cache] work plan: {enum_stats}")

    if args.report:
        rep = report_coverage(per_image, out_path)
        if args.stats_out:
            args.stats_out.write_text(json.dumps({"enumerate": enum_stats,
                                                  "coverage": rep}, indent=2),
                                      encoding="utf-8")
        obj_ok = rep.get("exists") and rep["object"]["coverage_pct"] >= 99.99
        uni_ok = (not args.include_union) or (
            rep.get("exists") and rep["union"]["coverage_pct"] >= 99.99)
        print("\n[cache] STATUS:", "READY" if (obj_ok and uni_ok) else "INCOMPLETE")
        return 0 if (obj_ok and uni_ok) else 1

    if not image_dir.is_dir():
        raise SystemExit(
            f"[cache] image dir not found: {image_dir}\n"
            f"  Run: python prepare_visual_genome.py --download")

    stats = build(per_image, image_dir, out_path, args.batch_size, device,
                  args.flush_every, args.limit_images)
    stats.update(enum_stats)
    print(f"\n[cache] build stats: {stats}")
    if stats["images_missing"]:
        print(f"[cache] WARNING: {stats['images_missing']:,} images were missing and "
              f"produced NO features. Samples referencing them will be dropped by "
              f"--require-visual (they are not silently zero-filled).")
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print()
    report_coverage(per_image, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
