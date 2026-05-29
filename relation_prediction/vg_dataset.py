"""
Visual Genome relationship dataset with visual-semantic features.

Supports two feature modes:
    1. Geometry-only (legacy): 5-dim geometric features.
    2. Visual-semantic (new):   Geometry + CLIP visual embeddings from object crops.

Expected data layout (set VG_ROOT in train.py or pass paths directly):
    <vg_root>/relationships.json
    <vg_root>/image_data.json
    <vg_root>/images/  (directory of VG images, e.g. 1.jpg, 2.jpg, ...)
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .clip_extractor import CLIPExtractor, CLIP_DIM
from .clip_cache import ClipCache
from .pose_extractor import PoseExtractor, POSE_FEATURE_DIM, POSE_OBJECT_FEATURE_DIM


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

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
    "man":          "person", "men":          "person",
    "woman":        "person", "women":        "person",
    "boy":          "person", "girl":         "person",
    "people":       "person", "child":        "person",
    "children":     "person", "guy":          "person",
    "lady":         "person",
    "bike":         "bicycle", "cycle":       "bicycle",
    "vehicle":      "car",    "automobile":  "car",
    "sofa":         "couch",
    "television":   "tv",     "tv monitor":  "tv",
    "monitor":      "tv",
    "cellphone":    "cell phone", "mobile":  "cell phone",
    "phone":        "cell phone",
    "motorbike":    "motorcycle",
    "aeroplane":    "airplane", "aero plane": "airplane",
    "dining table": "dining table",
    "plant":        "potted plant",
}


# ---------------------------------------------------------------------------
# Predicate allowlist and normalisation
# ---------------------------------------------------------------------------

ALLOWED_PREDICATES: frozenset = frozenset({
    "on", "under", "above", "next to", "near",
    "in", "holding", "riding", "sitting on",
    "standing on", "wearing", "carrying",
    "looking at", "attached to", "behind",
    "in front of", "over", "inside", "covering",
})

PREDICATE_MAP: Dict[str, str] = {
    "on top of":    "on", "lying on":     "on",
    "resting on":   "on",
    "next to":      "near", "beside":     "near",
    "close to":     "near",
    "underneath":   "under", "below":     "under",
    "riding on":    "riding", "mounted on": "riding",
    "holding in":   "holding", "grasping": "holding",
    "gripping":     "holding",
    "carrying in":  "carrying", "carried by": "carrying",
}


def normalize_predicate(pred: str) -> Optional[str]:
    pred = pred.lower().strip()
    pred = PREDICATE_MAP.get(pred, pred)
    if pred not in ALLOWED_PREDICATES:
        return None
    return pred


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    label = SYNONYM_MAP.get(label, label)
    if label not in COCO_LABELS and label.endswith("s") and label[:-1] in COCO_LABELS:
        label = label[:-1]
    return label if label in COCO_LABELS else "UNK"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

GEO_DIM = 5
MIN_BOX_SIZE = 10  # minimum box dimension (pixels) for meaningful CLIP crops

# Interaction-aware feature dimensions (optional, backward-compatible).
POSE_FEATURE_DIM  = 20   # compact pose features from PoseExtractor (see pose_extractor.py)
POSE_OBJECT_FEATURE_DIM = 7  # interaction-aware pose-object relative features
UNION_FEATURE_DIM = CLIP_DIM  # union-region CLIP embedding, same dim as single crop


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return float(x), float(y), float(x + w), float(y + h)


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def extract_geo_features(
    subj_box: Tuple[float, float, float, float],
    obj_box:  Tuple[float, float, float, float],
    img_w: float, img_h: float,
) -> List[float]:
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box
    scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
    ocx, ocy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
    sw, sh = max(sx2 - sx1, 1.0), max(sy2 - sy1, 1.0)
    ow, oh = max(ox2 - ox1, 1.0), max(oy2 - oy1, 1.0)
    denom_w, denom_h = max(img_w, 1.0), max(img_h, 1.0)
    dx     = (ocx - scx) / denom_w
    dy     = (ocy - scy) / denom_h
    log_wr = float(np.log(ow / sw))
    log_hr = float(np.log(oh / sh))
    iou    = compute_iou(subj_box, obj_box)
    return [dx, dy, log_wr, log_hr, iou]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self) -> None:
        self._tok2idx: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self._idx2tok: List[str] = [self.PAD, self.UNK]

    def add(self, token: str) -> int:
        if token not in self._tok2idx:
            idx = len(self._idx2tok)
            self._tok2idx[token] = idx
            self._idx2tok.append(token)
        return self._tok2idx[token]

    def __getitem__(self, token: str) -> int:
        return self._tok2idx.get(token, self._tok2idx[self.UNK])

    def __len__(self) -> int:
        return len(self._idx2tok)

    def token(self, idx: int) -> str:
        return self._idx2tok[idx] if 0 <= idx < len(self._idx2tok) else self.UNK

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._idx2tok, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path) as f:
            tokens = json.load(f)
        v = cls()
        v._idx2tok = tokens
        v._tok2idx = {t: i for i, t in enumerate(tokens)}
        return v


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VGRelationshipDataset(Dataset):
    CLIP_DIM = CLIP_DIM

    def __init__(
        self,
        relationships_json: str,
        image_data_json: str,
        vg_image_dir: Optional[str] = None,
        label_vocab: Optional[Vocab] = None,
        pred_vocab: Optional[Vocab] = None,
        min_pred_count: int = 50,
        max_samples: Optional[int] = None,
        use_visual: bool = False,
        clip_cache_path: Optional[str] = None,
        force_rebuild_cache: bool = False,
        require_visual: bool = False,
        use_pose: bool = False,
        use_pose_object: bool = False,
        use_union: bool = False,
    ) -> None:
        if require_visual and not use_visual:
            raise ValueError("require_visual=True requires use_visual=True")
        if use_union and not use_visual:
            raise ValueError("use_union=True requires use_visual=True")
        if use_pose and not use_visual:
            raise ValueError("use_pose=True requires use_visual=True")
        if use_pose_object and not use_pose:
            raise ValueError("use_pose_object=True requires use_pose=True")

        self.label_vocab = label_vocab if label_vocab is not None else Vocab()
        self.pred_vocab  = pred_vocab  if pred_vocab  is not None else Vocab()
        self._build_vocab = label_vocab is None

        self.vg_image_dir = vg_image_dir
        self.use_visual = use_visual
        self.require_visual = require_visual
        self.use_pose = use_pose
        self.use_pose_object = use_pose_object
        self.use_union = use_union
        self.clip_extractor: Optional[CLIPExtractor] = None
        self.pose_extractor: Optional[PoseExtractor] = None
        self.clip_cache: ClipCache = ClipCache()
        self._obj_box_map: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}

        # Load relationship samples with cache keys for visual features.
        # Also builds self._obj_box_map for CLIP cache construction.
        self.samples, self.sample_keys = self._load(
            relationships_json, image_data_json,
            min_pred_count=min_pred_count,
            max_samples=max_samples,
        )

        # Build CLIP cache (uses _obj_box_map populated by _load).
        if use_visual:
            self._init_clip_cache(
                clip_cache_path, force_rebuild_cache,
            )
            # Strict visual filtering: exclude any sample with missing CLIP embeddings.
            if require_visual:
                self._filter_strict_visual()

        # Interaction-aware feature caches (union-region CLIP, pose features).
        self.union_feats: List[torch.Tensor] = []
        self.pose_feats: List[torch.Tensor] = []
        self.pose_object_feats: List[torch.Tensor] = []
        if use_visual and (use_union or use_pose or use_pose_object):
            self._init_interaction_cache()

        # --- dataset inspection logging ---
        stats = self._load_stats
        total_raw       = stats["total_raw"]
        kept_before_cap = stats["kept_before_cap"]
        pred_counter    = stats["pred_counter"]

        print(f"\nTotal raw samples:       {total_raw}")
        print(f"Kept samples:            {kept_before_cap}")
        print(f"Dropped samples:         {total_raw - kept_before_cap}")
        print(f"Skipped (tiny box):      {stats['skipped_small']}")
        if total_raw:
            print(f"Retention ratio:         {kept_before_cap / total_raw:.2f}")
        print(f"Total usable samples:    {len(self.samples)}")
        print(f"\nNumber of unique predicates: {len(pred_counter)}")
        print("Top predicates by frequency:")
        for pred, count in pred_counter.most_common():
            print(f"  {pred}: {count}")
        if use_visual:
            print(f"\nVisual features: ENABLED (CLIP_DIM={CLIP_DIM})")
            print(f"  Cache size: {len(self.clip_cache)} embeddings")
            if require_visual:
                print("  Strict visual filtering: ENABLED (require_visual=True)")
            if use_union:
                print(f"  Union-region features: ENABLED ({UNION_FEATURE_DIM}-dim CLIP)")
            if use_pose:
                print(f"  Pose features: ENABLED ({POSE_FEATURE_DIM}-dim)")
        else:
            print("\nVisual features: DISABLED (geometry-only)")

    # ------------------------------------------------------------------

    def _resolve_vg_image_path(self, image_id: Union[int, str]) -> Optional[str]:
        image_id = str(image_id)
        extensions = [".jpg", ".png", ".jpeg"]
        vg_subdirs = ["", "VG_100K", "VG_100K_2"]
        for subdir in vg_subdirs:
            for ext in extensions:
                if subdir:
                    p = os.path.join(self.vg_image_dir, subdir, f"{image_id}{ext}")
                else:
                    p = os.path.join(self.vg_image_dir, f"{image_id}{ext}")
                if os.path.isfile(p):
                    return p
        return None

    def _init_clip_cache(
        self,
        clip_cache_path: Optional[str],
        force_rebuild: bool,
    ) -> None:
        if clip_cache_path is None:
            clip_cache_path = "clip_cache.pt"

        # Backward compat: load old .pkl format, convert to new on the fly.
        if not force_rebuild:
            loaded = self._try_load_cache(clip_cache_path)
            if loaded is not None:
                self.clip_cache = loaded
                return

        # Build cache from scratch.
        if self.vg_image_dir is None or not os.path.isdir(self.vg_image_dir):
            raise RuntimeError(
                "Visual features requested but no VG images found.\n"
                f"  vg_image_dir={self.vg_image_dir}\n"
                "  Download VG images and set vg_image_dir, or disable use_visual."
            )

        # Group objects by image using pre-built _obj_box_map (populated by _load).
        image_to_objects: Dict[int, List[Tuple[int, Tuple]]] = defaultdict(list)
        for (iid, oid), box in self._obj_box_map.items():
            image_to_objects[iid].append((oid, box))

        n_images = len(image_to_objects)
        print(f"[VG] Extracting CLIP embeddings for {n_images} images …")
        self.clip_extractor = CLIPExtractor()

        keys_list: List[str] = []
        emb_list: List[torch.Tensor] = []
        clip_missing = 0
        for img_idx, (iid, objects) in enumerate(image_to_objects.items()):
            img_path = self._resolve_vg_image_path(iid)
            if img_path is None:
                if clip_missing < 20:
                    candidates_tried = [
                        os.path.join(self.vg_image_dir, f"{iid}.jpg"),
                        os.path.join(self.vg_image_dir, "VG_100K", f"{iid}.jpg"),
                        os.path.join(self.vg_image_dir, "VG_100K_2", f"{iid}.jpg"),
                    ]
                    exist_status = " | ".join(
                        f"exist={os.path.isfile(p)}" for p in candidates_tried
                    )
                    print(f"[DEBUG] CLIP cache: missing image id={iid!r} (type={type(iid).__name__})")
                    print(f"[DEBUG]   Candidates: {exist_status}")
                    clip_missing += 1
                continue

            pil_img = Image.open(img_path).convert("RGB")
            boxes = [box for _, box in objects]
            embs = self.clip_extractor.extract_crops(pil_img, boxes)

            for (oid, _), emb in zip(objects, embs):
                keys_list.append(f"{iid}_obj_{oid}")
                emb_list.append(emb)

            if (img_idx + 1) % 1000 == 0:
                print(f"  [{img_idx + 1}/{n_images}] {len(keys_list)} embeddings cached")

        self.clip_cache = ClipCache.build(keys_list, emb_list)
        print(f"[VG] CLIP cache built: {len(self.clip_cache)} embeddings")
        if clip_missing > 0:
            print(f"[VG]   WARNING: {clip_missing} images missing during CLIP cache build")

        # Ensure .pt extension for new format.
        save_path = clip_cache_path
        if save_path.endswith(".pkl"):
            save_path = save_path.replace(".pkl", ".pt")
            print(f"[VG] Changing cache extension to .pt: {save_path}")
        self.clip_cache.save(save_path)
        print(f"[VG] CLIP cache saved to {save_path}")

    def _try_load_cache(self, path: str) -> Optional[ClipCache]:
        """Try loading cache, handling both new .pt and legacy .pkl formats.

        Resolution order:
            1. Exact .pt path
            2. Same-basename .pt (e.g. user passed .pkl but .pt exists)
            3. Legacy .pkl → convert to .pt
        """
        # 1. Exact .pt path
        if path.endswith(".pt") and os.path.exists(path):
            print(f"[VG] Loading CLIP cache from {path} …")
            return ClipCache.load(path)

        # 2. Same-basename .pt (handles stale .pkl references gracefully)
        base = path.rsplit(".", 1)[0] if "." in path else path
        pt_path = base + ".pt"
        if pt_path != path and os.path.exists(pt_path):
            print(f"[VG] Loading CLIP cache from {pt_path} …")
            return ClipCache.load(pt_path)

        # 3. Legacy .pkl — convert on load.
        legacy_path = base + ".pkl"
        if legacy_path != path and os.path.exists(legacy_path):
            print(f"[VG] Converting legacy pickle cache from {legacy_path} …")
            cache = ClipCache.load_old_pickle(legacy_path)
            cache.save(pt_path)
            print(f"[VG] Converted and saved to {pt_path}")
            return cache

        return None

    # ------------------------------------------------------------------
    # Strict visual-semantic filtering
    # ------------------------------------------------------------------

    def _filter_strict_visual(self) -> None:
        """Remove samples with missing CLIP embeddings.

        Only retains samples where BOTH subject and object CLIP cache
        entries exist and produce non-zero feature vectors.
        """
        total_before = len(self.samples)
        kept_samples: List = []
        kept_keys: List[Tuple[str, str]] = []
        removed = 0

        for sample, (subj_key, obj_key) in zip(self.samples, self.sample_keys):
            if self._has_valid_clip(subj_key) and self._has_valid_clip(obj_key):
                kept_samples.append(sample)
                kept_keys.append((subj_key, obj_key))
            else:
                removed += 1

        self.samples = kept_samples
        self.sample_keys = kept_keys
        total_after = len(self.samples)

        # Compute post-filter predicate distribution.
        new_pred_counter: Counter = Counter()
        for sample in self.samples:
            new_pred_counter[self.pred_vocab.token(sample[3])] += 1

        # Extract unique VG image IDs from retained sample keys.
        unique_images: set = set()
        for _, (subj_key, _) in zip(self.samples, self.sample_keys):
            if "_obj_" in subj_key:
                unique_images.add(subj_key.split("_obj_")[0])

        # --- strict filtering report ---
        print("\n" + "=" * 65)
        print("  STRICT VISUAL FILTERING REPORT")
        print("=" * 65)
        print(f"  Total samples before:      {total_before}")
        print(f"  Retained samples:          {total_after}")
        print(f"  Removed samples:           {removed}")
        if total_before > 0:
            retained_pct = 100.0 * total_after / total_before
            removed_pct  = 100.0 * removed / total_before
            print(f"  % retained:                {retained_pct:.2f}%")
            print(f"  % removed:                 {removed_pct:.2f}%")
        print(f"  Unique VG images:          {len(unique_images)}")

        # Verify no zero vectors remain.
        self._validate_features_nonzero()

        # Predicate distribution after filtering.
        print(f"\n  Predicate distribution after strict filtering:")
        for pred, count in new_pred_counter.most_common():
            print(f"    {pred}: {count}")
        print("=" * 65 + "\n")

    def _has_valid_clip(self, key: str) -> bool:
        """Check if a cache key has a non-zero CLIP embedding."""
        if key not in self.clip_cache:
            return False
        emb = self.clip_cache.get(key)
        if emb is None:
            return False
        return emb.norm().item() > 0.0

    def compute_clip_coverage(self) -> tuple:
        """Compute real CLIP coverage.

        Returns:
            (real_count, total_count, coverage_pct) for the current dataset.
            In strict mode, coverage_pct is always 100.0.
        """
        if not self.use_visual:
            return 0, 0, 0.0
        if self.require_visual:
            return len(self.samples), len(self.samples), 100.0
        total = len(self.samples)
        if total == 0:
            return 0, 0, 0.0
        real = 0
        for sk, ok in self.sample_keys:
            if sk in self.clip_cache and ok in self.clip_cache:
                es = self.clip_cache.get(sk)
                eo = self.clip_cache.get(ok)
                if es is not None and eo is not None and es.norm().item() > 0.0 and eo.norm().item() > 0.0:
                    real += 1
        pct = 100.0 * real / max(total, 1)
        return real, total, pct

    def _validate_features_nonzero(self) -> None:
        """Verify ALL retained samples have non-zero CLIP feature norms."""
        norms_subj: List[float] = []
        norms_obj: List[float] = []
        for idx in range(len(self.samples)):
            subj_key, obj_key = self.sample_keys[idx]
            subj_feat = self.clip_cache.get(subj_key)
            obj_feat  = self.clip_cache.get(obj_key)
            if subj_feat is None:
                raise RuntimeError(
                    f"[STRICT VISUAL] Sample {idx}: subj_feat not in cache"
                )
            if obj_feat is None:
                raise RuntimeError(
                    f"[STRICT VISUAL] Sample {idx}: obj_feat not in cache"
                )
            subj_norm = subj_feat.norm().item()
            obj_norm  = obj_feat.norm().item()
            if subj_norm == 0.0:
                raise RuntimeError(
                    f"[STRICT VISUAL] Sample {idx}: subj_feat is ALL ZERO"
                )
            if obj_norm == 0.0:
                raise RuntimeError(
                    f"[STRICT VISUAL] Sample {idx}: obj_feat is ALL ZERO"
                )
            norms_subj.append(subj_norm)
            norms_obj.append(obj_norm)

        all_norms = norms_subj + norms_obj
        mean_norm = sum(all_norms) / len(all_norms)
        min_norm  = min(all_norms)
        max_norm  = max(all_norms)
        print(f"  Feature norm validation:  ALL NON-ZERO ({len(all_norms)} features checked)")
        print(f"  Mean feature norm:        {mean_norm:.4f}")
        print(f"  Min feature norm:         {min_norm:.4f}")
        print(f"  Max feature norm:         {max_norm:.4f}")
        rng = random.Random(42)
        sample_idxs = rng.sample(range(len(all_norms)), min(5, len(all_norms)))
        norms_str = ", ".join(f"{all_norms[i]:.4f}" for i in sample_idxs)
        print(f"  Random feature norms:     [{norms_str}]")

    # ------------------------------------------------------------------
    # Interaction-aware feature cache (union-region CLIP + pose)
    # ------------------------------------------------------------------

    def _init_interaction_cache(self) -> None:
        from collections import defaultdict
        n = len(self.samples)
        self.union_feats = [torch.zeros(UNION_FEATURE_DIM) for _ in range(n)]
        self.pose_feats = [torch.zeros(POSE_FEATURE_DIM) for _ in range(n)]
        self.pose_object_feats = [torch.zeros(POSE_OBJECT_FEATURE_DIM) for _ in range(n)]

        if self.vg_image_dir is None or not os.path.isdir(self.vg_image_dir):
            print("[VG] Interaction cache: no images dir, using zeros.")
            print(f"[VG]   vg_image_dir={self.vg_image_dir!r}")
            return

        # ------------------------------------------------------------------
        # Path validation: verify image directories actually exist
        # ------------------------------------------------------------------
        vg_100k_dir = os.path.join(self.vg_image_dir, "VG_100K")
        vg_100k_2_dir = os.path.join(self.vg_image_dir, "VG_100K_2")
        print(f"[VG]   Image root:  {self.vg_image_dir}")
        print(f"[VG]   VG_100K:     {vg_100k_dir}  exist={os.path.isdir(vg_100k_dir)}")
        print(f"[VG]   VG_100K_2:   {vg_100k_2_dir}  exist={os.path.isdir(vg_100k_2_dir)}")
        if not os.path.isdir(vg_100k_dir) and not os.path.isdir(vg_100k_2_dir):
            print("[VG]   WARNING: Neither VG_100K nor VG_100K_2 found!")

        # Quick spot-check: try to resolve a known image
        _test_path = self._resolve_vg_image_path("1")
        print(f"[VG]   Spot-check id=1: {_test_path or 'NOT FOUND'}")

        # ------------------------------------------------------------------

        if self.use_union and self.clip_extractor is None:
            self.clip_extractor = CLIPExtractor()

        pose_available = True
        if self.use_pose:
            self.pose_extractor = PoseExtractor()
            if not PoseExtractor.is_available():
                print("[VG] Pose features requested but MediaPipe not available. Pose will be zeros.")
                pose_available = False

        # Group sample indices by image id. Normalize to str for consistency.
        image_to_sample_idxs: Dict[str, List[int]] = defaultdict(list)
        for idx, (subj_key, obj_key) in enumerate(self.sample_keys):
            if "_obj_" in subj_key:
                iid = str(subj_key.split("_obj_")[0])
                image_to_sample_idxs[iid].append(idx)

        total_imgs = len(image_to_sample_idxs)
        processed = 0
        print(f"[VG] Extracting interaction features for {total_imgs} images …")

        # Debug counters
        total_samples_processed = 0
        union_success = 0
        pose_success = 0
        pose_fail = 0
        missing_images = 0
        corrupt_images = 0
        missing_boxes = 0
        person_samples = 0
        skipped_non_person = 0

        # Inspect box map key types (for diagnostics)
        box_iid_type = str
        box_oid_type = int
        if self._obj_box_map:
            test_key = next(iter(self._obj_box_map.keys()))
            box_iid_type = type(test_key[0])
            box_oid_type = type(test_key[1])

        failures_logged = 0
        for iid, sample_idxs in image_to_sample_idxs.items():
            img_path = self._resolve_vg_image_path(iid)
            if img_path is None:
                if failures_logged < 20:
                    candidates_tried = [
                        os.path.join(self.vg_image_dir, f"{iid}.jpg"),
                        os.path.join(self.vg_image_dir, "VG_100K", f"{iid}.jpg"),
                        os.path.join(self.vg_image_dir, "VG_100K_2", f"{iid}.jpg"),
                    ]
                    exist_status = " | ".join(
                        f"exist={os.path.isfile(p)}" for p in candidates_tried
                    )
                    print(f"[DEBUG] Missing image id={iid!r} (type={type(iid).__name__}) in {self.vg_image_dir}")
                    print(f"[DEBUG]   Candidates: {exist_status}")
                    failures_logged += 1
                missing_images += 1
                continue

            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                corrupt_images += 1
                if corrupt_images <= 5:
                    print(f"[DEBUG] Corrupt image: {img_path} — {e}")
                continue

            img_id = int(iid)

            for idx in sample_idxs:
                subj_key, obj_key = self.sample_keys[idx]
                subj_oid = int(subj_key.split("_obj_")[1])
                obj_oid = int(obj_key.split("_obj_")[1])

                # Lookup boxes with type-consistent keys.
                # Primary: (int, int) — matches _load's iid_int normalization.
                subj_box = self._obj_box_map.get((img_id, subj_oid))
                obj_box = self._obj_box_map.get((img_id, obj_oid))
                # Fallback: try (str, int) in case _load stored string keys.
                if subj_box is None or obj_box is None:
                    subj_box = self._obj_box_map.get((str(img_id), subj_oid), subj_box)
                    obj_box = self._obj_box_map.get((str(img_id), obj_oid), obj_box)
                if subj_box is None or obj_box is None:
                    missing_boxes += 1
                    if missing_boxes <= 5:
                        print(f"[DEBUG] Missing box: img_id={img_id} subj_oid={subj_oid} obj_oid={obj_oid} "
                              f"subj_key={subj_key!r} obj_key={obj_key!r}")
                    continue

                total_samples_processed += 1

                # Union-region CLIP embedding.
                if self.use_union:
                    union_box = (
                        min(subj_box[0], obj_box[0]),
                        min(subj_box[1], obj_box[1]),
                        max(subj_box[2], obj_box[2]),
                        max(subj_box[3], obj_box[3]),
                    )
                    uemb = self.clip_extractor.extract_crop(pil_img, union_box)
                    self.union_feats[idx] = uemb.clone()
                    union_success += 1

                # Pose features (only for person subjects).
                if self.use_pose and pose_available:
                    subj_label = self.label_vocab.token(self.samples[idx][0])
                    if subj_label == "person":
                        person_samples += 1
                        pemb = self.pose_extractor.extract_pose_features(pil_img, subj_box)
                        if pemb is not None:
                            self.pose_feats[idx] = pemb.clone()
                            pose_success += 1
                        else:
                            pose_fail += 1
                        # Pose-object interaction features.
                        if self.use_pose_object:
                            poemb = self.pose_extractor.extract_pose_object_features(pil_img, subj_box, obj_box)
                            if poemb is not None:
                                self.pose_object_feats[idx] = poemb.clone()
                    else:
                        skipped_non_person += 1

            processed += 1
            if processed % 500 == 0:
                print(f"  [{processed}/{total_imgs}] interaction features cached")

        union_count = sum(1 for f in self.union_feats if f.norm().item() > 0) if self.use_union else 0
        pose_count = sum(1 for f in self.pose_feats if f.norm().item() > 0) if self.use_pose else 0
        pose_object_count = sum(1 for f in self.pose_object_feats if f.norm().item() > 0) if self.use_pose_object else 0
        print(f"[VG] Interaction cache built: {union_count} union, {pose_count} pose, {pose_object_count} pose_object features")
        print(f"[VG]   Total unique images resolved:     {processed}")
        print(f"[VG]   Total image pairs processed:      {total_samples_processed}")
        if self.use_union:
            print(f"[VG]   Successful union features:       {union_success}")
        if self.use_pose:
            print(f"[VG]   Successful pose features:        {pose_success}")
            if self.use_pose_object:
                print(f"[VG]   Successful pose-object features: {pose_object_count}")
            print(f"[VG]   Failed pose detections:          {pose_fail}")
            print(f"[VG]   Person samples:                  {person_samples}")
            print(f"[VG]   Skipped (non-person subject):    {skipped_non_person}")
        if missing_images > 0:
            print(f"[VG]   WARNING: Missing images skipped:     {missing_images}")
        if corrupt_images > 0:
            print(f"[VG]   WARNING: Corrupt images skipped:     {corrupt_images}")
        if missing_boxes > 0:
            print(f"[VG]   WARNING: Missing box lookups:        {missing_boxes}")
            print(f"[VG]   Box map key type:   ({box_iid_type.__name__}, {box_oid_type.__name__})")
            print(f"[VG]   Lookup key type:    (int, int)")

    # ------------------------------------------------------------------

    def _load(
        self,
        rel_path: str,
        img_path: str,
        min_pred_count: int,
        max_samples: Optional[int],
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Returns:
            samples:      List of (subj_idx, obj_idx, geo_feats, pred_idx)
            sample_keys:  List of (subj_cache_key, obj_cache_key)
                          (empty strings if use_visual is False or no object_id)
        """
        with open(img_path) as f:
            img_meta: List[Dict] = json.load(f)
        img_size: Dict[int, Tuple[int, int]] = {
            m["image_id"]: (int(m["width"]), int(m["height"]))
            for m in img_meta
        }

        with open(rel_path) as f:
            all_rels: List[Dict] = json.load(f)

        if self._build_vocab:
            for pred in sorted(ALLOWED_PREDICATES):
                self.pred_vocab.add(pred)

        raw: List[Tuple[str, str, List[float], str, str, str]] = []
        total_raw = 0
        skipped_small = 0

        for img in all_rels:
            iid = img.get("image_id")
            img_w, img_h = img_size.get(iid, (1, 1))

            for r in img.get("relationships", []):
                total_raw += 1
                pred = normalize_predicate(r.get("predicate", ""))
                if pred is None:
                    continue

                subj_d = r.get("subject", {})
                obj_d  = r.get("object",  {})

                subj_name = normalize_label(_get_name(subj_d))
                obj_name  = normalize_label(_get_name(obj_d))
                if subj_name == "UNK" or obj_name == "UNK":
                    continue

                subj_box = _xywh_to_xyxy(
                    subj_d.get("x", 0), subj_d.get("y", 0),
                    subj_d.get("w", 1), subj_d.get("h", 1),
                )
                obj_box = _xywh_to_xyxy(
                    obj_d.get("x", 0), obj_d.get("y", 0),
                    obj_d.get("w", 1), obj_d.get("h", 1),
                )

                # Skip degenerate crops that produce meaningless CLIP embeddings.
                if (subj_box[2] - subj_box[0]) < MIN_BOX_SIZE or (subj_box[3] - subj_box[1]) < MIN_BOX_SIZE:
                    skipped_small += 1
                    continue
                if (obj_box[2] - obj_box[0]) < MIN_BOX_SIZE or (obj_box[3] - obj_box[1]) < MIN_BOX_SIZE:
                    skipped_small += 1
                    continue

                geo = extract_geo_features(subj_box, obj_box, img_w, img_h)

                # Cache keys for visual feature lookup.
                subj_oid = subj_d.get("object_id", -1)
                obj_oid  = obj_d.get("object_id", -1)
                subj_key = f"{iid}_obj_{subj_oid}" if subj_oid >= 0 else ""
                obj_key  = f"{iid}_obj_{obj_oid}"  if obj_oid >= 0 else ""

                # Record valid object boxes for CLIP cache construction.
                # Normalize iid to int for type-consistent lookups.
                iid_int = int(iid) if not isinstance(iid, int) else iid
                if subj_oid >= 0:
                    self._obj_box_map[(iid_int, subj_oid)] = subj_box
                if obj_oid >= 0:
                    self._obj_box_map[(iid_int, obj_oid)] = obj_box

                raw.append((subj_name, obj_name, geo, pred, subj_key, obj_key))

        kept_before_cap = len(raw)

        if self._build_vocab:
            for subj_name, obj_name, _, _, _, _ in raw:
                self.label_vocab.add(subj_name)
                self.label_vocab.add(obj_name)

        if max_samples is not None:
            raw = raw[:max_samples]

        self._load_stats = {
            "total_raw":       total_raw,
            "kept_before_cap": kept_before_cap,
            "skipped_small":   skipped_small,
            "pred_counter":    Counter(r[3] for r in raw),
        }

        samples: List[Tuple] = []
        sample_keys: List[Tuple[str, str]] = []
        for subj_name, obj_name, geo, pred, subj_key, obj_key in raw:
            samples.append((
                self.label_vocab[subj_name],
                self.label_vocab[obj_name],
                geo,
                self.pred_vocab[pred],
            ))
            sample_keys.append((subj_key, obj_key))

        return samples, sample_keys

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj_idx, obj_idx, geo, pred_idx = self.samples[idx]

        result = (
            torch.tensor(subj_idx, dtype=torch.long),
            torch.tensor(obj_idx,  dtype=torch.long),
            torch.tensor(geo,      dtype=torch.float32),
            torch.tensor(pred_idx, dtype=torch.long),
        )

        if self.use_visual:
            subj_key, obj_key = self.sample_keys[idx]
            if self.require_visual:
                subj_feat = self.clip_cache.get(subj_key)
                obj_feat = self.clip_cache.get(obj_key)
                if subj_feat is None or obj_feat is None:
                    raise RuntimeError(
                        f"[STRICT VISUAL] Sample {idx}: missing CLIP embedding. "
                        f"subj_key={subj_key}, obj_key={obj_key}. "
                        "All samples should have valid CLIP features in strict mode."
                    )
                if subj_feat.norm().item() == 0.0 or obj_feat.norm().item() == 0.0:
                    raise RuntimeError(
                        f"[STRICT VISUAL] Sample {idx}: zero CLIP embedding detected. "
                        "All samples must have non-zero CLIP features in strict mode."
                    )
            else:
                subj_feat = self.clip_cache.get(subj_key, torch.zeros(CLIP_DIM))
                obj_feat = self.clip_cache.get(obj_key, torch.zeros(CLIP_DIM))
            result = result + (subj_feat.clone(), obj_feat.clone())

            if self.use_union:
                uf = self.union_feats[idx] if idx < len(self.union_feats) else torch.zeros(UNION_FEATURE_DIM)
                result = result + (uf.clone(),)

            if self.use_pose:
                pf = self.pose_feats[idx] if idx < len(self.pose_feats) else torch.zeros(POSE_FEATURE_DIM)
                result = result + (pf.clone(),)

            if self.use_pose_object:
                pof = self.pose_object_feats[idx] if idx < len(self.pose_object_feats) else torch.zeros(POSE_OBJECT_FEATURE_DIM)
                result = result + (pof.clone(),)

        return result


# ---------------------------------------------------------------------------
# Predicate prior computation for logit debiasing
# ---------------------------------------------------------------------------

def compute_predicate_priors(
    pred_vocab: Vocab,
    predicate_distribution: Optional[Dict[str, int]] = None,
    training_meta_path: Optional[str] = None,
    vg_relationships_path: Optional[str] = None,
    vg_image_data_path: Optional[str] = None,
    smoothing: float = 1e-6,
) -> Dict[str, float]:
    """Compute global predicate frequency priors for logit debiasing.

    Args:
        pred_vocab: Predicate vocabulary.
        predicate_distribution: Dict mapping predicate name to count.
        training_meta_path: Path to training_meta.json (fallback).
        vg_relationships_path: Path to VG relationships.json (fallback).
        vg_image_data_path: Path to VG image_data.json (fallback).
        smoothing: Additive smoothing for rare predicates.

    Returns:
        Dict mapping predicate name to normalized probability.
    """
    counts: Dict[str, int] = {}

    if predicate_distribution is not None:
        counts = dict(predicate_distribution)
    elif training_meta_path and os.path.exists(training_meta_path):
        with open(training_meta_path) as f:
            meta = json.load(f)
        counts = meta.get("predicate_distribution", {})

    if not counts and vg_relationships_path and vg_image_data_path:
        counts = _compute_predicate_counts_from_vg(
            vg_relationships_path, vg_image_data_path, pred_vocab,
        )

    if not counts:
        raise ValueError(
            "Cannot compute predicate priors. Provide predicate_distribution, "
            "training_meta_path, or vg_relationships_path."
        )

    total = sum(counts.values()) + smoothing * len(pred_vocab)
    priors: Dict[str, float] = {}
    for pred_name in pred_vocab._idx2tok:
        if pred_name in (Vocab.PAD, Vocab.UNK):
            continue
        raw_count = counts.get(pred_name, 0)
        priors[pred_name] = (raw_count + smoothing) / total

    return priors


def _compute_predicate_counts_from_vg(
    rel_path: str,
    img_path: str,
    pred_vocab: Vocab,
) -> Dict[str, int]:
    """Compute predicate frequencies directly from VG relationship data."""
    from collections import Counter
    with open(img_path) as f:
        img_meta = json.load(f)
    with open(rel_path) as f:
        all_rels = json.load(f)

    counts: Counter = Counter()
    for img in all_rels:
        for r in img.get("relationships", []):
            pred = normalize_predicate(r.get("predicate", ""))
            if pred is None:
                continue
            if pred not in pred_vocab._tok2idx:
                continue
            counts[pred] += 1
    return dict(counts)


def save_predicate_priors(
    priors: Dict[str, float],
    output_path: str,
) -> None:
    """Save predicate priors to JSON."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"[predicate_priors] Saved to {output_path}")


def load_predicate_priors(path: str) -> Dict[str, float]:
    """Load predicate priors from JSON."""
    if not os.path.exists(path):
        print(f"[predicate_priors] File not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_name(entity: Dict) -> str:
    name = entity.get("name") or ""
    if not name:
        names = entity.get("names", [])
        name = names[0] if names else ""
    return name.lower().strip()
