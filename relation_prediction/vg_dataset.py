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
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .clip_extractor import CLIPExtractor, CLIP_DIM


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
    "on top of":    "on", "sitting on":   "on",
    "standing on":  "on", "lying on":     "on",
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
    ) -> None:
        self.label_vocab = label_vocab if label_vocab is not None else Vocab()
        self.pred_vocab  = pred_vocab  if pred_vocab  is not None else Vocab()
        self._build_vocab = label_vocab is None

        self.vg_image_dir = vg_image_dir
        self.use_visual = use_visual
        self.clip_extractor: Optional[CLIPExtractor] = None
        self.clip_cache: Dict[str, torch.Tensor] = {}

        # Load relationship samples with cache keys for visual features.
        self.samples, self.sample_keys = self._load(
            relationships_json, image_data_json,
            min_pred_count=min_pred_count,
            max_samples=max_samples,
        )

        # Build or load CLIP cache (needs to know which images/objects
        # from the relationship data).
        if use_visual:
            self._init_clip_cache(
                relationships_json, image_data_json,
                clip_cache_path, force_rebuild_cache,
            )

        # --- dataset inspection logging ---
        stats = self._load_stats
        total_raw       = stats["total_raw"]
        kept_before_cap = stats["kept_before_cap"]
        pred_counter    = stats["pred_counter"]

        print(f"\nTotal raw samples:    {total_raw}")
        print(f"Kept samples:         {kept_before_cap}")
        print(f"Dropped samples:      {total_raw - kept_before_cap}")
        if total_raw:
            print(f"Retention ratio:      {kept_before_cap / total_raw:.2f}")
        print(f"Total usable samples: {len(self.samples)}")
        print(f"\nNumber of unique predicates: {len(pred_counter)}")
        print("Top predicates by frequency:")
        for pred, count in pred_counter.most_common():
            print(f"  {pred}: {count}")
        if use_visual:
            print(f"\nVisual features: ENABLED (CLIP_DIM={CLIP_DIM})")
            print(f"  Cache size: {len(self.clip_cache)} embeddings")
        else:
            print("\nVisual features: DISABLED (geometry-only)")

    # ------------------------------------------------------------------

    def _init_clip_cache(
        self,
        rel_path: str,
        img_path: str,
        clip_cache_path: Optional[str],
        force_rebuild: bool,
    ) -> None:
        if clip_cache_path is None:
            clip_cache_path = "clip_cache.pkl"

        if os.path.exists(clip_cache_path) and not force_rebuild:
            print(f"[VG] Loading CLIP cache from {clip_cache_path} …")
            with open(clip_cache_path, "rb") as f:
                self.clip_cache = pickle.load(f)
            return

        # Build cache from scratch.
        if self.vg_image_dir is None or not os.path.isdir(self.vg_image_dir):
            raise RuntimeError(
                "Visual features requested but no VG images found.\n"
                f"  vg_image_dir={self.vg_image_dir}\n"
                "  Download VG images and set vg_image_dir, or disable use_visual."
            )

        print(f"[VG] Building CLIP cache from {self.vg_image_dir} …")
        self.clip_extractor = CLIPExtractor()

        # Collect unique (image_id, object_id) pairs from relationships.
        with open(img_path) as f:
            img_meta: List[Dict] = json.load(f)

        with open(rel_path) as f:
            all_rels: List[Dict] = json.load(f)

        # Map object_id -> box for each image.
        obj_box_map: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        for img in all_rels:
            iid = img.get("image_id")
            for r in img.get("relationships", []):
                for role in ("subject", "object"):
                    ent = r.get(role, {})
                    oid = ent.get("object_id", -1)
                    if oid == -1:
                        continue
                    box = _xywh_to_xyxy(
                        ent.get("x", 0), ent.get("y", 0),
                        ent.get("w", 1), ent.get("h", 1),
                    )
                    key = (iid, oid)
                    if key not in obj_box_map:
                        obj_box_map[key] = box

        # Group objects by image.
        image_to_objects: Dict[int, List[Tuple[int, Tuple]]] = defaultdict(list)
        for (iid, oid), box in obj_box_map.items():
            image_to_objects[iid].append((oid, box))

        n_images = len(image_to_objects)
        print(f"[VG] Extracting CLIP embeddings for {n_images} images …")

        cache: Dict[str, torch.Tensor] = {}
        for img_idx, (iid, objects) in enumerate(image_to_objects.items()):
            img_path = os.path.join(self.vg_image_dir, f"{iid}.jpg")
            if not os.path.isfile(img_path):
                continue

            pil_img = Image.open(img_path).convert("RGB")
            boxes = [box for _, box in objects]
            embs = self.clip_extractor.extract_crops(pil_img, boxes)

            for (oid, _), emb in zip(objects, embs):
                cache[f"{iid}_obj_{oid}"] = emb

            if (img_idx + 1) % 1000 == 0:
                print(f"  [{img_idx + 1}/{n_images}] {len(cache)} embeddings cached")

        self.clip_cache = cache
        print(f"[VG] CLIP cache built: {len(cache)} embeddings")

        os.makedirs(os.path.dirname(clip_cache_path) or ".", exist_ok=True)
        with open(clip_cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[VG] CLIP cache saved to {clip_cache_path}")

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

                geo = extract_geo_features(subj_box, obj_box, img_w, img_h)

                # Cache keys for visual feature lookup.
                subj_oid = subj_d.get("object_id", -1)
                obj_oid  = obj_d.get("object_id", -1)
                subj_key = f"{iid}_obj_{subj_oid}" if subj_oid >= 0 else ""
                obj_key  = f"{iid}_obj_{obj_oid}"  if obj_oid >= 0 else ""

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
            subj_feat = self.clip_cache.get(subj_key, torch.zeros(CLIP_DIM))
            obj_feat  = self.clip_cache.get(obj_key,  torch.zeros(CLIP_DIM))
            result = result + (subj_feat.clone(), obj_feat.clone())

        return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_name(entity: Dict) -> str:
    name = entity.get("name") or ""
    if not name:
        names = entity.get("names", [])
        name = names[0] if names else ""
    return name.lower().strip()
