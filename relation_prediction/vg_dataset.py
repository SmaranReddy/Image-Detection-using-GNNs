"""
Visual Genome relationship dataset.

Expected data layout (set VG_ROOT in train.py or pass paths directly):
    <vg_root>/relationships.json
    <vg_root>/image_data.json

Both files are from the official Visual Genome download:
    https://visualgenome.org/api/v0/api_home.html
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Label normalisation — aligns VG labels to the COCO / YOLO label space
# ---------------------------------------------------------------------------

# Canonical COCO-80 labels exactly as output by Ultralytics YOLO.
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

# Maps VG surface forms → canonical COCO label.
# Keys are already lowercased so the lookup is a single dict hit.
SYNONYM_MAP: Dict[str, str] = {
    # person variants (singular and irregular plural)
    "man":          "person",
    "men":          "person",
    "woman":        "person",
    "women":        "person",
    "boy":          "person",
    "girl":         "person",
    "people":       "person",
    "child":        "person",
    "children":     "person",
    "guy":          "person",
    "lady":         "person",
    # bicycle variants
    "bike":         "bicycle",
    "cycle":        "bicycle",
    # car variants
    "vehicle":      "car",
    "automobile":   "car",
    # couch / sofa — COCO canonical is "couch"
    "sofa":         "couch",
    # tv — COCO canonical is "tv"
    "television":   "tv",
    "tv monitor":   "tv",
    "monitor":      "tv",
    # cell phone — COCO canonical is "cell phone"
    "cellphone":    "cell phone",
    "mobile":       "cell phone",
    "phone":        "cell phone",
    # misc
    "motorbike":    "motorcycle",
    "aeroplane":    "airplane",
    "aero plane":   "airplane",
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
    # on variants
    "on top of":    "on",
    "sitting on":   "on",
    "standing on":  "on",
    "lying on":     "on",
    "resting on":   "on",
    # near variants
    "next to":      "near",
    "beside":       "near",
    "close to":     "near",
    # under variants
    "underneath":   "under",
    "below":        "under",
    # riding variants
    "riding on":    "riding",
    "mounted on":   "riding",
    # holding variants
    "holding in":   "holding",
    "grasping":     "holding",
    "gripping":     "holding",
    # carrying variants
    "carrying in":  "carrying",
    "carried by":   "carrying",
}


def normalize_predicate(pred: str) -> Optional[str]:
    pred = pred.lower().strip()
    pred = PREDICATE_MAP.get(pred, pred)
    if pred not in ALLOWED_PREDICATES:
        return None
    return pred


def normalize_label(label: str) -> str:
    """
    Normalise a raw entity label to its canonical COCO form.

    Steps:
      1. Lowercase and strip whitespace.
      2. Apply SYNONYM_MAP (handles irregular plurals and synonyms).
      3. Generic plural strip: if label ends with "s" and label[:-1] is a
         COCO label, use the singular form.
      4. Return "UNK" for anything not in COCO_LABELS.

    Examples:
        normalize_label("Men")       -> "person"
        normalize_label("cars")      -> "car"
        normalize_label("dogs")      -> "dog"
        normalize_label("trees")     -> "UNK"
        normalize_label("bike")      -> "bicycle"
    """
    label = label.lower().strip()
    label = SYNONYM_MAP.get(label, label)
    if label not in COCO_LABELS and label.endswith("s") and label[:-1] in COCO_LABELS:
        label = label[:-1]
    return label if label in COCO_LABELS else "UNK"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return float(x), float(y), float(x + w), float(y + h)


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def extract_geo_features(
    subj_box: Tuple[float, float, float, float],
    obj_box: Tuple[float, float, float, float],
    img_w: float,
    img_h: float,
) -> List[float]:
    """
    5-dim geometric feature vector between two boxes.

    Features:
        dx       — normalised horizontal centre offset (obj - subj)
        dy       — normalised vertical centre offset   (obj - subj)
        log_wr   — log(obj_w / subj_w)  size ratio, width
        log_hr   — log(obj_h / subj_h)  size ratio, height
        iou      — intersection-over-union
    """
    sx1, sy1, sx2, sy2 = subj_box
    ox1, oy1, ox2, oy2 = obj_box

    scx = (sx1 + sx2) / 2.0
    scy = (sy1 + sy2) / 2.0
    ocx = (ox1 + ox2) / 2.0
    ocy = (oy1 + oy2) / 2.0

    sw = max(sx2 - sx1, 1.0)
    sh = max(sy2 - sy1, 1.0)
    ow = max(ox2 - ox1, 1.0)
    oh = max(oy2 - oy1, 1.0)

    denom_w = max(img_w, 1.0)
    denom_h = max(img_h, 1.0)

    dx     = (ocx - scx) / denom_w
    dy     = (ocy - scy) / denom_h
    log_wr = float(np.log(ow / sw))
    log_hr = float(np.log(oh / sh))
    iou    = compute_iou(subj_box, obj_box)

    return [dx, dy, log_wr, log_hr, iou]


GEO_DIM = 5


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Bidirectional token ↔ index mapping with PAD and UNK."""

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
    """
    Torch dataset of Visual Genome relationship triples.

    Each __getitem__ returns:
        subj_idx  : LongTensor scalar — subject label index
        obj_idx   : LongTensor scalar — object label index
        geo       : FloatTensor (GEO_DIM,) — geometric features
        pred_idx  : LongTensor scalar — predicate (target) index

    Args:
        relationships_json: Path to VG relationships.json
        image_data_json:    Path to VG image_data.json
        label_vocab:        Pre-built label Vocab; if None, one is built from data.
        pred_vocab:         Pre-built predicate Vocab; if None, one is built from data.
        min_pred_count:     Drop predicates that appear fewer than this many times.
        max_samples:        Cap total samples (useful for quick experiments).
    """

    def __init__(
        self,
        relationships_json: str,
        image_data_json: str,
        label_vocab: Optional[Vocab] = None,
        pred_vocab: Optional[Vocab] = None,
        min_pred_count: int = 50,
        max_samples: Optional[int] = None,
    ) -> None:
        self.label_vocab = label_vocab if label_vocab is not None else Vocab()
        self.pred_vocab  = pred_vocab  if pred_vocab  is not None else Vocab()
        self._build_vocab = label_vocab is None

        self.samples = self._load(
            relationships_json,
            image_data_json,
            min_pred_count=min_pred_count,
            max_samples=max_samples,
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
        print("\nTop predicates by frequency:")
        for pred, count in pred_counter.most_common():
            print(f"  {pred}: {count}")
        # --- end dataset inspection logging ---

    # ------------------------------------------------------------------

    def _load(
        self,
        rel_path: str,
        img_path: str,
        min_pred_count: int,
        max_samples: Optional[int],
    ) -> List[Tuple[int, int, List[float], int]]:
        with open(img_path) as f:
            img_meta: List[Dict] = json.load(f)
        img_size: Dict[int, Tuple[int, int]] = {
            m["image_id"]: (int(m["width"]), int(m["height"]))
            for m in img_meta
        }

        with open(rel_path) as f:
            all_rels: List[Dict] = json.load(f)

        # Pre-populate pred_vocab with every allowed predicate for consistent indices.
        if self._build_vocab:
            for pred in sorted(ALLOWED_PREDICATES):
                self.pred_vocab.add(pred)

        # Parse samples — keep only normalised predicates in ALLOWED_PREDICATES.
        raw: List[Tuple[str, str, List[float], str]] = []
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

                # VG boxes are (x, y, w, h) in absolute pixel coordinates.
                subj_box = _xywh_to_xyxy(
                    subj_d.get("x", 0), subj_d.get("y", 0),
                    subj_d.get("w", 1), subj_d.get("h", 1),
                )
                obj_box = _xywh_to_xyxy(
                    obj_d.get("x", 0), obj_d.get("y", 0),
                    obj_d.get("w", 1), obj_d.get("h", 1),
                )

                geo = extract_geo_features(subj_box, obj_box, img_w, img_h)
                raw.append((subj_name, obj_name, geo, pred))

        kept_before_cap = len(raw)

        if self._build_vocab:
            for subj_name, obj_name, _, _ in raw:
                self.label_vocab.add(subj_name)
                self.label_vocab.add(obj_name)

        if max_samples is not None:
            raw = raw[:max_samples]

        self._load_stats = {
            "total_raw":       total_raw,
            "kept_before_cap": kept_before_cap,
            "pred_counter":    Counter(r[3] for r in raw),
        }

        # Convert to integer indices now that vocabs are finalised.
        samples: List[Tuple[int, int, List[float], int]] = []
        for subj_name, obj_name, geo, pred in raw:
            samples.append((
                self.label_vocab[subj_name],
                self.label_vocab[obj_name],
                geo,
                self.pred_vocab[pred],
            ))
        return samples

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        subj_idx, obj_idx, geo, pred_idx = self.samples[idx]
        return (
            torch.tensor(subj_idx, dtype=torch.long),
            torch.tensor(obj_idx,  dtype=torch.long),
            torch.tensor(geo,      dtype=torch.float32),
            torch.tensor(pred_idx, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_name(entity: Dict) -> str:
    name = entity.get("name") or ""
    if not name:
        names = entity.get("names", [])
        name = names[0] if names else ""
    return name.lower().strip()
