"""
Inference for the learned relation predictor.

Two public entry points:

1.  predict_relation(subject, obj_label, box1, box2, ...)
        Returns the most likely predicate string for a single pair,
        or None when the model is not confident enough.

2.  infer_relationships_learned(detections, ...)
        Drop-in replacement for utils.causal_caption.infer_relationships().
        Accepts the same detection-dict list and returns the same
        List[Tuple[str, str, str]] format.

Loading
-------
The module loads the trained checkpoint lazily on first call.
The model can be trained in two modes:
    - Geometry-only (clip_dim=0): uses only label embeddings + geometry
    - Visual-semantic (clip_dim>0): adds CLIP embeddings from image crops

When using a visual-semantic model, you MUST pass the image to inference.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .model import RelationMLP
from .vg_dataset import (
    ALLOWED_PREDICATES,
    Vocab,
    extract_geo_features,
    normalize_label,
    GEO_DIM,
)
from .clip_extractor import CLIPExtractor, CLIP_DIM


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_model:        Optional[RelationMLP] = None
_label_vocab:  Optional[Vocab]       = None
_pred_vocab:   Optional[Vocab]       = None
_device:       Optional[torch.device] = None
_clip_model:   Optional[CLIPExtractor] = None
_model_clip_dim: int = 0

_DEFAULT_CKPT_DIR = os.environ.get("REL_CKPT_DIR", "./checkpoints")


def load_relation_model(checkpoint_dir: str = _DEFAULT_CKPT_DIR) -> None:
    global _model, _label_vocab, _pred_vocab, _device, _model_clip_dim

    model_path = os.path.join(checkpoint_dir, "relation_mlp.pt")
    lv_path    = os.path.join(checkpoint_dir, "label_vocab.json")
    pv_path    = os.path.join(checkpoint_dir, "pred_vocab.json")

    for p, name in [(model_path, "relation_mlp.pt"),
                    (lv_path,    "label_vocab.json"),
                    (pv_path,    "pred_vocab.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[relation_prediction] Checkpoint file not found: {p}\n"
                f"  Run training first:\n"
                f"    python -m relation_prediction.train --vg-root ./data/visual_genome"
            )

    _label_vocab = Vocab.load(lv_path)
    _pred_vocab  = Vocab.load(pv_path)
    _device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(model_path, map_location=_device, weights_only=True)

    embed_dim   = state["label_emb.weight"].shape[1]
    hidden_dims = _infer_hidden_dims(state)
    clip_dim    = _infer_clip_dim(state, embed_dim)

    _model_clip_dim = clip_dim

    _model = RelationMLP(
        num_labels=len(_label_vocab),
        num_predicates=len(_pred_vocab),
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        clip_dim=clip_dim,
    )
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()

    mode_str = "visual-semantic" if clip_dim > 0 else "geometry-only"
    print(
        f"[relation_prediction] Loaded model from {checkpoint_dir} "
        f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates, "
        f"{mode_str}, input_dim={2*embed_dim + GEO_DIM + 2*clip_dim})"
    )


def _infer_hidden_dims(state: dict) -> Tuple[int, ...]:
    hidden: List[int] = []
    idx = 0
    while True:
        key = f"mlp.{idx}.weight"
        if key not in state:
            break
        hidden.append(state[key].shape[0])
        idx += 3
    return tuple(hidden[:-1])


def _infer_clip_dim(state: dict, embed_dim: int) -> int:
    """
    Infer clip_dim from the first MLP layer's input weight shape.
    clip_dim = (in_features - 2*embed_dim - GEO_DIM) // 2
    """
    first_weight = state["mlp.0.weight"]  # (out_features, in_features)
    in_features = first_weight.shape[1]
    clip_portion = in_features - 2 * embed_dim - GEO_DIM
    if clip_portion <= 0:
        return 0
    # clip_portion should be 2 * clip_dim
    if clip_portion % 2 != 0:
        print(f"[WARNING] Unexpected clip_portion={clip_portion}, treating as 0")
        return 0
    return clip_portion // 2


def _ensure_loaded() -> None:
    if _model is None:
        load_relation_model()


def _ensure_clip_model() -> None:
    global _clip_model
    if _clip_model is None and _model_clip_dim > 0:
        _clip_model = CLIPExtractor(device=_device)


# ---------------------------------------------------------------------------
# Single-pair prediction
# ---------------------------------------------------------------------------

def predict_relation(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    threshold: float = 0.15,
    image: Optional[Image.Image] = None,
) -> Optional[Tuple[str, float]]:
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return None

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]

    subj_box = (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3]))
    obj_box  = (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3]))
    geo      = extract_geo_features(subj_box, obj_box, img_w, img_h)

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        if _model_clip_dim > 0 and image is not None:
            _ensure_clip_model()
            subj_emb = _clip_model.extract_crop(image, subj_box).to(_device)
            obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device)
            subj_emb = subj_emb.unsqueeze(0)  # (1, CLIP_DIM)
            obj_emb  = obj_emb.unsqueeze(0)
            logits = _model(s, o, g, subj_emb, obj_emb)
        else:
            logits = _model(s, o, g)

        probs  = F.softmax(logits, dim=-1)
        top_prob, top_idx = probs[0].max(dim=-1)

    confidence = top_prob.item()
    if confidence < threshold:
        return None

    pred_token = _pred_vocab.token(top_idx.item())
    if pred_token in (Vocab.PAD, Vocab.UNK):
        return None
    if pred_token not in ALLOWED_PREDICATES:
        return None
    return (pred_token, confidence)


def predict_relation_topk(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    k: int = 3,
    image: Optional[Image.Image] = None,
) -> List[Tuple[str, float]]:
    _ensure_loaded()

    subj_norm = normalize_label(subject)
    obj_norm  = normalize_label(obj_label)
    if subj_norm == "UNK" or obj_norm == "UNK":
        return []

    subj_idx = _label_vocab[subj_norm]
    obj_idx  = _label_vocab[obj_norm]
    geo = extract_geo_features(
        (float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])),
        (float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])),
        img_w, img_h,
    )

    with torch.no_grad():
        s = torch.tensor([subj_idx], dtype=torch.long,    device=_device)
        o = torch.tensor([obj_idx],  dtype=torch.long,    device=_device)
        g = torch.tensor([geo],      dtype=torch.float32, device=_device)

        if _model_clip_dim > 0 and image is not None:
            _ensure_clip_model()
            subj_emb = _clip_model.extract_crop(image, subj_box).to(_device)
            obj_emb  = _clip_model.extract_crop(image, obj_box).to(_device)
            subj_emb = subj_emb.unsqueeze(0)
            obj_emb  = obj_emb.unsqueeze(0)
            logits = _model(s, o, g, subj_emb, obj_emb)
        else:
            logits = _model(s, o, g)

        probs = F.softmax(logits, dim=-1)[0]

    topk_probs, topk_idxs = probs.topk(min(k, len(probs)))
    results = []
    for prob, idx in zip(topk_probs.tolist(), topk_idxs.tolist()):
        token = _pred_vocab.token(idx)
        if token not in (Vocab.PAD, Vocab.UNK):
            results.append((token, prob))
    return results


# ---------------------------------------------------------------------------
# Drop-in replacement for utils.causal_caption.infer_relationships
# ---------------------------------------------------------------------------

Detection    = Dict
Relationship = Tuple[str, str, str]


def infer_relationships_learned(
    detections: List[Detection],
    threshold: float = 0.15,
    img_w: float = 1.0,
    img_h: float = 1.0,
    top_k: int = 2,
    image: Optional[Image.Image] = None,
) -> List[Relationship]:
    """
    Drop-in replacement for utils.causal_caption.infer_relationships().

    When using a visual-semantic model (trained with clip_dim > 0),
    pass the image to enable CLIP feature extraction from crops.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
        threshold:  Minimum model confidence to include a relationship.
        img_w:      Image width for normalisation.
        img_h:      Image height for normalisation.
        top_k:      Maximum number of relations to return (default 2).
        image:      PIL Image (required for visual-semantic model).

    Returns:
        [(subject_label, predicate, object_label), ...]
    """
    if len(detections) < 2:
        return []

    candidates: List[Tuple[Relationship, float]] = []
    seen: set = set()

    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            a = detections[i]
            b = detections[j]

            subj_norm = normalize_label(a["label"])
            obj_norm  = normalize_label(b["label"])
            if subj_norm == "UNK" or obj_norm == "UNK":
                continue

            result = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
                image=image,
            )
            if result is None:
                continue

            pred, confidence = result
            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                candidates.append((triple, confidence))

    if not candidates:
        return []

    candidates.sort(key=lambda x: -x[1])

    seen_pair: set = set()
    deduped: List[Relationship] = []
    for triple, _ in candidates:
        s, p, o = triple
        pair_key = (s, o)
        if pair_key not in seen_pair:
            seen_pair.add(pair_key)
            deduped.append(triple)

    final = deduped[:top_k]

    total_candidates = len(candidates)
    total_deduped    = len(deduped)
    discarded = [t for t in deduped if t not in final]
    print(f"[relation_prediction] infer_relationships_learned: "
          f"{total_candidates} candidates -> {total_deduped} after dedup "
          f"-> {len(final)} selected (top_k={top_k})")
    if discarded:
        print(f"[relation_prediction]   discarded: "
              f"{[f'{s} {p} {o}' for s, p, o in discarded]}")
    print(f"[relation_prediction]   final relations: "
          f"{[f'{s} {p} {o}' for s, p, o in final]}")

    return final
