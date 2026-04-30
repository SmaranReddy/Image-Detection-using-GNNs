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
Default paths match what train.py writes; override with:

    from relation_prediction.predict import load_relation_model
    load_relation_model(checkpoint_dir="path/to/ckpts")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import RelationMLP
from .vg_dataset import (
    ALLOWED_PREDICATES,
    Vocab,
    extract_geo_features,
    normalize_label,
)


# ---------------------------------------------------------------------------
# Singleton state — loaded once on first inference call
# ---------------------------------------------------------------------------

_model:       Optional[RelationMLP] = None
_label_vocab: Optional[Vocab]       = None
_pred_vocab:  Optional[Vocab]       = None
_device:      Optional[torch.device] = None

_DEFAULT_CKPT_DIR = os.environ.get("REL_CKPT_DIR", "./checkpoints")


def load_relation_model(checkpoint_dir: str = _DEFAULT_CKPT_DIR) -> None:
    """
    Explicitly load (or reload) the trained model from *checkpoint_dir*.

    Called automatically on first inference call; invoke manually to
    override the checkpoint path.
    """
    global _model, _label_vocab, _pred_vocab, _device

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

    # Reconstruct model from vocabulary sizes — no need to save hyper-params
    # separately because embed_dim and hidden_dims are baked into the state dict.
    state = torch.load(model_path, map_location=_device, weights_only=True)

    # Infer embed_dim and hidden_dims from weight shapes.
    embed_dim   = state["label_emb.weight"].shape[1]
    hidden_dims = _infer_hidden_dims(state)

    _model = RelationMLP(
        num_labels=len(_label_vocab),
        num_predicates=len(_pred_vocab),
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
    )
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()
    print(
        f"[relation_prediction] Loaded model from {checkpoint_dir} "
        f"({len(_label_vocab):,} labels, {len(_pred_vocab):,} predicates)"
    )


def _infer_hidden_dims(state: dict) -> Tuple[int, ...]:
    """Recover hidden layer widths by scanning Linear layer keys."""
    hidden: List[int] = []
    idx = 0
    while True:
        key = f"mlp.{idx}.weight"
        if key not in state:
            break
        hidden.append(state[key].shape[0])
        idx += 3  # Linear, ReLU, Dropout each occupy one index
    # The last Linear is the output layer — drop it.
    return tuple(hidden[:-1])


def _ensure_loaded() -> None:
    if _model is None:
        load_relation_model()


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
) -> Optional[str]:
    """
    Predict the most likely predicate between *subject* and *obj_label*.

    Args:
        subject:   Subject entity label (e.g. "person").
        obj_label: Object entity label  (e.g. "bicycle").
        box1:      Subject bounding box [x1, y1, x2, y2].
        box2:      Object  bounding box [x1, y1, x2, y2].
        img_w:     Image width  used for normalisation (default 1.0 = already normalised).
        img_h:     Image height used for normalisation.
        threshold: Minimum softmax probability; returns None when below.

    Returns:
        Predicate string (e.g. "on", "near", "riding") or None.
    """
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

        logits = _model(s, o, g)            # (1, num_predicates)
        probs  = F.softmax(logits, dim=-1)  # (1, num_predicates)
        top_prob, top_idx = probs[0].max(dim=-1)

    if top_prob.item() < threshold:
        return None

    pred_token = _pred_vocab.token(top_idx.item())
    if pred_token in (Vocab.PAD, Vocab.UNK):
        return None
    if pred_token not in ALLOWED_PREDICATES:
        return None
    return pred_token


def predict_relation_topk(
    subject: str,
    obj_label: str,
    box1: List[float],
    box2: List[float],
    img_w: float = 1.0,
    img_h: float = 1.0,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Return the top-k predicted (predicate, probability) pairs.

    Useful for inspection / debugging.
    """
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

        logits = _model(s, o, g)
        probs  = F.softmax(logits, dim=-1)[0]

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
) -> List[Relationship]:
    """
    Drop-in replacement for utils.causal_caption.infer_relationships().

    Iterates every ordered pair (i, j) of detections, predicts the most
    likely predicate, and returns the deduplicated list of triples.

    Args:
        detections: [{"label": str, "box": [x1,y1,x2,y2], "score": float}, ...]
                    Boxes may be pixel-space or normalised — pass img_w / img_h
                    for pixel-space inputs; leave at 1.0 for normalised.
        threshold:  Minimum model confidence to include a relationship.
        img_w:      Image width  (pixels) for normalisation; 1.0 = already normalised.
        img_h:      Image height (pixels) for normalisation.

    Returns:
        [(subject_label, predicate, object_label), ...]
    """
    if len(detections) < 2:
        return []

    relationships: List[Relationship] = []
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

            pred = predict_relation(
                subj_norm, obj_norm,
                a["box"], b["box"],
                img_w=img_w, img_h=img_h,
                threshold=threshold,
            )
            if pred is None:
                continue

            triple = (subj_norm, pred, obj_norm)
            if triple not in seen:
                seen.add(triple)
                relationships.append(triple)

    return relationships
