"""
analyze_relation_failures.py -- Deep diagnostic analysis of relation prediction failures.

Quantifies exactly:
    1. Which predicates dominate incorrectly
    2. Which object pairs trigger nonsense predictions
    3. Whether geometry overpowers semantics
    4. Whether semantic predicates are undertrained
    5. Whether priors mask weak logits
    6. Which feature group drives predictions

Usage:
    python analyze_relation_failures.py
    python analyze_relation_failures.py --checkpoint-dir ./checkpoints
    python analyze_relation_failures.py --num-samples 500 --quick
    python analyze_relation_failures.py --output ./analysis_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from relation_prediction.model import RelationMLP
from relation_prediction.relation_transformer import RelationTransformer
from relation_prediction.vg_dataset import (
    ALLOWED_PREDICATES,
    GEO_DIM,
    POSE_FEATURE_DIM,
    UNION_FEATURE_DIM,
    VGRelationshipDataset,
    Vocab,
    extract_geo_features,
    normalize_label,
)
from relation_prediction.clip_extractor import CLIPExtractor, CLIP_DIM
from relation_prediction.pose_extractor import PoseExtractor

# ---------------------------------------------------------------------------
# Predicate category definitions (mirrors predict.py)
# ---------------------------------------------------------------------------
SEMANTIC_PREDS = frozenset({
    "riding", "holding", "carrying", "wearing", "looking at",
    "sitting on", "standing on",
})

WEAK_SPATIAL = frozenset({
    "under", "above", "over", "inside", "next to", "near",
    "attached to", "behind", "in front of", "covering",
})

NEUTRAL_SPATIAL = frozenset({
    "on", "in",
})

ALL_PREDICATES_SORTED = [
    "above", "attached to", "behind", "carrying", "covering",
    "holding", "in", "in front of", "inside", "looking at",
    "near", "next to", "on", "over", "riding",
    "sitting on", "standing on", "under", "wearing",
]

# Semantic prior rules (mirrors predict.py)
ANIMATE = frozenset({
    "person", "dog", "horse", "cat", "bird",
    "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
})
WEARABLE = frozenset({"backpack", "handbag", "tie", "suitcase"})
RIDEABLE = frozenset({
    "bicycle", "horse", "motorcycle", "skateboard", "surfboard",
    "skis", "snowboard",
})
HANDHELD = frozenset({
    "cell phone", "umbrella", "bottle", "cup", "book",
    "fork", "knife", "spoon", "bowl", "frisbee",
    "kite", "baseball bat", "baseball glove", "tennis racket",
    "remote", "keyboard", "mouse", "scissors", "toothbrush",
    "wine glass", "hot dog", "apple", "banana", "orange",
    "sandwich", "donut", "cake", "pizza", "carrot", "broccoli",
})
FURNITURE = frozenset({
    "chair", "couch", "bench", "bed", "dining table", "toilet",
})

_OBJECT_CATEGORIES = {
    "animate": ANIMATE, "wearable": WEARABLE,
    "rideable": RIDEABLE, "handheld": HANDHELD, "furniture": FURNITURE,
}

_PREDICATE_PRIORS = {
    "riding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"rideable"},
        "unsuitable_object_cats": {"furniture", "wearable", "handheld"},
        "animate_subject_bonus": 0.12, "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10, "unsuitable_object_penalty": -0.20,
    },
    "wearing": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"wearable"},
        "unsuitable_object_cats": {"furniture", "rideable", "animate"},
        "animate_subject_bonus": 0.12, "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.10, "unsuitable_object_penalty": -0.25,
    },
    "holding": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld"},
        "unsuitable_object_cats": {"furniture", "animate"},
        "animate_subject_bonus": 0.12, "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08, "unsuitable_object_penalty": -0.20,
    },
    "carrying": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"handheld", "wearable"},
        "unsuitable_object_cats": {"furniture"},
        "animate_subject_bonus": 0.12, "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08, "unsuitable_object_penalty": -0.15,
    },
    "looking at": {
        "requires_animate_subject": True,
        "preferred_object_cats": set(),
        "unsuitable_object_cats": set(),
        "animate_subject_bonus": 0.10, "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.0, "unsuitable_object_penalty": 0.0,
    },
    "sitting on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture", "rideable"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.12, "inanimate_subject_penalty": -0.35,
        "preferred_object_bonus": 0.08, "unsuitable_object_penalty": -0.15,
    },
    "standing on": {
        "requires_animate_subject": True,
        "preferred_object_cats": {"furniture"},
        "unsuitable_object_cats": {"handheld", "wearable"},
        "animate_subject_bonus": 0.10, "inanimate_subject_penalty": -0.30,
        "preferred_object_bonus": 0.06, "unsuitable_object_penalty": -0.12,
    },
}


def get_categories(label: str) -> List[str]:
    cats = []
    for cat_name, cat_set in _OBJECT_CATEGORIES.items():
        if label in cat_set:
            cats.append(cat_name)
    return cats


def compute_prior_adjustment(subject: str, predicate: str, object: str) -> float:
    prior = _PREDICATE_PRIORS.get(predicate)
    subj_cats = get_categories(subject)
    obj_cats = get_categories(object)
    bonus, penalty = 0.0, 0.0
    if prior is not None:
        if prior["requires_animate_subject"]:
            if "animate" in subj_cats:
                bonus += prior["animate_subject_bonus"]
            else:
                penalty += prior["inanimate_subject_penalty"]
        if prior["preferred_object_cats"]:
            if any(c in obj_cats for c in prior["preferred_object_cats"]):
                bonus += prior["preferred_object_bonus"]
        if prior["unsuitable_object_cats"]:
            if any(c in obj_cats for c in prior["unsuitable_object_cats"]):
                penalty += prior["unsuitable_object_penalty"]
    return bonus + penalty


# ---------------------------------------------------------------------------
# Feature group norms extraction
# ---------------------------------------------------------------------------

def get_feature_group_norms(model: nn.Module) -> Dict[str, float]:
    if isinstance(model, RelationTransformer):
        norms = {}
        norms["subj_label_emb"] = model.subj_label_proj.weight.norm().item()
        norms["obj_label_emb"] = model.obj_label_proj.weight.norm().item()
        norms["geometry"] = model.geo_proj.weight.norm().item()
        if hasattr(model, 'subj_clip_proj'):
            norms["subj_clip"] = model.subj_clip_proj.weight.norm().item()
            norms["obj_clip"] = model.obj_clip_proj.weight.norm().item()
        if hasattr(model, 'union_proj'):
            norms["union_clip"] = model.union_proj.weight.norm().item()
        if hasattr(model, 'pose_proj'):
            norms["pose"] = model.pose_proj.weight.norm().item()
        return norms

    first_weight = model.mlp[0].weight
    embed_dim = model.label_emb.weight.shape[1]
    clip_dim = model.clip_dim
    union_dim = model.union_dim
    pose_dim = model.pose_dim

    groups = {
        "subj_label_emb": (0, embed_dim),
        "obj_label_emb": (embed_dim, 2 * embed_dim),
        "geometry": (2 * embed_dim, 2 * embed_dim + GEO_DIM),
    }
    offset = 2 * embed_dim + GEO_DIM
    if clip_dim > 0:
        groups["subj_clip"] = (offset, offset + clip_dim)
        offset += clip_dim
        groups["obj_clip"] = (offset, offset + clip_dim)
        offset += clip_dim
    if union_dim > 0:
        groups["union_clip"] = (offset, offset + union_dim)
        offset += union_dim
    if pose_dim > 0:
        groups["pose"] = (offset, offset + pose_dim)

    norms = {}
    for name, (start, end) in groups.items():
        gw = first_weight[:, start:end]
        norms[name] = gw.norm().item()
    return norms


def compute_feature_contribution_percentages(norms: Dict[str, float]) -> Dict[str, float]:
    total = sum(norms.values()) or 1.0
    geo = norms.get("geometry", 0)
    subj_clip = norms.get("subj_clip", 0)
    obj_clip = norms.get("obj_clip", 0)
    union_clip = norms.get("union_clip", 0)
    pose = norms.get("pose", 0)
    subj_label = norms.get("subj_label_emb", 0)
    obj_label = norms.get("obj_label_emb", 0)

    return {
        "geometry_pct": round(geo / total * 100, 2),
        "clip_total_pct": round((subj_clip + obj_clip) / total * 100, 2),
        "subj_clip_pct": round(subj_clip / total * 100, 2),
        "obj_clip_pct": round(obj_clip / total * 100, 2),
        "union_clip_pct": round(union_clip / total * 100, 2),
        "pose_pct": round(pose / total * 100, 2),
        "label_emb_pct": round((subj_label + obj_label) / total * 100, 2),
        "subj_label_pct": round(subj_label / total * 100, 2),
        "obj_label_pct": round(obj_label / total * 100, 2),
    }


# ---------------------------------------------------------------------------
# Ablation analysis
# ---------------------------------------------------------------------------

def analyze_ablation(
    model: nn.Module,
    subj_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    geo: torch.Tensor,
    subj_feat: Optional[torch.Tensor],
    obj_feat: Optional[torch.Tensor],
    union_feat: Optional[torch.Tensor],
    pose_feat: Optional[torch.Tensor],
    pred_vocab: Vocab,
) -> Dict:
    """Measure prediction changes when feature groups are zeroed."""
    with torch.no_grad():
        def _forward(sf, of, uf, pf, g):
            return model(subj_idx, obj_idx, g, subj_feat=sf, obj_feat=of, union_feat=uf, pose_feat=pf)

        full_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, geo)
        full_pred_idx = full_logits[0].argmax(dim=-1).item()
        full_pred = pred_vocab.token(full_pred_idx)

        result = {"full_prediction": full_pred}

        # Ablate CLIP
        if subj_feat is not None and obj_feat is not None:
            zeros_c = torch.zeros_like(subj_feat)
            nc_logits = _forward(zeros_c, zeros_c, union_feat, pose_feat, geo)
            result["ablate_clip_pred"] = pred_vocab.token(nc_logits[0].argmax(dim=-1).item())
            result["clip_logit_diff"] = round((full_logits[0] - nc_logits[0]).norm().item(), 4)

        # Ablate union
        if union_feat is not None and model.union_dim > 0:
            zeros_u = torch.zeros_like(union_feat)
            nu_logits = _forward(subj_feat, obj_feat, zeros_u, pose_feat, geo)
            result["ablate_union_pred"] = pred_vocab.token(nu_logits[0].argmax(dim=-1).item())
            result["union_logit_diff"] = round((full_logits[0] - nu_logits[0]).norm().item(), 4)

        # Ablate pose
        if pose_feat is not None and model.pose_dim > 0:
            zeros_p = torch.zeros_like(pose_feat)
            np_logits = _forward(subj_feat, obj_feat, union_feat, zeros_p, geo)
            result["ablate_pose_pred"] = pred_vocab.token(np_logits[0].argmax(dim=-1).item())
            result["pose_logit_diff"] = round((full_logits[0] - np_logits[0]).norm().item(), 4)

        # Ablate geometry
        zero_geo = torch.zeros_like(geo)
        ng_logits = _forward(subj_feat, obj_feat, union_feat, pose_feat, zero_geo)
        result["ablate_geo_pred"] = pred_vocab.token(ng_logits[0].argmax(dim=-1).item())
        result["geo_logit_diff"] = round((full_logits[0] - ng_logits[0]).norm().item(), 4)

        return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_relation_failures(
    checkpoint_dir: str = "./checkpoints",
    vg_root: str = "./data/visual_genome",
    output_dir: str = "./analysis_results",
    num_samples: Optional[int] = None,
    val_fraction: float = 0.1,
    temperature: float = 2.0,
    batch_size: int = 256,
    quick: bool = False,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Load model -------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    model_path = os.path.join(checkpoint_dir, "relation_mlp.pt")
    lv_path = os.path.join(checkpoint_dir, "label_vocab.json")
    pv_path = os.path.join(checkpoint_dir, "pred_vocab.json")

    label_vocab = Vocab.load(lv_path)
    pred_vocab = Vocab.load(pv_path)

    raw_data = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(raw_data, dict) and "model_state_dict" in raw_data:
        state = raw_data["model_state_dict"]
        config = raw_data.get("model_config", {})
        model_type = config.get("model_type", "mlp")
        pose_dim = config.get("pose_dim", 0)
        union_dim = config.get("union_dim", 0)
        clip_dim = config.get("clip_dim", 0)
        embed_dim = config.get("embed_dim", state["label_emb.weight"].shape[1])

        if model_type == "transformer":
            d_model = config.get("d_model", 256)
            model = RelationTransformer(
                num_labels=len(label_vocab),
                num_predicates=len(pred_vocab),
                d_model=d_model,
                embed_dim=embed_dim,
                clip_dim=clip_dim,
                pose_dim=pose_dim,
                union_dim=union_dim,
            )
            hidden_dims_str = f"d_model={d_model}"
        else:
            # Infer hidden dims from state dict weights
            hidden_list: List[int] = []
            idx = 0
            while True:
                key = f"mlp.{idx}.weight"
                if key not in state:
                    break
                hidden_list.append(state[key].shape[0])
                idx += 3
            hidden_dims = tuple(hidden_list[:-1])
            hidden_dims_str = str(hidden_dims)
            model = RelationMLP(
                num_labels=len(label_vocab),
                num_predicates=len(pred_vocab),
                embed_dim=embed_dim,
                hidden_dims=hidden_dims,
                clip_dim=clip_dim,
                pose_dim=pose_dim,
                union_dim=union_dim,
            )
    else:
        raise ValueError("Expected saved dict with model_state_dict + model_config")

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Labels: {len(label_vocab)}, Predicates: {len(pred_vocab)}")
    print(f"Model type: {model_type}")
    print(f"Mode: clip_dim={clip_dim}, pose_dim={pose_dim}, union_dim={union_dim}")
    print(f"Dims: {hidden_dims_str}")

    # -- 2. Feature group norms (from projection weights) ------------
    print("\n" + "=" * 70)
    print(f"FEATURE GROUP NORM ANALYSIS ({'Projection Weights' if model_type == 'transformer' else 'First Layer Weights'})")
    print("=" * 70)
    norms = get_feature_group_norms(model)
    contrib = compute_feature_contribution_percentages(norms)
    total_norm = sum(norms.values())
    for name, norm_val in sorted(norms.items(), key=lambda x: -x[1]):
        pct = norm_val / total_norm * 100
        print(f"  {name:20s}: {norm_val:8.4f}  ({pct:5.1f}%)")
    print(f"  {'-' * 40}")
    print(f"  Geometry:           {contrib['geometry_pct']:5.1f}%")
    print(f"  CLIP (subj+obj):    {contrib['clip_total_pct']:5.1f}%")
    print(f"  Union CLIP:         {contrib['union_clip_pct']:5.1f}%")
    print(f"  Pose:               {contrib['pose_pct']:5.1f}%")
    print(f"  Label embeddings:   {contrib['label_emb_pct']:5.1f}%")

    # -- 3. Load dataset ----------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    clip_cache_path = os.path.join(vg_root, "clip_cache_proper.pt")
    if not os.path.exists(clip_cache_path):
        clip_cache_path = None

    full_ds = VGRelationshipDataset(
        relationships_json=os.path.join(vg_root, "relationships.json"),
        image_data_json=os.path.join(vg_root, "image_data.json"),
        vg_image_dir=os.path.join(vg_root, "images"),
        min_pred_count=50,
        max_samples=num_samples,
        use_visual=(clip_dim > 0),
        clip_cache_path=clip_cache_path,
        require_visual=(clip_dim > 0),
        use_pose=(pose_dim > 0),
        use_union=(union_dim > 0),
    )
    print(f"Total dataset: {len(full_ds)} samples")

    # Create validation split (matches training split)
    n_val = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    if quick:
        val_ds.indices = val_ds.indices[:200]

    # -- 4. Run model on validation set ------------------------------
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE ON VALIDATION SET")
    print("=" * 70)

    def collate_fn(batch):
        subj_idxs, obj_idxs, geos, preds = [], [], [], []
        subj_feats, obj_feats, union_feats, pose_feats = [], [], [], []
        for item in batch:
            subj_idxs.append(item[0])
            obj_idxs.append(item[1])
            geos.append(item[2])
            preds.append(item[3])
            idx = 4
            if clip_dim > 0:
                subj_feats.append(item[idx])
                obj_feats.append(item[idx + 1])
                idx += 2
                if union_dim > 0:
                    union_feats.append(item[idx])
                    idx += 1
                if pose_dim > 0:
                    pose_feats.append(item[idx])
                    idx += 1
        batch_out = {
            "subj_idx": torch.stack(subj_idxs).to(device),
            "obj_idx": torch.stack(obj_idxs).to(device),
            "geo": torch.stack(geos).to(device),
            "pred_idx": torch.stack(preds).to(device),
        }
        if clip_dim > 0:
            batch_out["subj_feat"] = torch.stack(subj_feats).to(device)
            batch_out["obj_feat"] = torch.stack(obj_feats).to(device)
        if union_dim > 0 and union_feats:
            batch_out["union_feat"] = torch.stack(union_feats).to(device)
        if pose_dim > 0 and pose_feats:
            batch_out["pose_feat"] = torch.stack(pose_feats).to(device)
        return batch_out

    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # -- 4a. Collect per-sample data ----------------------------------
    all_results = []
    all_confusion = []  # (true_pred, pred_pred)
    pred_correct = Counter()
    pred_total = Counter()
    pred_raw_logits = defaultdict(list)
    pred_calibrated_probs = defaultdict(list)
    pred_adjusted_scores = defaultdict(list)
    pred_prior_adjustments = defaultdict(list)

    confusion_matrix = defaultdict(lambda: defaultdict(int))
    # Track raw confusion (before priors)
    confusion_raw = defaultdict(lambda: defaultdict(int))

    object_pair_failures = defaultdict(lambda: {
        "count": 0, "confidences": [], "raw_logits_top": [], "adjusted_logits_top": [],
        "true_predicates": Counter(), "predicted_predicates": Counter(),
    })

    total = len(val_loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{total}]")

            subj_idx = batch["subj_idx"]
            obj_idx = batch["obj_idx"]
            geo = batch["geo"]
            gt_pred_idx = batch["pred_idx"]

            # Forward pass
            kwargs = {}
            if clip_dim > 0:
                kwargs["subj_feat"] = batch["subj_feat"]
                kwargs["obj_feat"] = batch["obj_feat"]
            if union_dim > 0:
                kwargs["union_feat"] = batch["union_feat"]
            if pose_dim > 0:
                kwargs["pose_feat"] = batch["pose_feat"]

            logits = model(subj_idx, obj_idx, geo, **kwargs)

            # Temperature-calibrated softmax
            probs = F.softmax(logits / temperature, dim=-1)
            raw_probs = F.softmax(logits, dim=-1)  # T=1.0

            # Get top predictions
            pred_idxs = logits.argmax(dim=-1)
            top_probs, top_indices = probs.topk(min(5, probs.size(-1)), dim=-1)

            # Per-sample processing
            for i in range(subj_idx.size(0)):
                gt_idx = gt_pred_idx[i].item()
                gt_pred = pred_vocab.token(gt_idx)
                pred_idx_i = pred_idxs[i].item()
                pred_pred = pred_vocab.token(pred_idx_i)

                # Raw logits for all predicates
                logits_i = logits[i].cpu()
                probs_i = probs[i].cpu()
                raw_probs_i = raw_probs[i].cpu()

                # Get subject/object labels
                s_idx = subj_idx[i].item()
                o_idx = obj_idx[i].item()
                subj_label = label_vocab.token(s_idx)
                obj_label = label_vocab.token(o_idx)
                geo_i = geo[i].cpu().tolist()

                # Compute prior adjustments for each predicate
                prior_adj = {}
                for p_name in ALLOWED_PREDICATES:
                    p_vocab_idx = pred_vocab[p_name]
                    prior_adj[p_name] = compute_prior_adjustment(subj_label, p_name, obj_label)

                # Final scores (calibrated + prior)
                final_scores = {}
                for p_name in ALLOWED_PREDICATES:
                    p_vocab_idx = pred_vocab[p_name]
                    calib = probs_i[p_vocab_idx].item()
                    prior = prior_adj.get(p_name, 0.0)
                    final_scores[p_name] = calib + prior

                # Get top after adjustment
                best_after_prior = max(final_scores, key=final_scores.get)
                best_before_prior = pred_vocab.token(logits_i.argmax().item())
                best_raw_prob = pred_vocab.token(raw_probs_i.argmax().item())

                # Collect confusion (true vs predicted BEFORE priors)
                confusion_raw[gt_pred][best_before_prior] += 1
                # Collect confusion (true vs predicted AFTER priors)
                confusion_matrix[gt_pred][best_after_prior] += 1

                # Top-5 logits for dominance analysis
                logit_vals = [(pred_vocab.token(j), logits_i[j].item()) for j in range(len(pred_vocab))
                              if pred_vocab.token(j) not in (Vocab.PAD, Vocab.UNK) and pred_vocab.token(j) in ALLOWED_PREDICATES]
                logit_vals.sort(key=lambda x: -x[1])
                top5_logits = logit_vals[:5]

                pred_total[gt_pred] += 1
                if gt_pred == pred_pred:
                    pred_correct[gt_pred] += 1
                elif gt_pred == best_before_prior:
                    pass  # correct before priors

                pred_raw_logits[gt_pred].append(logits_i[gt_idx].item())
                pred_calibrated_probs[gt_pred].append(probs_i[gt_idx].item())

                # Track prior adjustments
                prior_for_gt = prior_adj.get(gt_pred, 0.0)
                pred_prior_adjustments[gt_pred].append(prior_for_gt)
                pred_adjusted_scores[gt_pred].append(final_scores.get(gt_pred, probs_i[gt_idx].item()))

                # Object-pair failure tracking
                pair_key = f"{subj_label} -- {obj_label}"
                object_pair_failures[pair_key]["count"] += 1
                if gt_pred != best_after_prior:
                    object_pair_failures[pair_key]["confidences"].append(probs_i[pred_idx_i].item())
                    object_pair_failures[pair_key]["raw_logits_top"].append(top5_logits[0][1] if top5_logits else 0)
                    object_pair_failures[pair_key]["true_predicates"][gt_pred] += 1
                    object_pair_failures[pair_key]["predicted_predicates"][best_after_prior] += 1

                # Feature ablation for a subset (every 20th sample)
                do_ablation = (batch_idx * subj_idx.size(0) + i) % 20 == 0

                record = {
                    "subj_label": subj_label,
                    "obj_label": obj_label,
                    "geo": geo_i,
                    "gt_predicate": gt_pred,
                    "pred_predicate": pred_pred,
                    "best_before_prior": best_before_prior,
                    "best_after_prior": best_after_prior,
                    "best_raw_prob": best_raw_prob,
                    "is_correct": gt_pred == pred_pred,
                    "is_correct_before_prior": gt_pred == best_before_prior,
                    "is_correct_after_prior": gt_pred == best_after_prior,
                    "gt_logit": logits_i[gt_idx].item(),
                    "gt_calibrated_prob": probs_i[gt_idx].item(),
                    "gt_prior_adjustment": prior_for_gt,
                    "gt_final_score": final_scores.get(gt_pred, probs_i[gt_idx].item()),
                    "top5_logits": top5_logits,
                    "top5_probs": [(pred_vocab.token(top_indices[i][k].item()), top_probs[i][k].item())
                                   for k in range(min(5, probs.size(-1)))],
                }

                if do_ablation:
                    ablation_kwargs = {}
                    if clip_dim > 0:
                        ablation_kwargs["subj_feat"] = batch["subj_feat"][i:i+1]
                        ablation_kwargs["obj_feat"] = batch["obj_feat"][i:i+1]
                    if union_dim > 0:
                        ablation_kwargs["union_feat"] = batch["union_feat"][i:i+1]
                    if pose_dim > 0:
                        ablation_kwargs["pose_feat"] = batch["pose_feat"][i:i+1]

                    record["ablation"] = analyze_ablation(
                        model,
                        subj_idx[i:i+1], obj_idx[i:i+1],
                        geo[i:i+1],
                        **ablation_kwargs,
                        pred_vocab=pred_vocab,
                    )

                all_results.append(record)

    n_eval = len(all_results)
    n_correct = sum(1 for r in all_results if r["is_correct"])
    n_correct_bp = sum(1 for r in all_results if r["is_correct_before_prior"])
    n_correct_ap = sum(1 for r in all_results if r["is_correct_after_prior"])
    print(f"\nEvaluated: {n_eval}")
    print(f"  Accuracy (final):           {n_correct}/{n_eval} = {n_correct/max(n_eval,1)*100:.2f}%")
    print(f"  Accuracy (before prior):    {n_correct_bp}/{n_eval} = {n_correct_bp/max(n_eval,1)*100:.2f}%")
    print(f"  Accuracy (after prior):     {n_correct_ap}/{n_eval} = {n_correct_ap/max(n_eval,1)*100:.2f}%")

    # -- 5. BUILD CONFUSION MATRIX -------------------------------------
    print("\n" + "=" * 70)
    print("PREDICATE CONFUSION MATRIX (After Priors)")
    print("=" * 70)

    all_preds_sorted = [p for p in ALL_PREDICATES_SORTED if p in pred_total]
    print(f"  {'':>20}", end="")
    for p in all_preds_sorted:
        print(f" {p:>12}", end="")
    print()

    for gt in all_preds_sorted:
        if gt not in confusion_matrix:
            continue
        print(f"  {gt:>20}", end="")
        total_gt = pred_total.get(gt, 0)
        for pred in all_preds_sorted:
            count = confusion_matrix[gt].get(pred, 0)
            pct = count / max(total_gt, 1) * 100
            print(f" {count:>6d}/{pct:>4.0f}%", end="")
        print()

    # -- 5a. Top confusion pairs --------------------------------------
    print("\n  TOP CONFUSION PAIRS (After Priors):")
    confusion_pairs = []
    for gt in confusion_matrix:
        for pred in confusion_matrix[gt]:
            if gt != pred and confusion_matrix[gt][pred] > 0:
                confusion_pairs.append((gt, pred, confusion_matrix[gt][pred]))
    confusion_pairs.sort(key=lambda x: -x[2])
    for gt, pred, count in confusion_pairs[:30]:
        pct = count / max(pred_total.get(gt, 1), 1) * 100
        print(f"    {gt:>15} -> {pred:15s}: {count:>5d} ({pct:.1f}%)")

    # -- 5b. Confusion BEFORE priors ----------------------------------
    print("\n  TOP CONFUSION PAIRS (Before Priors -- RAW LOGITS):")
    confusion_pairs_raw = []
    for gt in confusion_raw:
        for pred in confusion_raw[gt]:
            if gt != pred and confusion_raw[gt][pred] > 0:
                confusion_pairs_raw.append((gt, pred, confusion_raw[gt][pred]))
    confusion_pairs_raw.sort(key=lambda x: -x[2])
    for gt, pred, count in confusion_pairs_raw[:30]:
        pct = count / max(pred_total.get(gt, 1), 1) * 100
        print(f"    {gt:>15} -> {pred:15s}: {count:>5d} ({pct:.1f}%)")

    # -- 6. PREDICATE-WISE METRICS -------------------------------------
    print("\n" + "=" * 70)
    print("PREDICATE-WISE PERFORMANCE")
    print("=" * 70)
    print(f"  {'Predicate':<20} {'Total':>7} {'Correct':>8} {'Acc':>6} {'Mean Logit':>11} {'Mean Prob':>10} {'Mean Prior':>10} {'Category':>14}")
    print(f"  {'-' * 86}")

    pred_metrics = {}
    for p in all_preds_sorted:
        t = pred_total.get(p, 0)
        c = pred_correct.get(p, 0)
        acc = c / max(t, 1) * 100
        mean_logit = np.mean(pred_raw_logits.get(p, [0]))
        mean_prob = np.mean(pred_calibrated_probs.get(p, [0]))
        mean_prior = np.mean(pred_prior_adjustments.get(p, [0]))

        if p in SEMANTIC_PREDS:
            cat = "semantic"
        elif p in WEAK_SPATIAL:
            cat = "weak_spatial"
        elif p in NEUTRAL_SPATIAL:
            cat = "neutral_spatial"
        else:
            cat = "other"

        pred_metrics[p] = {
            "total": t, "correct": c, "accuracy": round(acc, 2),
            "mean_logit": round(mean_logit, 4), "mean_prob": round(mean_prob, 4),
            "mean_prior": round(mean_prior, 4), "category": cat,
        }
        print(f"  {p:<20} {t:>7} {c:>8} {acc:>5.1f}% {mean_logit:>10.4f} {mean_prob:>9.4f} {mean_prior:>+9.4f} {cat:>14}")

    # -- 7. OBJECT-PAIR FAILURE ANALYSIS ------------------------------
    print("\n" + "=" * 70)
    print("OBJECT-PAIR FAILURE ANALYSIS")
    print("=" * 70)

    # Find pairs with most failures
    pair_failure_rate = {}
    for pair, data in object_pair_failures.items():
        failures = data["count"]
        true_pred_total = sum(data["true_predicates"].values())
        failure_rate = true_pred_total / max(failures, 1)
        pair_failure_rate[pair] = {
            "total_pairs": failures,
            "failure_count": true_pred_total,
            "failure_rate": round(true_pred_total / max(failures, 1) * 100, 1),
            "top_true": data["true_predicates"].most_common(3),
            "top_pred": data["predicted_predicates"].most_common(3),
            "mean_conf": round(np.mean(data["confidences"]), 4) if data["confidences"] else 0,
            "mean_raw_logit": round(np.mean(data["raw_logits_top"]), 4) if data["raw_logits_top"] else 0,
        }

    sorted_pairs = sorted(pair_failure_rate.items(), key=lambda x: -x[1]["failure_count"])
    print(f"\n  Worst object pairs (most failures):")
    print(f"  {'Subject -> Object':<35} {'Failures':>9} {'Rate':>6} {'Top GT':>20} {'Top Pred':>20} {'Mean Conf':>10}")
    print(f"  {'-' * 100}")
    for pair_str, info in sorted_pairs[:20]:
        subj, obj = pair_str.split(" -- ")
        pair_compact = f"{subj} -> {obj}"
        top_gt = info["top_true"][0][0] if info["top_true"] else "-"
        top_pred = info["top_pred"][0][0] if info["top_pred"] else "-"
        print(f"  {pair_compact:<35} {info['failure_count']:>9} {info['failure_rate']:>5.1f}% {top_gt:>20} {top_pred:>20} {info['mean_conf']:>9.4f}")

    # -- 7a. Specific nonsense pattern search -------------------------
    print(f"\n  Specific nonsense patterns:")
    nonsense_patterns = [
        ("chair", "holding", "bird"),
        ("truck", "holding", "bus"),
        ("elephant", "holding", "elephant"),
        ("boat", "holding", "car"),
    ]
    for subj, pred, obj in nonsense_patterns:
        pair_key = f"{subj} -- {obj}"
        if pair_key in pair_failure_rate:
            info = pair_failure_rate[pair_key]
            print(f"    {subj} -> {obj}:")
            print(f"      Total failures: {info['failure_count']}")
            print(f"      Top GT predicates: {info['top_true']}")
            print(f"      Top predicted: {info['top_pred']}")
            print(f"      Mean raw logit: {info['mean_raw_logit']:.4f}")
        else:
            print(f"    {subj} -> {obj}: not found in validation set")

    # -- 8. SEMANTIC VS SPATIAL BREAKDOWN -----------------------------
    print("\n" + "=" * 70)
    print("SEMANTIC vs SPATIAL BREAKDOWN")
    print("=" * 70)

    semantic_preds = [p for p in all_preds_sorted if p in SEMANTIC_PREDS]
    weak_spatial_preds = [p for p in all_preds_sorted if p in WEAK_SPATIAL]
    neutral_spatial_preds = [p for p in all_preds_sorted if p in NEUTRAL_SPATIAL]

    def compute_group_metrics(pred_list, group_name):
        total = sum(pred_total.get(p, 0) for p in pred_list)
        correct = sum(pred_correct.get(p, 0) for p in pred_list)
        acc = correct / max(total, 1) * 100
        mean_logits = [np.mean(pred_raw_logits.get(p, [0])) for p in pred_list if pred_total.get(p, 0) > 0]
        mean_probs = [np.mean(pred_calibrated_probs.get(p, [0])) for p in pred_list if pred_total.get(p, 0) > 0]
        mean_priors = [np.mean(pred_prior_adjustments.get(p, [0])) for p in pred_list if pred_total.get(p, 0) > 0]
        return {
            "total": total, "correct": correct, "accuracy": round(acc, 2),
            "mean_logit": round(np.mean(mean_logits), 4) if mean_logits else 0,
            "mean_prob": round(np.mean(mean_probs), 4) if mean_probs else 0,
            "mean_prior": round(np.mean(mean_priors), 4) if mean_priors else 0,
        }

    sem_metrics = compute_group_metrics(semantic_preds, "semantic")
    ws_metrics = compute_group_metrics(weak_spatial_preds, "weak_spatial")
    ns_metrics = compute_group_metrics(neutral_spatial_preds, "neutral_spatial")

    print(f"  {'Group':<20} {'Total':>7} {'Correct':>8} {'Acc':>6} {'Mean Logit':>11} {'Mean Prob':>10} {'Mean Prior':>10}")
    print(f"  {'-' * 72}")
    print(f"  {'Semantic':<20} {sem_metrics['total']:>7} {sem_metrics['correct']:>8} {sem_metrics['accuracy']:>5.1f}% "
          f"{sem_metrics['mean_logit']:>10.4f} {sem_metrics['mean_prob']:>9.4f} {sem_metrics['mean_prior']:>+9.4f}")
    print(f"  {'Weak Spatial':<20} {ws_metrics['total']:>7} {ws_metrics['correct']:>8} {ws_metrics['accuracy']:>5.1f}% "
          f"{ws_metrics['mean_logit']:>10.4f} {ws_metrics['mean_prob']:>9.4f} {ws_metrics['mean_prior']:>+9.4f}")
    print(f"  {'Neutral Spatial':<20} {ns_metrics['total']:>7} {ns_metrics['correct']:>8} {ns_metrics['accuracy']:>5.1f}% "
          f"{ns_metrics['mean_logit']:>10.4f} {ns_metrics['mean_prob']:>9.4f} {ns_metrics['mean_prior']:>+9.4f}")

    # -- 8a. Fallback predicate analysis ------------------------------
    print(f"\n  Predicate dominance (which predicates are overpredicted):")
    pred_as_predicted = Counter()
    for r in all_results:
        pred_as_predicted[r["best_after_prior"]] += 1
    pred_as_gt = Counter()
    for r in all_results:
        pred_as_gt[r["gt_predicate"]] += 1

    print(f"  {'Predicate':<20} {'As GT':>8} {'As Predicted':>13} {'Ratio (pred/gt)':>16}")
    print(f"  {'-' * 57}")
    for p in all_preds_sorted:
        gt_count = pred_as_gt.get(p, 0)
        pred_count = pred_as_predicted.get(p, 0)
        ratio = pred_count / max(gt_count, 1)
        marker = " *" if ratio > 1.3 else " v" if ratio < 0.7 else ""
        print(f"  {p:<20} {gt_count:>8} {pred_count:>13} {ratio:>14.2f}x{marker}")

    # -- 9. FEATURE CONTRIBUTION ANALYSIS -----------------------------
    print("\n" + "=" * 70)
    print("FEATURE CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Geometry:           {contrib['geometry_pct']:5.1f}%  {'* DOMINANT' if contrib['geometry_pct'] > 15 else 'controlled'}")
    print(f"  CLIP (subj+obj):    {contrib['clip_total_pct']:5.1f}%  {'(underused)' if contrib['clip_total_pct'] < 30 else '(active)'}")
    print(f"  Subj CLIP:          {contrib['subj_clip_pct']:5.1f}%")
    print(f"  Obj CLIP:           {contrib['obj_clip_pct']:5.1f}%")
    print(f"  Union CLIP:         {contrib['union_clip_pct']:5.1f}%  {'(underused)' if contrib['union_clip_pct'] < 5 else '(active)'}")
    print(f"  Pose:               {contrib['pose_pct']:5.1f}%  {'(underused)' if contrib['pose_pct'] < 5 else '(active)'}")
    print(f"  Label embeddings:   {contrib['label_emb_pct']:5.1f}%")

    # -- 9a. Ablation statistics -------------------------------------
    ablation_records = [r for r in all_results if "ablation" in r]
    if ablation_records:
        print(f"\n  Ablation analysis ({len(ablation_records)} samples):")
        ablation_summary = defaultdict(lambda: {"count": 0, "changed": 0, "mean_logit_diff": 0})
        for r in ablation_records:
            a = r["ablation"]
            for key in ["clip", "union", "pose", "geo"]:
                ablation_summary[key]["count"] += 1
                if a.get(f"ablate_{key}_pred", r["pred_predicate"]) != r["pred_predicate"]:
                    ablation_summary[key]["changed"] += 1
                ablation_summary[key]["mean_logit_diff"] += a.get(f"{key}_logit_diff", 0)

        print(f"  {'Feature':<15} {'Pred Changed':>15} {'Logit Norm Diff':>16}")
        print(f"  {'-' * 46}")
        for key in ["clip", "union", "pose", "geo"]:
            info = ablation_summary[key]
            avg_diff = info["mean_logit_diff"] / max(info["count"], 1)
            change_pct = info["changed"] / max(info["count"], 1) * 100
            print(f"  {key:<15} {info['changed']:>4d}/{info['count']:<4d} ({change_pct:>4.1f}%)  {avg_diff:>14.4f}")

    # -- 10. LOGIT DOMINANCE ANALYSIS ---------------------------------
    print("\n" + "=" * 70)
    print("LOGIT DOMINANCE ANALYSIS")
    print("=" * 70)

    # For incorrect predictions, analyze whether bad predicate was dominant
    incorrect_records = [r for r in all_results if not r["is_correct"]]
    print(f"\n  Analyzing {len(incorrect_records)} incorrect predictions:")

    # Check if top-1 raw logit (before prior) was already wrong
    wrong_before_prior = sum(1 for r in incorrect_records if not r["is_correct_before_prior"])
    wrong_only_after_prior = sum(1 for r in incorrect_records if r["is_correct_before_prior"])
    print(f"    Wrong BEFORE priors (raw logits): {wrong_before_prior}/{len(incorrect_records)} "
          f"({wrong_before_prior/max(len(incorrect_records),1)*100:.1f}%)")
    print(f"    Correct BEFORE priors, wrong AFTER: {wrong_only_after_prior}/{len(incorrect_records)} "
          f"({wrong_only_after_prior/max(len(incorrect_records),1)*100:.1f}%)")

    # For wrong-before-prior: what was the top-1 logit gap?
    gaps_before = []
    for r in incorrect_records:
        if not r["is_correct_before_prior"] and len(r["top5_logits"]) >= 2:
            top1_val = r["top5_logits"][0][1]
            top2_val = r["top5_logits"][1][1]
            gaps_before.append(top1_val - top2_val)

    if gaps_before:
        mean_gap = np.mean(gaps_before)
        median_gap = np.median(gaps_before)
        pct_dominant = sum(1 for g in gaps_before if g > 0.5) / max(len(gaps_before), 1) * 100
        print(f"\n    Logit gap analysis (wrong-before-prior cases):")
        print(f"      Mean top1-top2 gap: {mean_gap:.4f}")
        print(f"      Median gap: {median_gap:.4f}")
        print(f"      % dominant (gap>0.5): {pct_dominant:.1f}%")

    # -- 10a. Prior analysis: does prior fix or break predictions? ----
    print(f"\n  Prior effect analysis:")
    n_helped = sum(1 for r in all_results if not r["is_correct_before_prior"] and r["is_correct_after_prior"])
    n_harmed = sum(1 for r in all_results if r["is_correct_before_prior"] and not r["is_correct_after_prior"])
    n_neutral = sum(1 for r in all_results if r["is_correct_before_prior"] == r["is_correct_after_prior"])
    print(f"    Priors HELPED (corrected wrong->right): {n_helped}")
    print(f"    Priors HARMED (flipped right->wrong):  {n_harmed}")
    print(f"    Priors NEUTRAL (no change):            {n_neutral}")

    # -- 10b. What are the "default fallback" predicates? -------------
    print('\n  "Fallback" predicates (dominant when wrong):')
    fallback_counter = Counter()
    for r in incorrect_records:
        fallback_counter[r["best_after_prior"]] += 1
    for pred, count in fallback_counter.most_common(10):
        pct = count / max(len(incorrect_records), 1) * 100
        print(f"    {pred:20s}: {count:>5d} ({pct:.1f}% of all errors)")

    # -- 11. SEMANTIC COLLAPSE ANALYSIS -------------------------------
    print("\n" + "=" * 70)
    print("SEMANTIC -> SPATIAL COLLAPSE ANALYSIS")
    print("=" * 70)
    semantic_to_spatial = 0
    semantic_to_semantic = 0
    spatial_to_semantic = 0
    spatial_to_spatial = 0

    for r in all_results:
        gt = r["gt_predicate"]
        pred = r["best_after_prior"]
        if gt in SEMANTIC_PREDS:
            if pred in SEMANTIC_PREDS:
                semantic_to_semantic += 1
            else:
                semantic_to_spatial += 1
        else:
            if pred in SEMANTIC_PREDS:
                spatial_to_semantic += 1
            else:
                spatial_to_spatial += 1

    total_sem_gt = semantic_to_semantic + semantic_to_spatial
    total_spatial_gt = spatial_to_semantic + spatial_to_spatial
    print(f"  Semantic GT -> Semantic pred: {semantic_to_semantic}/{total_sem_gt} "
          f"({semantic_to_semantic/max(total_sem_gt,1)*100:.1f}%)")
    print(f"  Semantic GT -> Spatial pred (COLLAPSE): {semantic_to_spatial}/{total_sem_gt} "
          f"({semantic_to_spatial/max(total_sem_gt,1)*100:.1f}%)")
    print(f"  Spatial GT -> Spatial pred: {spatial_to_spatial}/{total_spatial_gt} "
          f"({spatial_to_spatial/max(total_spatial_gt,1)*100:.1f}%)")
    print(f"  Spatial GT -> Semantic pred: {spatial_to_semantic}/{total_spatial_gt} "
          f"({spatial_to_semantic/max(total_spatial_gt,1)*100:.1f}%)")

    # -- 12. SAVE REPORTS ---------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING REPORTS")
    print("=" * 70)

    report = {
        "summary": {
            "total_evaluated": n_eval,
            "accuracy_final": round(n_correct / max(n_eval, 1) * 100, 2),
            "accuracy_before_prior": round(n_correct_bp / max(n_eval, 1) * 100, 2),
            "accuracy_after_prior": round(n_correct_ap / max(n_eval, 1) * 100, 2),
        },
        "feature_contributions": contrib,
        "feature_group_norms": {k: round(v, 4) for k, v in norms.items()},
        "predicate_metrics": pred_metrics,
        "semantic_vs_spatial": {
            "semantic": sem_metrics,
            "weak_spatial": ws_metrics,
            "neutral_spatial": ns_metrics,
        },
        "semantic_collapse": {
            "semantic_to_semantic": semantic_to_semantic,
            "semantic_to_spatial": semantic_to_spatial,
            "spatial_to_semantic": spatial_to_semantic,
            "spatial_to_spatial": spatial_to_spatial,
        },
        "fallback_predicates": dict(fallback_counter.most_common(20)),
        "prior_effects": {
            "priors_helped": n_helped,
            "priors_harmed": n_harmed,
            "priors_neutral": n_neutral,
        },
    }

    # Save full results as JSON (per-sample)
    print(f"  Saving per-sample results...")
    serializable_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "ablation"}
        sr["top5_logits"] = [(p, round(v, 4)) for p, v in sr.get("top5_logits", [])]
        sr["top5_probs"] = [(p, round(v, 4)) for p, v in sr.get("top5_probs", [])]
        serializable_results.append(sr)
    report["per_sample"] = serializable_results

    # Save ablation data separately
    ablation_data = [r["ablation"] for r in all_results if "ablation" in r]
    report["ablation_samples"] = ablation_data
    report["ablation_summary"] = {
        key: {
            "total": ablation_summary[key]["count"],
            "pred_changed": ablation_summary[key]["changed"],
            "pred_changed_pct": round(ablation_summary[key]["changed"] / max(ablation_summary[key]["count"], 1) * 100, 1),
            "mean_logit_diff": round(ablation_summary[key]["mean_logit_diff"] / max(ablation_summary[key]["count"], 1), 4),
        }
        for key in ["clip", "union", "pose", "geo"]
    }

    # Confusion matrix JSON-safe
    confusion_serializable = {}
    for gt in confusion_matrix:
        confusion_serializable[gt] = dict(confusion_matrix[gt])
    report["confusion_matrix_after_priors"] = confusion_serializable

    confusion_raw_serializable = {}
    for gt in confusion_raw:
        confusion_raw_serializable[gt] = dict(confusion_raw[gt])
    report["confusion_matrix_before_priors"] = confusion_raw_serializable

    # Top confusion pairs
    report["top_confusion_pairs"] = [
        {"true": gt, "predicted": pred, "count": count}
        for gt, pred, count in confusion_pairs[:30]
    ]

    # Object pair failures
    report["worst_object_pairs"] = [
        {
            "subject": pair.split(" -- ")[0],
            "object": pair.split(" -- ")[1],
            **info,
        }
        for pair, info in sorted_pairs[:30]
    ]

    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: {report_path}")

    # -- 12a. CSV summaries -------------------------------------------
    import csv

    # Predicate metrics CSV
    csv_path = os.path.join(output_dir, "predicate_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicate", "category", "total", "correct", "accuracy_pct",
                         "mean_logit", "mean_prob", "mean_prior"])
        for p in all_preds_sorted:
            m = pred_metrics.get(p, {})
            writer.writerow([p, m.get("category", ""), m.get("total", 0), m.get("correct", 0),
                            m.get("accuracy", 0), m.get("mean_logit", 0),
                            m.get("mean_prob", 0), m.get("mean_prior", 0)])
    print(f"  Saved: {csv_path}")

    # Confusion matrix CSV
    csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [""] + all_preds_sorted
        writer.writerow(header)
        for gt in all_preds_sorted:
            row = [gt]
            for pred in all_preds_sorted:
                row.append(confusion_matrix[gt].get(pred, 0))
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # Object pair failures CSV
    csv_path = os.path.join(output_dir, "object_pair_failures.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "object", "total_pairs", "failures", "failure_rate_pct",
                         "top_true_predicates", "top_predicted_predicates"])
        for pair_str, info in sorted_pairs[:50]:
            subj, obj = pair_str.split(" -- ")
            top_true = "; ".join(f"{p}({c})" for p, c in info["top_true"])
            top_pred = "; ".join(f"{p}({c})" for p, c in info["top_pred"])
            writer.writerow([subj, obj, info["total_pairs"], info["failure_count"],
                            info["failure_rate"], top_true, top_pred])
    print(f"  Saved: {csv_path}")

    # -- 13. FINAL REPORT ---------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSTIC REPORT")
    print("=" * 70)

    # Question 1: Which predicates dominate incorrectly?
    print(f"\n  Q1: Which predicates dominate incorrectly?")
    print(f"  {'-' * 50}")
    for pred, count in fallback_counter.most_common(5):
        pct = count / max(len(incorrect_records), 1) * 100
        acc = pred_correct.get(pred, 0) / max(pred_total.get(pred, 1), 1) * 100
        print(f"    {pred:20s}: {count:>5d} errors ({pct:.1f}%), accuracy={acc:.1f}%")

    # Question 2: Which object pairs are worst?
    print(f"\n  Q2: Which object pairs are worst?")
    print(f"  {'-' * 50}")
    for pair_str, info in sorted_pairs[:5]:
        subj, obj = pair_str.split(" -- ")
        print(f"    {subj:>15} -> {obj:<15}: {info['failure_count']:>4d} failures "
              f"({info['failure_rate']}%), predicted as {info['top_pred'][0][0] if info['top_pred'] else '?'}")

    # Question 3: Is geometry overpowering semantics?
    print(f"\n  Q3: Is geometry overpowering semantics?")
    print(f"  {'-' * 50}")
    print(f"    Geometry weight:         {contrib['geometry_pct']:.1f}%")
    print(f"    CLIP total weight:       {contrib['clip_total_pct']:.1f}%")
    print(f"    Union CLIP weight:       {contrib['union_clip_pct']:.1f}%")
    print(f"    Pose weight:             {contrib['pose_pct']:.1f}%")
    verdict = "YES -- geometry dominates" if contrib['geometry_pct'] > contrib['clip_total_pct'] else "NO -- CLIP matches or exceeds geometry"
    print(f"    Verdict: {verdict}")
    # Also check ablation
    geo_change_pct = ablation_summary["geo"]["changed"] / max(ablation_summary["geo"]["count"], 1) * 100
    clip_change_pct = ablation_summary["clip"]["changed"] / max(ablation_summary["clip"]["count"], 1) * 100
    print(f"    Ablation: removing geo changes pred {geo_change_pct:.1f}% of time")
    print(f"    Ablation: removing clip changes pred {clip_change_pct:.1f}% of time")

    # Question 4: Are semantic predicates undertrained?
    print(f"\n  Q4: Are semantic predicates undertrained?")
    print(f"  {'-' * 50}")
    sem_acc = sem_metrics["accuracy"]
    spatial_acc = (ws_metrics["total"] + ns_metrics["total"]) > 0 and \
        round((ws_metrics["correct"] + ns_metrics["correct"]) / max(ws_metrics["total"] + ns_metrics["total"], 1) * 100, 2) or 0
    print(f"    Semantic accuracy:      {sem_acc:.1f}%")
    print(f"    Spatial accuracy:       {spatial_acc:.1f}%")
    print(f"    Semantic mean logit:    {sem_metrics['mean_logit']:.4f}")
    print(f"    Spatial mean logit:     {ws_metrics['mean_logit']:.4f} (weak), {ns_metrics['mean_logit']:.4f} (neutral)")
    print(f"    Semantic occurrence:    {sem_metrics['total']} samples ({sem_metrics['total']/max(n_eval,1)*100:.1f}%)")
    print(f"    Spatial occurrence:     {ws_metrics['total'] + ns_metrics['total']} samples "
          f"({(ws_metrics['total'] + ns_metrics['total'])/max(n_eval,1)*100:.1f}%)")

    # Question 5: Are priors masking weak logits?
    print(f"\n  Q5: Are priors masking weak logits?")
    print(f"  {'-' * 50}")
    print(f"    Priors helped (wrong->right): {n_helped}")
    print(f"    Priors harmed (right->wrong): {n_harmed}")
    print(f"    Net prior effect: {n_helped - n_harmed}")
    print(f"    % of decisions changed by priors: {(n_helped + n_harmed)/max(n_eval,1)*100:.1f}%")
    prior_counter = Counter()
    for r in all_results:
        pa = r.get("gt_prior_adjustment", 0)
        if abs(pa) > 0.01:
            prior_counter[r["gt_predicate"]] += 1
    top_priors = ", ".join(p for p, _ in prior_counter.most_common(3))
    print(f"    Most common priors applied to: {top_priors}")

    # Question 6: Which feature group actually drives predictions?
    print(f"\n  Q6: Which feature group actually drives predictions?")
    print(f"  {'-' * 50}")
    print(f"    By weight norm: geometry ({contrib['geometry_pct']:.1f}%) dominates over "
          f"CLIP ({contrib['clip_total_pct']:.1f}%)")
    print(f"    By ablation effect: removing geo changes {geo_change_pct:.1f}% " +
          f"(vs clip {clip_change_pct:.1f}%)")
    geo_logit_diff = ablation_summary["geo"]["mean_logit_diff"] / max(ablation_summary["geo"]["count"], 1)
    clip_logit_diff = ablation_summary["clip"]["mean_logit_diff"] / max(ablation_summary["clip"]["count"], 1)
    print(f"    Geo ablation logit diff: {geo_logit_diff:.4f} (avg)")
    print(f"    CLIP ablation logit diff: {clip_logit_diff:.4f} (avg)")

    print(f"\n  Report saved to: {output_dir}/")
    print(f"  Key files:")
    print(f"    analysis_report.json  -- Full data")
    print(f"    predicate_metrics.csv  -- Per-predicate accuracy")
    print(f"    confusion_matrix.csv   -- Full confusion matrix")
    print(f"    object_pair_failures.csv -- Worst object pairs")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deep diagnostic analysis of relation prediction failures."
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Model checkpoint directory")
    parser.add_argument("--vg-root", default="./data/visual_genome",
                        help="Visual Genome data root")
    parser.add_argument("--output", default="./analysis_results",
                        help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit total dataset samples")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Validation fraction")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Softmax temperature")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true",
                        help="Only 200 validation samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze_relation_failures(
        checkpoint_dir=args.checkpoint_dir,
        vg_root=args.vg_root,
        output_dir=args.output,
        num_samples=args.num_samples,
        val_fraction=args.val_fraction,
        temperature=args.temperature,
        batch_size=args.batch_size,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
