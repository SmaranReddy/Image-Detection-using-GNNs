"""
Train the MLP relation classifier on Visual Genome.

Supports two modes:
    1. Geometry-only (legacy): 5-dim geometric features.
    2. Visual-semantic (new):   Geo + CLIP visual embeddings.

Quick start (geometry-only):
    python -m relation_prediction.train

Quick start (visual-semantic):
    python -m relation_prediction.train --use-visual --vg-image-dir ./data/visual_genome/images

Programmatic use:
    from relation_prediction.train import train_relation_model
    train_relation_model(vg_root="./data/visual_genome")
"""

from __future__ import annotations

import os
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .model import RelationMLP
from .relation_transformer import RelationTransformer
from .vg_dataset import VGRelationshipDataset, Vocab, GEO_DIM, POSE_FEATURE_DIM, UNION_FEATURE_DIM


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

VG_ROOT        = os.environ.get("VG_ROOT",        "./data/visual_genome")
CHECKPOINT_DIR = os.environ.get("REL_CKPT_DIR",   "./checkpoints")

BATCH_SIZE     = 512
EPOCHS         = 20
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
VAL_FRACTION   = 0.1
EMBED_DIM      = 64
HIDDEN_DIMS    = (256, 128)
DROPOUT        = 0.3
MIN_PRED_COUNT = 50
MAX_SAMPLES    = None
SEED           = 42
USE_VISUAL      = False
REQUIRE_VISUAL  = False
VG_IMAGE_DIR    = None
CLIP_CACHE_PATH = None
USE_POSE        = False
USE_UNION       = False
MODEL_TYPE      = "mlp"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_relation_model(
    vg_root: str = VG_ROOT,
    checkpoint_dir: str = CHECKPOINT_DIR,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    val_fraction: float = VAL_FRACTION,
    embed_dim: int = EMBED_DIM,
    hidden_dims: tuple = HIDDEN_DIMS,
    dropout: float = DROPOUT,
    min_pred_count: int = MIN_PRED_COUNT,
    max_samples: Optional[int] = MAX_SAMPLES,
    seed: int = SEED,
    device: Optional[torch.device] = None,
    use_visual: bool = USE_VISUAL,
    require_visual: bool = REQUIRE_VISUAL,
    vg_image_dir: Optional[str] = VG_IMAGE_DIR,
    clip_cache_path: Optional[str] = CLIP_CACHE_PATH,
    use_pose: bool = USE_POSE,
    use_union: bool = USE_UNION,
    model_type: str = MODEL_TYPE,
) -> nn.Module:
    torch.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[relation_prediction] device: {device}")
    mode_parts = []
    if use_visual:
        mode_parts.append('visual-semantic')
        if require_visual:
            mode_parts.append('strict')
    else:
        mode_parts.append('geometry-only')
    if use_pose:
        mode_parts.append('pose')
    if use_union:
        mode_parts.append('union')
    mode_str = '+'.join(mode_parts) if mode_parts else 'geometry-only'
    print(f"[relation_prediction] mode: {mode_str}")

    rel_json = os.path.join(vg_root, "relationships.json")
    img_json = os.path.join(vg_root, "image_data.json")

    for path, label in [(rel_json, "relationships.json"), (img_json, "image_data.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[relation_prediction] Missing: {path}\n"
                f"  Download from https://visualgenome.org/api/v0/api_home.html\n"
                f"  and place under: {vg_root}/"
            )

    # --- Dataset -------------------------------------------------------
    print("[relation_prediction] Loading VG relationships …")
    full_ds = VGRelationshipDataset(
        relationships_json=rel_json,
        image_data_json=img_json,
        vg_image_dir=vg_image_dir,
        min_pred_count=min_pred_count,
        max_samples=max_samples,
        use_visual=use_visual,
        clip_cache_path=clip_cache_path,
        require_visual=require_visual,
        use_pose=use_pose,
        use_union=use_union,
    )
    label_vocab = full_ds.label_vocab
    pred_vocab  = full_ds.pred_vocab

    print(f"  samples      : {len(full_ds):,}")
    print(f"  label vocab  : {len(label_vocab):,}")
    print(f"  pred vocab   : {len(pred_vocab):,}")

    # Pre-training validation: verify dataset purity
    if use_visual:
        real, total, coverage_pct = full_ds.compute_clip_coverage()
        print(f"\n  [CLIP COVERAGE] Real: {real:,} / {total:,} = {coverage_pct:.2f}%")
        if require_visual:
            assert coverage_pct == 100.0, (
                f"[PURE VISUAL] Coverage is {coverage_pct:.2f}%, expected 100%. "
                "Strict mode filtering failed."
            )
            # Full zero-vector audit
            zero_count = 0
            subj_norms, obj_norms = [], []
            for idx in range(total):
                item = full_ds[idx]
                sn = item[4].norm().item()
                on = item[5].norm().item()
                subj_norms.append(sn)
                obj_norms.append(on)
                if sn < 0.001 or on < 0.001:
                    zero_count += 1
            assert zero_count == 0, (
                f"[PURE VISUAL] {zero_count} zero-vector features detected!"
            )
            all_norms = subj_norms + obj_norms
            mean_n = sum(all_norms) / len(all_norms)
            min_n = min(all_norms)
            max_n = max(all_norms)
            print(f"  [VALIDATION] Zero-vector features: 0/{total * 2} — PASS")
            print(f"  [NORM CHECK] Mean {mean_n:.4f}, Min {min_n:.4f}, Max {max_n:.4f}")
            # Predicate distribution
            from collections import Counter
            pred_counter = Counter()
            for idx in range(total):
                pred_name = full_ds.pred_vocab.token(full_ds[idx][3].item())
                pred_counter[pred_name] += 1
            print(f"  [PRED DIST] Retained predicate distribution:")
            for pred, count in pred_counter.most_common():
                print(f"    {pred}: {count}")

    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    def _collate(batch):
        """Collate function that handles optional visual, union, and pose features."""
        subj_idxs, obj_idxs, geos, preds = [], [], [], []
        subj_feats, obj_feats = [], []
        union_feats, pose_feats = [], []

        for item in batch:
            subj_idxs.append(item[0])
            obj_idxs.append(item[1])
            geos.append(item[2])
            preds.append(item[3])
            idx = 4
            if use_visual:
                subj_feats.append(item[idx]); obj_feats.append(item[idx + 1])
                idx += 2
                if use_union:
                    union_feats.append(item[idx]); idx += 1
                if use_pose:
                    pose_feats.append(item[idx]); idx += 1

        result = (
            torch.stack(subj_idxs),
            torch.stack(obj_idxs),
            torch.stack(geos),
            torch.stack(preds),
        )
        if subj_feats:
            result = result + (torch.stack(subj_feats), torch.stack(obj_feats))
        if union_feats:
            result = result + (torch.stack(union_feats),)
        if pose_feats:
            result = result + (torch.stack(pose_feats),)
        return result

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )

    # --- Model ---------------------------------------------------------
    clip_dim = full_ds.CLIP_DIM if use_visual else 0
    pose_dim = POSE_FEATURE_DIM if use_pose else 0
    union_dim = UNION_FEATURE_DIM if use_union else 0

    if model_type == "transformer":
        model = RelationTransformer(
            num_labels=len(label_vocab),
            num_predicates=len(pred_vocab),
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            dropout=dropout if dropout < 0.3 else 0.1,
        ).to(device)
        input_dim = model.d_model
    else:
        model = RelationMLP(
            num_labels=len(label_vocab),
            num_predicates=len(pred_vocab),
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
        ).to(device)
        input_dim = 2 * embed_dim + GEO_DIM + 2 * clip_dim + union_dim + pose_dim

    print(f"  input dim    : {input_dim}")
    print(f"  model type   : {model_type}")
    print(f"  parameters   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    # --- Epoch loop ----------------------------------------------------
    has_visual = use_visual
    has_union = use_union
    has_pose = use_pose

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for batch in train_loader:
            subj = batch[0].to(device)
            obj  = batch[1].to(device)
            geo  = batch[2].to(device)
            pred = batch[3].to(device)

            optimizer.zero_grad()

            idx = 4
            subj_feat = batch[idx].to(device) if has_visual else None
            obj_feat  = batch[idx + 1].to(device) if has_visual else None
            idx += 2 if has_visual else 0
            union_feat = batch[idx].to(device) if has_union else None
            idx += 1 if has_union else 0
            pose_feat  = batch[idx].to(device) if has_pose else None

            logits = model(subj, obj, geo,
                           subj_feat=subj_feat, obj_feat=obj_feat,
                           union_feat=union_feat, pose_feat=pose_feat)

            loss = criterion(logits, pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * pred.size(0)
            preds          = logits.argmax(dim=-1)
            train_correct += (preds == pred).sum().item()
            train_total   += pred.size(0)

        scheduler.step()

        # --- Validation ------------------------------------------------
        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for batch in val_loader:
                subj = batch[0].to(device)
                obj  = batch[1].to(device)
                geo  = batch[2].to(device)
                pred = batch[3].to(device)

                idx = 4
                subj_feat = batch[idx].to(device) if has_visual else None
                obj_feat  = batch[idx + 1].to(device) if has_visual else None
                idx += 2 if has_visual else 0
                union_feat = batch[idx].to(device) if has_union else None
                idx += 1 if has_union else 0
                pose_feat  = batch[idx].to(device) if has_pose else None

                logits = model(subj, obj, geo,
                               subj_feat=subj_feat, obj_feat=obj_feat,
                               union_feat=union_feat, pose_feat=pose_feat)

                preds  = logits.argmax(dim=-1)
                val_correct += (preds == pred).sum().item()
                val_total   += pred.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc   = val_correct   / max(val_total,   1)
        avg_loss  = train_loss    / max(train_total,  1)

        print(
            f"  epoch {epoch:3d}/{epochs} | "
            f"loss {avg_loss:.4f} | "
            f"train acc {train_acc:.3f} | "
            f"val acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save(model, label_vocab, pred_vocab, checkpoint_dir,
                  dataset=full_ds, mode_label=mode_str,
                  use_visual=use_visual, require_visual=require_visual)
            print(f"    -> saved (val acc {best_val_acc:.3f})")

    print(f"[relation_prediction] Training done. Best val acc: {best_val_acc:.3f}")
    return model


def _save(model, label_vocab, pred_vocab, ckpt_dir, dataset=None, mode_label="unknown",
          use_visual=False, require_visual=False):
    # Save model state dict + architecture config for forward-compatible loading.
    state = model.state_dict()
    model_type = "transformer" if isinstance(model, RelationTransformer) else "mlp"

    if model_type == "transformer":
        config = {
            "model_type": "transformer",
            "num_labels": model.label_emb.num_embeddings,
            "num_predicates": model.num_predicates,
            "d_model": model.d_model,
            "embed_dim": model.embed_dim,
            "clip_dim": model.clip_dim,
            "pose_dim": model.pose_dim,
            "union_dim": model.union_dim,
        }
    else:
        config = {
            "model_type": "mlp",
            "num_labels": model.label_emb.num_embeddings,
            "num_predicates": model.mlp[-1].out_features,
            "embed_dim": model.label_emb.embedding_dim,
            "clip_dim": model.clip_dim,
            "pose_dim": model.pose_dim,
            "union_dim": model.union_dim,
        }
        # Infer hidden_dims from state dict — iterate over mlp.{i}.weight
        # keys (skipping the final output layer) using the same stepped
        # indexing that _infer_hidden_dims in predict.py uses.
        hdims = []
        idx = 0
        while True:
            key = f"mlp.{idx}.weight"
            if key not in state:
                break
            hdims.append(state[key].shape[0])
            idx += 3
        config["hidden_dims"] = hdims[:-1]  # exclude the output layer
    torch.save({"model_state_dict": state, "model_config": config},
               os.path.join(ckpt_dir, "relation_mlp.pt"))
    label_vocab.save(os.path.join(ckpt_dir, "label_vocab.json"))
    pred_vocab.save(os.path.join(ckpt_dir, "pred_vocab.json"))

    import time, json
    from collections import Counter
    meta = {
        "timestamp": time.time(),
        "mode": mode_label,
        "model_type": model_type,
        "use_visual": use_visual,
        "require_visual": require_visual,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "embed_dim": EMBED_DIM,
        "hidden_dims": list(HIDDEN_DIMS) if model_type == "mlp" else None,
        "d_model": model.d_model if model_type == "transformer" else None,
        "dropout": DROPOUT,
        "seed": SEED,
    }
    if dataset is not None:
        meta["dataset_size"] = len(dataset)
        meta["num_labels"] = len(dataset.label_vocab)
        meta["num_predicates"] = len(dataset.pred_vocab)
        # Real CLIP coverage
        if use_visual:
            real, total, pct = dataset.compute_clip_coverage()
            meta["real_clip_coverage_pct"] = round(pct, 2)
            meta["real_clip_samples"] = real
            meta["total_samples"] = total
        # Retained sample count (= dataset size in pure visual, total in mixed)
        meta["retained_sample_count"] = len(dataset)
        pred_counter = Counter()
        for idx in range(len(dataset)):
            pred_counter[dataset.pred_vocab.token(dataset[idx][3].item())] += 1
        meta["predicate_distribution"] = dict(pred_counter.most_common())
    with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the VG relation MLP")
    parser.add_argument("--vg-root",        default=VG_ROOT)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--epochs",         type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--max-samples",    type=int,   default=None)
    parser.add_argument("--min-pred-count", type=int,   default=MIN_PRED_COUNT)
    parser.add_argument("--use-visual",     action="store_true",  default=False,
                        help="Enable CLIP visual features")
    parser.add_argument("--require-visual", action="store_true",  default=False,
                        help="Strict: drop samples with missing CLIP embeddings")
    parser.add_argument("--vg-image-dir",   type=str,   default=None,
                        help="Directory with VG images (e.g. ./data/visual_genome/images)")
    parser.add_argument("--clip-cache-path", type=str,  default=None,
                        help="Path to CLIP embedding cache (.pkl)")
    parser.add_argument("--use-pose",  action="store_true", default=False,
                        help="Enable pose features (requires MediaPipe)")
    parser.add_argument("--use-union", action="store_true", default=False,
                        help="Enable union-region CLIP features")
    parser.add_argument("--model", type=str, default=MODEL_TYPE, choices=["mlp", "transformer"],
                        help="Model architecture (mlp or transformer)")
    args = parser.parse_args()

    train_relation_model(
        vg_root=args.vg_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        min_pred_count=args.min_pred_count,
        use_visual=args.use_visual,
        require_visual=args.require_visual,
        vg_image_dir=args.vg_image_dir,
        clip_cache_path=args.clip_cache_path,
        use_pose=args.use_pose,
        use_union=args.use_union,
        model_type=args.model,
    )
