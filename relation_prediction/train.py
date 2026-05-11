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
from .vg_dataset import VGRelationshipDataset, Vocab, GEO_DIM


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
USE_VISUAL     = False
VG_IMAGE_DIR   = None
CLIP_CACHE_PATH = None


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
    vg_image_dir: Optional[str] = VG_IMAGE_DIR,
    clip_cache_path: Optional[str] = CLIP_CACHE_PATH,
) -> RelationMLP:
    torch.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[relation_prediction] device: {device}")
    print(f"[relation_prediction] mode: {'visual-semantic' if use_visual else 'geometry-only'}")

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
    )
    label_vocab = full_ds.label_vocab
    pred_vocab  = full_ds.pred_vocab

    print(f"  samples      : {len(full_ds):,}")
    print(f"  label vocab  : {len(label_vocab):,}")
    print(f"  pred vocab   : {len(pred_vocab):,}")

    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    def _collate(batch):
        """Collate function that handles optional visual features."""
        subj_idxs, obj_idxs, geos, preds = [], [], [], []
        subj_feats, obj_feats = [], []

        for item in batch:
            subj_idxs.append(item[0])
            obj_idxs.append(item[1])
            geos.append(item[2])
            preds.append(item[3])
            if len(item) > 4:
                subj_feats.append(item[4])
                obj_feats.append(item[5])

        result = (
            torch.stack(subj_idxs),
            torch.stack(obj_idxs),
            torch.stack(geos),
            torch.stack(preds),
        )
        if subj_feats:
            result = result + (torch.stack(subj_feats), torch.stack(obj_feats))
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
    model = RelationMLP(
        num_labels=len(label_vocab),
        num_predicates=len(pred_vocab),
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        clip_dim=clip_dim,
    ).to(device)

    input_dim = 2 * embed_dim + GEO_DIM + 2 * clip_dim
    print(f"  input dim    : {input_dim}")
    print(f"  parameters   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    # --- Epoch loop ----------------------------------------------------
    has_visual = use_visual

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

            if has_visual:
                subj_feat = batch[4].to(device)
                obj_feat  = batch[5].to(device)
                logits = model(subj, obj, geo, subj_feat, obj_feat)
            else:
                logits = model(subj, obj, geo)

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

                if has_visual:
                    subj_feat = batch[4].to(device)
                    obj_feat  = batch[5].to(device)
                    logits = model(subj, obj, geo, subj_feat, obj_feat)
                else:
                    logits = model(subj, obj, geo)

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
            _save(model, label_vocab, pred_vocab, checkpoint_dir)
            print(f"    -> saved (val acc {best_val_acc:.3f})")

    print(f"[relation_prediction] Training done. Best val acc: {best_val_acc:.3f}")
    return model


def _save(model, label_vocab, pred_vocab, ckpt_dir):
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "relation_mlp.pt"))
    label_vocab.save(os.path.join(ckpt_dir, "label_vocab.json"))
    pred_vocab.save(os.path.join(ckpt_dir, "pred_vocab.json"))


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
    parser.add_argument("--vg-image-dir",   type=str,   default=None,
                        help="Directory with VG images (e.g. ./data/visual_genome/images)")
    parser.add_argument("--clip-cache-path", type=str,  default=None,
                        help="Path to CLIP embedding cache (.pkl)")
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
        vg_image_dir=args.vg_image_dir,
        clip_cache_path=args.clip_cache_path,
    )
