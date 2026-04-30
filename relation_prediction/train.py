"""
Train the MLP relation classifier on Visual Genome.

Quick start
-----------
    python -m relation_prediction.train

Programmatic use:
    from relation_prediction.train import train_relation_model
    train_relation_model(vg_root="./data/visual_genome")

Expected VG data layout:
    <vg_root>/relationships.json
    <vg_root>/image_data.json

Download from:
    https://visualgenome.org/api/v0/api_home.html
    → "Relationships" and "Image Data" JSON files.

Outputs saved to <checkpoint_dir>/:
    relation_mlp.pt        — model weights
    label_vocab.json       — subject / object vocabulary
    pred_vocab.json        — predicate vocabulary
"""

from __future__ import annotations

import os
import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .model import RelationMLP
from .vg_dataset import VGRelationshipDataset, Vocab


# ---------------------------------------------------------------------------
# Defaults (override via function args or env vars)
# ---------------------------------------------------------------------------

VG_ROOT        = os.environ.get("VG_ROOT",        "./data/visual_genome")
CHECKPOINT_DIR = os.environ.get("REL_CKPT_DIR",   "./checkpoints")

BATCH_SIZE    = 512
EPOCHS        = 20
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_FRACTION  = 0.1
EMBED_DIM     = 64
HIDDEN_DIMS   = (256, 128)
DROPOUT       = 0.3
MIN_PRED_COUNT = 50   # drop rare predicates
MAX_SAMPLES   = None  # set e.g. 200_000 for a quick smoke-test
SEED          = 42


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
) -> RelationMLP:
    """
    Full training run.  Returns the trained model (also saves to disk).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[relation_prediction] device: {device}")

    rel_json  = os.path.join(vg_root, "relationships.json")
    img_json  = os.path.join(vg_root, "image_data.json")

    for path, label in [(rel_json, "relationships.json"), (img_json, "image_data.json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[relation_prediction] Missing: {path}\n"
                f"  Download the Visual Genome JSON files from "
                f"https://visualgenome.org/api/v0/api_home.html\n"
                f"  and place them under: {vg_root}/"
            )

    # --- Dataset -------------------------------------------------------
    print("[relation_prediction] Loading VG relationships …")
    full_ds = VGRelationshipDataset(
        relationships_json=rel_json,
        image_data_json=img_json,
        min_pred_count=min_pred_count,
        max_samples=max_samples,
    )
    label_vocab = full_ds.label_vocab
    pred_vocab  = full_ds.pred_vocab

    print(f"  samples      : {len(full_ds):,}")
    print(f"  label vocab  : {len(label_vocab):,}")
    print(f"  pred vocab   : {len(pred_vocab):,}")

    # Train / val split.
    n_val   = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # --- Model ---------------------------------------------------------
    model = RelationMLP(
        num_labels=len(label_vocab),
        num_predicates=len(pred_vocab),
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    print(f"  parameters   : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0

    # --- Epoch loop ----------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        for subj, obj, geo, pred in train_loader:
            subj = subj.to(device)
            obj  = obj.to(device)
            geo  = geo.to(device)
            pred = pred.to(device)

            optimizer.zero_grad()
            logits = model(subj, obj, geo)
            loss   = criterion(logits, pred)
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
            for subj, obj, geo, pred in val_loader:
                subj = subj.to(device)
                obj  = obj.to(device)
                geo  = geo.to(device)
                pred = pred.to(device)

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
            print(f"    → saved (val acc {best_val_acc:.3f})")

    print(f"[relation_prediction] Training done. Best val acc: {best_val_acc:.3f}")
    return model


def _save(model: RelationMLP, label_vocab: Vocab, pred_vocab: Vocab, ckpt_dir: str) -> None:
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "relation_mlp.pt"))
    label_vocab.save(os.path.join(ckpt_dir, "label_vocab.json"))
    pred_vocab.save(os.path.join(ckpt_dir, "pred_vocab.json"))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the VG relation MLP")
    parser.add_argument("--vg-root",        default=VG_ROOT)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--epochs",         type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--max-samples",    type=int,   default=None,
                        help="Cap total samples (handy for smoke-tests)")
    parser.add_argument("--min-pred-count", type=int,   default=MIN_PRED_COUNT)
    args = parser.parse_args()

    train_relation_model(
        vg_root=args.vg_root,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        min_pred_count=args.min_pred_count,
    )
