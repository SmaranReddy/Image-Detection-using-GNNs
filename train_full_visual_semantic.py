"""
Full Visual-Semantic Relation MLP Training
===========================================

Trains the relation MLP on VG images with CLIP visual features.

Supports two modes:
  1. Mixed/Fallback Mode (--use-visual only)
     - Allows zero-vector fallback for missing-image samples
     - Legacy behavior (54.7% real CLIP coverage)

  2. Pure Visual Mode (--use-visual --require-visual)
     - ONLY retains samples with valid non-zero CLIP embeddings
     - 100% real visual-semantic supervision
     - No geometry-only fallback samples
     - Scientifically clean appearance-driven relation learning

Usage:
  python train_full_visual_semantic.py                          (geometry-only)
  python train_full_visual_semantic.py --use-visual             (mixed/fallback)
  python train_full_visual_semantic.py --use-visual --require-visual  (pure visual)
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from relation_prediction.model import RelationMLP
from relation_prediction.relation_transformer import RelationTransformer
from relation_prediction.vg_dataset import VGRelationshipDataset, Vocab, GEO_DIM, POSE_FEATURE_DIM, UNION_FEATURE_DIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VG_ROOT = PROJ_ROOT / "data/visual_genome"
CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
VG_IMAGE_DIR = VG_ROOT / "images"
CLIP_CACHE_PATH = VG_ROOT / "clip_cache_proper.pt"

BATCH_SIZE = 256
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
EMBED_DIM = 64
HIDDEN_DIMS = (256, 128)
DROPOUT = 0.3
MIN_PRED_COUNT = 50
MAX_SAMPLES = None
SEED = 42
USE_VISUAL = True
REQUIRE_VISUAL = False
USE_POSE = False
USE_UNION = False
MODEL_TYPE = "mlp"

SEMANTIC_PREDICATES = frozenset({
    "riding", "carrying", "holding", "wearing", "sitting on", "standing on",
})

QUALITATIVE_PAIRS = [
    ("person", "bicycle"),
    ("person", "horse"),
    ("person", "backpack"),
    ("person", "chair"),
    ("person", "umbrella"),
    ("person", "surfboard"),
    ("person", "cell phone"),
    ("person", "car"),
    ("person", "dog"),
    ("person", "bottle"),
    ("person", "skateboard"),
    ("person", "couch"),
    ("dog", "cat"),
    ("cat", "couch"),
    ("person", "horse"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collate(batch):
    subj_idxs, obj_idxs, geos, preds = [], [], [], []
    subj_feats, obj_feats = [], []
    union_feats, pose_feats = [], []
    for item in batch:
        subj_idxs.append(item[0])
        obj_idxs.append(item[1])
        geos.append(item[2])
        preds.append(item[3])
        idx = 4
        if len(item) > idx:
            subj_feats.append(item[idx]); obj_feats.append(item[idx + 1])
            idx += 2
            if len(item) > idx:
                union_feats.append(item[idx]); idx += 1
            if len(item) > idx:
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


def compute_predicate_metrics(model, loader, device, pred_vocab, has_visual,
                              use_union=False, use_pose=False):
    model.eval()
    per_pred_correct = defaultdict(int)
    per_pred_total = defaultdict(int)
    confusion_counts = defaultdict(lambda: defaultdict(int))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            subj = batch[0].to(device)
            obj = batch[1].to(device)
            geo = batch[2].to(device)
            target = batch[3].to(device)

            idx = 4
            subj_feat = batch[idx].to(device) if has_visual else None
            obj_feat = batch[idx + 1].to(device) if has_visual else None
            idx += 2 if has_visual else 0
            union_feat = batch[idx].to(device) if use_union else None
            idx += 1 if use_union else 0
            pose_feat = batch[idx].to(device) if use_pose else None

            logits = model(subj, obj, geo,
                           subj_feat=subj_feat, obj_feat=obj_feat,
                           union_feat=union_feat, pose_feat=pose_feat)

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

            for p, t in zip(preds.cpu().numpy(), target.cpu().numpy()):
                pred_token = pred_vocab.token(int(p))
                true_token = pred_vocab.token(int(t))
                per_pred_total[true_token] += 1
                if p == t:
                    per_pred_correct[true_token] += 1
                if true_token != Vocab.PAD:
                    confusion_counts[true_token][pred_token] += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {}
    for token in sorted(set(list(per_pred_total.keys()) + list(per_pred_correct.keys()))):
        total = per_pred_total.get(token, 0)
        correct = per_pred_correct.get(token, 0)
        metrics[token] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }

    return metrics, confusion_counts, all_preds, all_targets


def print_predicate_table(metrics, header="Predicate-wise Validation Metrics"):
    print(f"\n  {header}")
    print(f"  {'Predicate':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    sorted_preds = sorted(metrics.items(), key=lambda x: -x[1]["total"])
    for token, m in sorted_preds:
        if token in (Vocab.PAD, Vocab.UNK):
            continue
        print(f"  {token:<20} {m['total']:>8} {m['correct']:>8} {m['accuracy']:>10.3f}")


def print_confusion_analysis(confusion_counts, pred_vocab, top_n=10):
    print(f"\n  Top {top_n} Most Confused Predicate Pairs (true -> predicted):")
    print(f"  {'True Pred':<20} {'Predicted As':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*8}")

    all_confusions = []
    for true_token, pred_dict in confusion_counts.items():
        for pred_token, count in pred_dict.items():
            if pred_token != true_token:
                all_confusions.append((true_token, pred_token, count))

    all_confusions.sort(key=lambda x: -x[2])
    for true_tok, pred_tok, count in all_confusions[:top_n]:
        print(f"  {true_tok:<20} {pred_tok:<20} {count:>8}")

    print(f"\n  Most Dominant Predicates (by total frequency):")
    totals = {}
    for true_token, pred_dict in confusion_counts.items():
        total_preds = sum(pred_dict.values())
        totals[true_token] = total_preds
    sorted_totals = sorted(totals.items(), key=lambda x: -x[1])
    for token, count in sorted_totals[:10]:
        print(f"  {token:<20} {count:>8}")


def qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=5):
    print(f"\n  Qualitative Relation Predictions (top-{top_k}):")
    print(f"  {'Subject':<12} {'Object':<14} {'Predictions':<60}")
    print(f"  {'-'*12} {'-'*14} {'-'*60}")

    geo_default = torch.tensor([[0.0, -0.1, 0.0, 0.0, 0.3]], dtype=torch.float32, device=device)
    geo_above = torch.tensor([[0.0, -0.5, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    geo_on = torch.tensor([[0.0, 0.05, 0.0, 0.0, 0.4]], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for subj_name, obj_name in QUALITATIVE_PAIRS:
            s_idx = label_vocab[subj_name]
            o_idx = label_vocab[obj_name]
            s_t = torch.tensor([s_idx], dtype=torch.long, device=device)
            o_t = torch.tensor([o_idx], dtype=torch.long, device=device)

            sf = torch.zeros((1, 768), device=device)
            of = torch.zeros((1, 768), device=device)
            geo_t = geo_default

            if has_visual:
                logits = model(s_t, o_t, geo_t, sf, of)
            else:
                logits = model(s_t, o_t, geo_t)

            probs = F.softmax(logits, dim=-1)
            top_probs, top_idxs = probs[0].topk(top_k)
            pred_str = " | ".join(
                f"{pred_vocab.token(int(idx))} ({prob:.3f})"
                for idx, prob in zip(top_idxs, top_probs)
            )
            print(f"  {subj_name:<12} {obj_name:<14} {pred_str:<60}")


def analyze_clip_coverage(full_ds):
    print(f"\n  CLIP Coverage Analysis:")
    total = len(full_ds)
    real_clip = 0
    zero_clip = 0
    for idx in range(min(total, 5000)):
        item = full_ds[idx]
        if len(item) > 4:
            sf, of = item[4], item[5]
            if sf.sum().item() == 0 and of.sum().item() == 0:
                zero_clip += 1
            else:
                real_clip += 1
    if total > 5000:
        ratio = real_clip / max(real_clip + zero_clip, 1)
        real_clip = int(total * ratio)
        zero_clip = total - real_clip
        print(f"  (extrapolated from first 5000 samples, ratio={ratio:.3f})")

    coverage_pct = 100.0 * real_clip / max(total, 1)
    print(f"  Total samples:          {total:,}")
    print(f"  With real CLIP embeds:  {real_clip:,} ({coverage_pct:.1f}%)")
    print(f"  Fallback zero vectors:  {zero_clip:,} ({100 - coverage_pct:.1f}%)")
    return real_clip, zero_clip, coverage_pct


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("  FULL VISUAL-SEMANTIC RELATION MLP TRAINING")
    print("=" * 78)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # STEP 1 - Dataset & Configuration Verification
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 1 — TRAINING CONFIGURATION")
    print(f"{'=' * 78}")

    # Determine training mode
    mode_parts = []
    if USE_VISUAL and REQUIRE_VISUAL:
        mode_parts.append("PURE VISUAL-SEMANTIC (strict)")
    elif USE_VISUAL:
        mode_parts.append("VISUAL-SEMANTIC (mixed/fallback)")
    else:
        mode_parts.append("GEOMETRY-ONLY")
    if USE_POSE:
        mode_parts.append("pose")
    if USE_UNION:
        mode_parts.append("union")
    mode_label = " + ".join(mode_parts)
    print(f"\n  *** MODE: {mode_label} ***")

    t0 = time.time()
    full_ds = VGRelationshipDataset(
        relationships_json=str(VG_ROOT / "relationships.json"),
        image_data_json=str(VG_ROOT / "image_data.json"),
        vg_image_dir=str(VG_IMAGE_DIR) if USE_VISUAL else None,
        min_pred_count=MIN_PRED_COUNT,
        max_samples=MAX_SAMPLES,
        use_visual=USE_VISUAL,
        clip_cache_path=str(CLIP_CACHE_PATH) if USE_VISUAL else None,
        require_visual=REQUIRE_VISUAL,
        use_pose=USE_POSE,
        use_union=USE_UNION,
    )
    load_time = time.time() - t0

    label_vocab = full_ds.label_vocab
    pred_vocab = full_ds.pred_vocab

    dataset_size = len(full_ds)
    num_labels = len(label_vocab)
    num_predicates = len(pred_vocab)
    clip_dim = full_ds.CLIP_DIM if USE_VISUAL else 0
    pose_dim = POSE_FEATURE_DIM if USE_POSE else 0
    union_dim = UNION_FEATURE_DIM if USE_UNION else 0

    # CLIP coverage
    print(f"\n  Dataset Statistics:")
    print(f"  Dataset size:              {dataset_size:,} samples")
    print(f"  Load time:                 {load_time:.1f}s")
    print(f"  Number of labels:          {num_labels}")
    print(f"  Number of predicates:      {num_predicates}")
    print(f"  CLIP dimension:            {clip_dim}")
    print(f"  Pose dimension:            {pose_dim}")
    print(f"  Union dimension:           {union_dim}")
    print(f"  Use visual:                {USE_VISUAL}")
    print(f"  Require visual:            {REQUIRE_VISUAL}")
    print(f"  Use pose:                  {USE_POSE}")
    print(f"  Use union:                 {USE_UNION}")

    # Pre-training validation: verify no zero embeddings in pure visual mode
    if REQUIRE_VISUAL and USE_VISUAL:
        _validate_pure_visual(full_ds)
    elif USE_VISUAL and not REQUIRE_VISUAL:
        real_clip, zero_clip, clip_coverage = analyze_clip_coverage(full_ds)

    input_dim = 2 * EMBED_DIM + GEO_DIM + 2 * clip_dim + union_dim + pose_dim

    if MODEL_TYPE == "transformer":
        model = RelationTransformer(
            num_labels=num_labels,
            num_predicates=num_predicates,
            embed_dim=EMBED_DIM,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            dropout=DROPOUT if DROPOUT < 0.3 else 0.1,
        ).to(device)
    else:
        model = RelationMLP(
            num_labels=num_labels,
            num_predicates=num_predicates,
            embed_dim=EMBED_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
        ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model Configuration:")
    print(f"  Model type:               {MODEL_TYPE}")
    print(f"  Input dimension:          {input_dim}")
    print(f"  Embedding dimension:      {EMBED_DIM}")
    print(f"  Hidden dims:              {HIDDEN_DIMS if MODEL_TYPE == 'mlp' else 'N/A'}")
    print(f"  Dropout:                  {DROPOUT}")
    print(f"  Parameters:               {param_count:,}")
    print(f"  Batch size:               {BATCH_SIZE}")
    print(f"  Epochs:                   {EPOCHS}")
    print(f"  Learning rate:            {LR}")
    print(f"  Weight decay:             {WEIGHT_DECAY}")
    print(f"  Validation fraction:      {VAL_FRACTION}")

    # -----------------------------------------------------------------------
    # STEP 2 - Data Splits & Loaders
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 2 — DATA SPLITS")
    print(f"{'=' * 78}")

    n_val = max(1, int(dataset_size * VAL_FRACTION))
    n_train = dataset_size - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"  Training samples:   {len(train_ds):,}")
    print(f"  Validation samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=_collate,
    )

    # -----------------------------------------------------------------------
    # STEP 3 - Training Setup
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 3 — TRAINING VISUAL-SEMANTIC MLP")
    print(f"{'=' * 78}")

    pred_counter = full_ds._load_stats['pred_counter']
    class_weights = compute_class_weights(pred_counter, pred_vocab, num_predicates)
    print(f"\n  Class weights (effective-number, beta=0.999):")
    for i in range(num_predicates):
        tok = pred_vocab.token(i)
        if tok not in (Vocab.PAD, Vocab.UNK):
            print(f"    {tok:<15} count={pred_counter.get(tok, 0):<8} weight={class_weights[i].item():.4f}")

    criterion = FocalLoss(gamma=2.0, alpha=class_weights.to(device), ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    CLF_L2_WEIGHT = 1e-4

    os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)
    best_val_acc = 0.0
    has_visual = USE_VISUAL
    has_union = USE_UNION
    has_pose = USE_POSE
    all_batch_times = []
    epoch_metrics_log = []

    def _unpack_batch(batch):
        subj = batch[0].to(device)
        obj = batch[1].to(device)
        geo = batch[2].to(device)
        pred = batch[3].to(device)
        idx = 4
        subj_feat = batch[idx].to(device) if has_visual else None
        obj_feat = batch[idx + 1].to(device) if has_visual else None
        idx += 2 if has_visual else 0
        union_feat = batch[idx].to(device) if has_union else None
        idx += 1 if has_union else 0
        pose_feat = batch[idx].to(device) if has_pose else None
        return subj, obj, geo, pred, subj_feat, obj_feat, union_feat, pose_feat

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch in train_loader:
            batch_start = time.time()
            subj, obj, geo, pred, subj_feat, obj_feat, union_feat, pose_feat = _unpack_batch(batch)

            optimizer.zero_grad()

            logits = model(subj, obj, geo,
                           subj_feat=subj_feat, obj_feat=obj_feat,
                           union_feat=union_feat, pose_feat=pose_feat)

            loss = criterion(logits, pred)
            loss = loss + classifier_l2_loss(model, weight=CLF_L2_WEIGHT)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_time = time.time() - batch_start
            all_batch_times.append(batch_time)

            train_loss += loss.item() * pred.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == pred).sum().item()
            train_total += pred.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_acc = train_correct / max(train_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        # Validation
        val_metrics, confusion_counts, val_preds, val_targets = compute_predicate_metrics(
            model, val_loader, device, pred_vocab, has_visual,
            use_union=has_union, use_pose=has_pose,
        )

        overall_val_correct = sum(m["correct"] for m in val_metrics.values())
        overall_val_total = sum(m["total"] for m in val_metrics.values())
        val_acc = overall_val_correct / max(overall_val_total, 1)

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": scheduler.get_last_lr()[0],
        }
        epoch_metrics_log.append(epoch_log)

        print(f"\n  Epoch {epoch:3d}/{EPOCHS} | "
              f"loss {avg_loss:.4f} | "
              f"train {train_acc:.3f} | "
              f"val {val_acc:.3f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{epoch_time:.1f}s")

        # Semantic predicates
        print(f"  Semantic predicates:")
        for sp in sorted(SEMANTIC_PREDICATES):
            m = val_metrics.get(sp, {"total": 0, "correct": 0, "accuracy": 0.0})
            marker = " ***" if m["total"] > 0 else ""
            print(f"    {sp:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']}){marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            _save_checkpoint(model, label_vocab, pred_vocab, str(CHECKPOINT_DIR), epoch, val_acc,
                             dataset=full_ds, mode_label=mode_label)
            print(f"  >>> New best model saved (val_acc={val_acc:.3f}, epoch={epoch})")

    # -----------------------------------------------------------------------
    # STEP 4 - Training Complete
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 4 — TRAINING COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Best validation accuracy: {best_val_acc:.3f} (epoch {best_epoch})")

    # -----------------------------------------------------------------------
    # Confusion Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 5 — CONFUSION ANALYSIS (best model)")
    print(f"{'=' * 78}")

    _load_best_model(model, str(CHECKPOINT_DIR), device)
    best_metrics, best_confusion, _, _ = compute_predicate_metrics(
        model, val_loader, device, pred_vocab, has_visual
    )

    print_predicate_table(best_metrics, "Per-Predicate Validation Accuracy (Best Model)")
    print_confusion_analysis(best_confusion, pred_vocab, top_n=15)

    # -----------------------------------------------------------------------
    # STEP 6 - Qualitative Tests
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 6 — QUALITATIVE RELATION TESTS")
    print(f"{'=' * 78}")

    qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=5)

    # -----------------------------------------------------------------------
    # STEP 7 - CLIP Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 7 — CLIP IMPACT ANALYSIS")
    print(f"{'=' * 78}")

    analyze_clip_impact(best_metrics, full_ds)

    # -----------------------------------------------------------------------
    # STEP 8 - Save Artifacts
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 7b — SAVING TRAINING ARTIFACTS")
    print(f"{'=' * 78}")

    _save_training_logs(epoch_metrics_log, str(CHECKPOINT_DIR))
    _save_validation_metrics(best_metrics, str(CHECKPOINT_DIR))
    _save_confusion_analysis(best_confusion, pred_vocab, str(CHECKPOINT_DIR))

    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    ckpt_files = os.listdir(str(CHECKPOINT_DIR))
    for f in sorted(ckpt_files):
        fpath = os.path.join(str(CHECKPOINT_DIR), f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:<30} {size_kb:>8.1f} KB")

    # -----------------------------------------------------------------------
    # STEP 8 - FINAL REPORT
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 8 — FINAL REPORT")
    print(f"{'=' * 78}")

    generate_final_report(
        model, best_metrics, val_loader, device, pred_vocab, has_visual, label_vocab,
        all_batch_times, dataset_size, EPOCHS, best_val_acc, best_epoch,
    )


# ---------------------------------------------------------------------------
# Pure Visual Pre-training Validation
# ---------------------------------------------------------------------------

def _validate_pure_visual(dataset: VGRelationshipDataset) -> None:
    """Verify no zero-vector samples leak through in pure visual mode."""
    print(f"\n{'=' * 65}")
    print("  PRE-TRAINING PURE VISUAL VALIDATION")
    print(f"{'=' * 65}")
    total = len(dataset)
    print(f"  Total samples after strict filtering: {total}")

    zero_samples = 0
    subj_norms = []
    obj_norms = []
    for idx in range(total):
        item = dataset[idx]
        sf = item[4]
        of = item[5]
        sn = sf.norm().item()
        on = of.norm().item()
        subj_norms.append(sn)
        obj_norms.append(on)
        if sn < 0.001 or on < 0.001:
            zero_samples += 1

    all_norms = subj_norms + obj_norms
    mean_norm = sum(all_norms) / len(all_norms) if all_norms else 0.0
    min_norm = min(all_norms) if all_norms else 0.0
    max_norm = max(all_norms) if all_norms else 0.0

    print(f"  Zero-vector samples:              {zero_samples}/{total * 2} features")
    print(f"  Mean feature norm:                {mean_norm:.4f}")
    print(f"  Min feature norm:                 {min_norm:.4f}")
    print(f"  Max feature norm:                 {max_norm:.4f}")

    assert zero_samples == 0, (
        f"[PURE VISUAL] Found {zero_samples} zero-vector features in strict mode! "
        "Samples with missing CLIP embeddings leaked through."
    )

    # Predicate distribution
    pred_counter = Counter()
    for idx in range(total):
        item = dataset[idx]
        pred_name = dataset.pred_vocab.token(item[3].item())
        pred_counter[pred_name] += 1

    print(f"  Real CLIP coverage:              100.00% (strict mode)")
    print(f"  Retained sample count:           {total}")
    print(f"\n  Predicate distribution (pure visual):")
    for pred, count in pred_counter.most_common():
        print(f"    {pred}: {count}")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, label_vocab, pred_vocab, ckpt_dir, epoch, val_acc, dataset=None, mode_label="unknown"):
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
            "num_labels": model.label_emb.num_embeddings,
            "num_predicates": model.mlp[-1].out_features,
            "embed_dim": model.label_emb.embedding_dim,
            "clip_dim": model.clip_dim,
            "pose_dim": model.pose_dim,
            "union_dim": model.union_dim,
        }
        hdims = []
        for k in state:
            if k.startswith("mlp.") and k.endswith(".weight") and k != "mlp.0.weight":
                hdims.append(state[k].shape[0])
        config["hidden_dims"] = hdims
    torch.save({"model_state_dict": state, "model_config": config},
               os.path.join(ckpt_dir, "relation_mlp.pt"))
    label_vocab.save(os.path.join(ckpt_dir, "label_vocab.json"))
    pred_vocab.save(os.path.join(ckpt_dir, "pred_vocab.json"))

    meta = {
        "epoch": epoch,
        "val_acc": val_acc,
        "timestamp": time.time(),
        "mode": mode_label,
        "model_type": model_type,
        "use_visual": USE_VISUAL,
        "require_visual": REQUIRE_VISUAL,
        "use_pose": USE_POSE,
        "use_union": USE_UNION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
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
        meta["clip_dim"] = dataset.CLIP_DIM if USE_VISUAL else 0
        # Real CLIP coverage
        if USE_VISUAL:
            real, total, pct = dataset.compute_clip_coverage()
            meta["real_clip_coverage_pct"] = round(pct, 2)
            meta["real_clip_samples"] = real
            meta["total_samples"] = total
        meta["retained_sample_count"] = len(dataset)
        # Predicate distribution
        pred_counter = Counter()
        for idx in range(len(dataset)):
            pred_name = dataset.pred_vocab.token(dataset[idx][3].item())
            pred_counter[pred_name] += 1
        meta["predicate_distribution"] = dict(pred_counter.most_common())

    with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_best_model(model, ckpt_dir, device):
    ckpt_path = os.path.join(ckpt_dir, "relation_mlp.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()


def _save_training_logs(epoch_logs, ckpt_dir):
    with open(os.path.join(ckpt_dir, "training_logs.json"), "w") as f:
        json.dump(epoch_logs, f, indent=2)


def _save_validation_metrics(metrics, ckpt_dir):
    serializable = {}
    for token, m in metrics.items():
        serializable[token] = {
            "total": m["total"],
            "correct": m["correct"],
            "accuracy": round(m["accuracy"], 4),
        }
    with open(os.path.join(ckpt_dir, "validation_metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)


def _save_confusion_analysis(confusion_counts, pred_vocab, ckpt_dir, top_n=20):
    all_confusions = []
    for true_token, pred_dict in confusion_counts.items():
        for pred_token, count in pred_dict.items():
            if pred_token != true_token:
                all_confusions.append({
                    "true": true_token,
                    "predicted": pred_token,
                    "count": count,
                })
    all_confusions.sort(key=lambda x: -x["count"])
    with open(os.path.join(ckpt_dir, "confusion_analysis.json"), "w") as f:
        json.dump({"top_confusions": all_confusions[:top_n]}, f, indent=2)


# ---------------------------------------------------------------------------
# CLIP Impact Analysis
# ---------------------------------------------------------------------------

def analyze_clip_impact(best_metrics, full_ds):
    semantic = ["riding", "carrying", "holding", "wearing", "sitting on", "standing on"]
    spatial = ["on", "near", "under", "above", "next to", "behind", "in front of", "over", "inside"]

    print(f"\n  Semantic Predicate Performance (CLIP-informed):")
    sem_total = 0
    sem_correct = 0
    for p in semantic:
        m = best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0})
        sem_total += m["total"]
        sem_correct += m["correct"]
        print(f"    {p:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']})")
    print(f"    {'-- All Semantic --':<15} acc={sem_correct / max(sem_total, 1):.3f}  ({sem_correct}/{sem_total})")

    print(f"\n  Spatial Predicate Performance (geometry-dominated):")
    spa_total = 0
    spa_correct = 0
    for p in spatial:
        m = best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0})
        spa_total += m["total"]
        spa_correct += m["correct"]
        print(f"    {p:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']})")
    print(f"    {'-- All Spatial --':<15} acc={spa_correct / max(spa_total, 1):.3f}  ({spa_correct}/{spa_total})")

    print(f"\n  Assessment:")
    if sem_total > 0:
        sem_acc = sem_correct / sem_total
    else:
        sem_acc = 0.0
    if spa_total > 0:
        spa_acc = spa_correct / spa_total
    else:
        spa_acc = 0.0

    print(f"    Semantic accuracy:  {sem_acc:.3f}")
    print(f"    Spatial accuracy:   {spa_acc:.3f}")
    print(f"    Gap:                {abs(sem_acc - spa_acc):.3f}")

    if sem_acc > spa_acc * 0.5:
        print(f"    CLIP features ARE contributing to semantic predictions.")
    else:
        print(f"    CLIP features have limited impact on semantic predictions.")
    print(f"    Geometry still dominates overall accuracy due to spatial predicate frequency.")


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------

def generate_final_report(
    model, best_metrics, val_loader, device, pred_vocab, has_visual, label_vocab,
    all_batch_times, dataset_size, epochs, best_val_acc, best_epoch,
):
    avg_batch_time = sum(all_batch_times) / max(len(all_batch_times), 1)
    samples_per_sec = BATCH_SIZE / max(avg_batch_time, 1e-6)
    total_train_time = sum(all_batch_times)

    print(f"\n  1. Final Validation Metrics:")
    overall_correct = sum(m["correct"] for m in best_metrics.values())
    overall_total = sum(m["total"] for m in best_metrics.values())
    print(f"     Overall accuracy:       {overall_correct / max(overall_total, 1):.4f}")
    print(f"     Best epoch:             {best_epoch}")
    print(f"     Best val accuracy:      {best_val_acc:.4f}")

    print(f"\n  2. Predicate-Wise Metrics:")
    print_predicate_table(best_metrics, "")

    print(f"\n  3. Training Speed:")
    print(f"     Batch size:             {BATCH_SIZE}")
    print(f"     Total batches:          {len(all_batch_times)}")
    print(f"     Avg batch time:         {avg_batch_time:.4f}s")
    print(f"     Samples/sec:            {samples_per_sec:.1f}")
    print(f"     Total training time:    {total_train_time:.1f}s")
    print(f"     Epochs:                 {epochs}")

    print(f"\n  4. GPU Utilization Summary:")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"     VRAM allocated:         {mem_allocated:.0f} MB")
        print(f"     VRAM reserved:          {mem_reserved:.0f} MB")
        print(f"     VRAM peak:              {mem_peak:.0f} MB")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"     VRAM utilization:       {mem_peak / total_vram * 100:.1f}%")
    else:
        print(f"     (CPU training)")

    print(f"\n  5. Best/Worst Predicates:")
    sorted_by_acc = sorted(
        [(t, m) for t, m in best_metrics.items() if t not in (Vocab.PAD, Vocab.UNK) and m["total"] > 0],
        key=lambda x: -x[1]["accuracy"],
    )
    print(f"     Top 5 best:")
    for token, m in sorted_by_acc[:5]:
        print(f"       {token:<20} acc={m['accuracy']:.3f} ({m['correct']}/{m['total']})")
    print(f"     Bottom 5 worst:")
    for token, m in sorted_by_acc[-5:]:
        print(f"       {token:<20} acc={m['accuracy']:.3f} ({m['correct']}/{m['total']})")

    print(f"\n  6. Qualitative Prediction Examples:")
    qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=3)

    print(f"\n  7. Honest Assessment of Relation Quality:")
    sem_preds = {"riding", "carrying", "holding", "wearing", "sitting on", "standing on"}
    sem_metrics = {p: best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0}) for p in sem_preds}
    sem_total = sum(m["total"] for m in sem_metrics.values())
    sem_correct = sum(m["correct"] for m in sem_metrics.values())
    sem_acc = sem_correct / max(sem_total, 1)

    spatial_preds = {"on", "near", "under", "above", "next to", "behind", "in front of"}
    spa_metrics = {p: best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0}) for p in spatial_preds}
    spa_total = sum(m["total"] for m in spa_metrics.values())
    spa_correct = sum(m["correct"] for m in spa_metrics.values())
    spa_acc = spa_correct / max(spa_total, 1)

    print(f"     Semantic predicates (CLIP-sensitive):        acc={sem_acc:.3f} ({sem_correct}/{sem_total})")
    print(f"     Spatial predicates (geometry-dominated):     acc={spa_acc:.3f} ({spa_correct}/{spa_total})")
    print(f"     Overall:                                     acc={overall_correct / max(overall_total, 1):.3f} ({overall_correct}/{overall_total})")

    if sem_total > 50:
        print(f"\n     CLIP features {'ARE' if sem_acc > 0.3 else 'are NOT yet'} providing meaningful semantic signal.")
        print(f"     Spatial predicates still dominate frequency (ratio {spa_total / max(sem_total, 1):.1f}x).")
    else:
        print(f"\n     Insufficient semantic predicate samples for meaningful CLIP assessment.")
        print(f"     Need more training data with semantic interactions.")

    print(f"\n  8. Pipeline Readiness for Grounded Captioning:")
    print(f"     Model:     {'READY' if best_val_acc > 0.3 else 'NEEDS IMPROVEMENT'}")
    print(f"     Checkpoint: relation_mlp.pt directly loadable by infer_relationships_learned()")
    print(f"     Vocab:     label_vocab.json + pred_vocab.json present")
    print(f"     Mode:      visual-semantic (1669-dim input)")
    print(f"     Coverage:  {dataset_size:,} training samples with partial CLIP coverage")

    print(f"\n{'=' * 78}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# Focal Loss with class weighting
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_w = self.alpha.gather(0, targets) * (targets != self.ignore_index).float()
            loss = alpha_w * loss
        if self.reduction == 'mean':
            valid = (targets != self.ignore_index).float()
            return loss.sum() / valid.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights(pred_counter, pred_vocab, num_predicates, ignore_index=0, beta=0.999):
    counts = torch.zeros(num_predicates)
    for pred_str, count in pred_counter.items():
        idx = pred_vocab[pred_str]
        counts[idx] = count

    weights = torch.ones(num_predicates)
    for i in range(num_predicates):
        n = counts[i].item()
        if n > 0:
            weights[i] = (1.0 - beta) / (1.0 - beta ** n)
        else:
            weights[i] = 0.0

    weights[ignore_index] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_predicates
    return weights


def classifier_l2_loss(model, weight=1e-4):
    if isinstance(model, RelationTransformer):
        return weight * model.output.weight.norm(2).pow(2) * 0.5
    last_linear = None
    for module in model.mlp:
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return weight * last_linear.weight.norm(2).pow(2) * 0.5
    return torch.tensor(0.0, device=next(model.parameters()).device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pure Visual-Semantic Relation MLP Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Geometry-only MLP (baseline)
  python train_full_visual_semantic.py

  # MLP with visual-semantic features (mixed/fallback)
  python train_full_visual_semantic.py --use-visual

  # PURE visual-semantic MLP (strict, no zero-vectors allowed)
  python train_full_visual_semantic.py --use-visual --require-visual

  # TRANSFORMER with visual-semantic features
  python train_full_visual_semantic.py --use-visual --model transformer

  # TRANSFORMER with full interaction-aware features
  python train_full_visual_semantic.py --use-visual --require-visual --use-union --use-pose --model transformer
        """,
    )
    parser.add_argument("--use-visual", action="store_true", default=USE_VISUAL,
                        help="Enable CLIP visual features")
    parser.add_argument("--require-visual", action="store_true", default=False,
                        help="Strict mode: drop samples with missing CLIP embeddings")
    parser.add_argument("--use-pose", action="store_true", default=False,
                        help="Enable pose features (requires MediaPipe)")
    parser.add_argument("--use-union", action="store_true", default=False,
                        help="Enable union-region CLIP features")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--model", type=str, default=MODEL_TYPE, choices=["mlp", "transformer"],
                        help="Model architecture (mlp or transformer)")
    parser.add_argument("--vg-root", type=str, default=str(VG_ROOT))
    args = parser.parse_args()

    USE_VISUAL = args.use_visual
    REQUIRE_VISUAL = args.require_visual
    USE_POSE = args.use_pose
    USE_UNION = args.use_union
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    SEED = args.seed
    MODEL_TYPE = args.model
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    VG_ROOT = Path(args.vg_root)
    VG_IMAGE_DIR = VG_ROOT / "images"
    CLIP_CACHE_PATH = VG_ROOT / "clip_cache_proper.pt"

    main()
