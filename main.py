import os

import torch
from torch.utils.data import DataLoader

from models.detector import HybridDetector
from dataset.coco_detection import COCODetectionDataset, collate_fn
from dataset.coco_detection import DETECTION_CLASSES
from training.train import train_detector
from utils.seed import set_seed
from utils.visualize import visualize_predictions


# ── Paths ─────────────────────────────────────────────────────────────────────
# Edit these two lines to point at your local COCO download.
# Expected layout:
#   COCO_ROOT/train2017/                    (images)
#   COCO_ROOT/val2017/                      (images)
#   COCO_ROOT/annotations/instances_train2017.json
#   COCO_ROOT/annotations/instances_val2017.json
COCO_ROOT       = "./data/coco"
CHECKPOINT_DIR  = "./checkpoints"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 4        # safe for most GPUs; increase to 8 if memory allows
EPOCHS      = 30
LR          = 1e-4
NUM_WORKERS = 2
SEED        = 42


def _require_path(path: str, label: str) -> None:
    """Raise a clear error if a required path is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] {label} not found: {path}\n"
            f"  Download COCO 2017 and set COCO_ROOT at the top of main.py.\n"
            f"  https://cocodataset.org/#download"
        )


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Validate paths before touching the dataset ────────────────────────────
    train_img_dir = os.path.join(COCO_ROOT, "val2017")
    val_img_dir     = os.path.join(COCO_ROOT, "val2017")
    train_ann_file = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
    val_ann_file    = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

    _require_path(train_img_dir,  "COCO train images")
    _require_path(val_img_dir,    "COCO val images")
    _require_path(train_ann_file, "COCO train annotations")
    _require_path(val_ann_file,   "COCO val annotations")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset = COCODetectionDataset(
        image_dir=train_img_dir,
        annotation_file=train_ann_file,
    )
    val_dataset = COCODetectionDataset(
        image_dir=val_img_dir,
        annotation_file=val_ann_file,
    )
    print(f"  Train samples : {len(train_dataset):,}")
    print(f"  Val samples   : {len(val_dataset):,}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # 5 foreground classes: person, car, dog, cat, bicycle
    # Index 0 is reserved for background inside the model.
    model = HybridDetector(num_classes=5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    train_detector(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # ── Quick visualization after training ────────────────────────────────────
    print("\nRunning quick visualization...\n")

    checkpoint = torch.load("checkpoints/best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    for i in range(3):
        image, target = val_dataset[i]

        image = image.to(device)

        with torch.no_grad():
            pred_logits, pred_boxes = model(image.unsqueeze(0))

        visualize_predictions(
            image=image.cpu(),
            pred_logits=pred_logits[0].cpu(),
            pred_boxes=pred_boxes[0].cpu(),
            gt_boxes=target["boxes"],
            gt_labels=target["labels"],
            class_names=DETECTION_CLASSES,
        )


if __name__ == "__main__":
    main()
