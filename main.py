import os
import random

import torch
from torch.utils.data import DataLoader, Subset

from models.detector import HybridDetector
from dataset.coco_detection import COCODetectionDataset, collate_fn
from dataset.coco_detection import DETECTION_CLASSES
from training.train import train_detector
from utils.seed import set_seed
from utils.visualize import visualize_predictions
from utils.yolo_detector import load_model, run_inference, format_detections
from utils.causal_caption import infer_relationships
from utils.blip_captioner import generate_blip_caption


# ── Paths ─────────────────────────────────────────────────────────────────────
# Edit this to point at your local COCO download.
# Expected layout:
#   COCO_ROOT/train2017/                    (images)
#   COCO_ROOT/annotations/instances_train2017.json
COCO_ROOT       = "./data/coco"
CHECKPOINT_DIR  = "./checkpoints"

# ── Dataset split sizes ───────────────────────────────────────────────────────
TRAIN_SIZE = 16_000
VAL_SIZE   = 2_000
TEST_SIZE  = 2_000

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 4        # safe for most GPUs; increase to 8 if memory allows
EPOCHS      = 30
LR          = 1e-4
NUM_WORKERS = 2
SEED        = 42


def _require_path(path: str, label: str) -> None:
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

    # ── Validate paths ────────────────────────────────────────────────────────
    img_dir  = os.path.join(COCO_ROOT, "train2017")
    ann_file = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")

    _require_path(img_dir,  "COCO train2017 images")
    _require_path(ann_file, "COCO train annotations")

    # ── Load full dataset, then carve reproducible 16k / 2k / 2k subsets ─────
    print("Loading dataset...")
    full_dataset = COCODetectionDataset(image_dir=img_dir, annotation_file=ann_file)
    print(f"  Total annotated images available: {len(full_dataset):,}")

    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(full_dataset) < total_needed:
        raise RuntimeError(
            f"Dataset has only {len(full_dataset):,} images but "
            f"{total_needed:,} were requested."
        )

    rng = random.Random(SEED)
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)
    indices = indices[:total_needed]

    train_dataset = Subset(full_dataset, indices[:TRAIN_SIZE])
    val_dataset   = Subset(full_dataset, indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE])
    test_dataset  = Subset(full_dataset, indices[TRAIN_SIZE + VAL_SIZE:])

    print(f"  Train : {len(train_dataset):,}")
    print(f"  Val   : {len(val_dataset):,}")
    print(f"  Test  : {len(test_dataset):,}")

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
    test_loader = DataLoader(
        test_dataset,
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

    # Load YOLOv11 once; reuse across all inference calls.
    yolo_model = load_model()

    for i in range(3):
        image, target = test_dataset[i]

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

        # YOLO detection → causal caption pipeline.
        # To use the learned relation model instead of heuristic rules:
        #   from relation_prediction import infer_relationships_learned
        #   relations = infer_relationships_learned(detections)
        raw        = run_inference(yolo_model, image.cpu())
        detections = format_detections(raw)
        relations  = infer_relationships(detections)
        caption    = generate_blip_caption(image.cpu(), detections, relations)
        print(f"[image {i}] {caption}")


if __name__ == "__main__":
    main()
