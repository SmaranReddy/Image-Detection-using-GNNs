"""
Inference and sanity-check entry point for HybridDetector.

Three modes — pick whichever matches what you have:

  1. Full val-set sanity check  (needs COCO data + checkpoint)
        python test.py --mode val

  2. Single-image inference     (needs one image + checkpoint)
        python test.py --mode image --image path/to/image.jpg

  3. Smoke test                 (no data needed; random tensor, untrained weights)
        python test.py --mode smoke

Checkpoint is loaded from CHECKPOINT_PATH below (edit as needed).
"""

import argparse
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from dataset.coco_detection import (
    COCODetectionDataset,
    DETECTION_CLASSES,
    collate_fn,
)
from models.detector import HybridDetector
from utils.checkpoint import load_checkpoint
from utils.sanity_check import sanity_check_detector
from utils.visualize import visualize_predictions


# ── Edit these paths ──────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./checkpoints/best.pt"   # try "last.pt" if best.pt is absent
COCO_ROOT       = "./data/coco"
NUM_CLASSES     = 5
IMAGE_SIZE      = 224
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _load_model(device: torch.device) -> HybridDetector:
    model = HybridDetector(num_classes=NUM_CLASSES).to(device)

    if os.path.isfile(CHECKPOINT_PATH):
        epoch, loss = load_checkpoint(model, CHECKPOINT_PATH, device=device)
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}  (epoch {epoch}, loss {loss:.4f})")
    else:
        print(f"[WARNING] No checkpoint found at '{CHECKPOINT_PATH}'. Using random weights.")

    model.eval()
    return model


# ── Mode 1: full sanity check on COCO val set ─────────────────────────────────

def run_val(device: torch.device) -> None:
    val_img_dir = os.path.join(COCO_ROOT, "val2017")
    val_ann     = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

    if not os.path.isdir(val_img_dir) or not os.path.isfile(val_ann):
        print(
            f"[ERROR] COCO val data not found.\n"
            f"  Expected image dir : {val_img_dir}\n"
            f"  Expected annotation: {val_ann}\n"
            f"  Download from https://cocodataset.org/#download and update COCO_ROOT."
        )
        return

    print("Loading COCO val dataset...")
    val_dataset = COCODetectionDataset(image_dir=val_img_dir, annotation_file=val_ann)
    val_loader  = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    print(f"  {len(val_dataset):,} val samples")

    model = _load_model(device)
    sanity_check_detector(model, val_loader, device, DETECTION_CLASSES)

    # Visualize first 3 val images
    print("Visualizing first 3 val samples...\n")
    for i in range(min(3, len(val_dataset))):
        image, target = val_dataset[i]
        with torch.no_grad():
            logits, boxes = model(image.unsqueeze(0).to(device))

        save_path = f"./val_pred_{i}.png"
        visualize_predictions(
            image       = image,
            pred_logits = logits[0].cpu(),
            pred_boxes  = boxes[0].cpu(),
            gt_boxes    = target["boxes"],
            gt_labels   = target["labels"],
            class_names = DETECTION_CLASSES,
            save_path   = save_path,
        )
        print(f"  Saved: {save_path}")


# ── Mode 2: single-image inference ────────────────────────────────────────────

def run_image(image_path: str, device: torch.device) -> None:
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    model = _load_model(device)

    img_pil    = Image.open(image_path).convert("RGB")
    image      = _transform(img_pil)                        # (3, 224, 224)
    dummy_gt   = {
        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,),   dtype=torch.long),
    }

    with torch.no_grad():
        logits, boxes = model(image.unsqueeze(0).to(device))

    save_path = "./pred_output.png"
    visualize_predictions(
        image       = image,
        pred_logits = logits[0].cpu(),
        pred_boxes  = boxes[0].cpu(),
        gt_boxes    = dummy_gt["boxes"],
        gt_labels   = dummy_gt["labels"],
        class_names = DETECTION_CLASSES,
        save_path   = save_path,
    )
    print(f"Prediction saved to: {save_path}")

    probs     = torch.softmax(logits[0].cpu().float(), dim=-1)
    fg_probs  = probs[:, 1:]
    conf, cls = fg_probs.max(dim=-1)
    above     = conf > 0.3
    print(f"\nDetections above 0.3 confidence: {int(above.sum())}")
    for i in above.nonzero(as_tuple=True)[0].tolist():
        cname = DETECTION_CLASSES[int(cls[i])] if int(cls[i]) < len(DETECTION_CLASSES) else f"cls{cls[i]}"
        print(f"  {cname:12s}  conf={conf[i]:.3f}  box={boxes[0][i].cpu().tolist()}")


# ── Mode 3: smoke test (no real data required) ────────────────────────────────

def run_smoke(device: torch.device) -> None:
    """
    Verifies the forward pass runs end-to-end with random input and random weights.
    Checks that output shapes and value ranges are correct.
    """
    print("[Smoke Test] Running forward pass with random input...")
    model = HybridDetector(num_classes=NUM_CLASSES).to(device)
    model.eval()

    B = 2
    dummy = torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    with torch.no_grad():
        logits, boxes = model(dummy)

    # Shape checks
    assert logits.shape == (B, 20, NUM_CLASSES + 1), \
        f"Expected logits (B,20,{NUM_CLASSES+1}), got {logits.shape}"
    assert boxes.shape == (B, 20, 4), \
        f"Expected boxes (B,20,4), got {boxes.shape}"

    # Value range check
    box_ok = (boxes >= 0.0).all() and (boxes <= 1.0).all()

    print(f"  logits shape : {tuple(logits.shape)}  OK")
    print(f"  boxes  shape : {tuple(boxes.shape)}   OK")
    print(f"  boxes in [0,1]: {box_ok}")

    if not box_ok:
        bad = int(((boxes < 0) | (boxes > 1)).any(dim=-1).sum().item())
        print(f"  [WARNING] {bad} queries have out-of-range box coordinates.")

    print("\n[Smoke Test] Pipeline is functional.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HybridDetector test script")
    parser.add_argument(
        "--mode", choices=["val", "image", "smoke"], default="smoke",
        help="val=COCO sanity check, image=single image, smoke=pipeline check",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image file (required for --mode image)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device: 'cpu' or 'cuda' (auto-detected if omitted)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}\n")

    if args.mode == "val":
        run_val(device)
    elif args.mode == "image":
        if not args.image:
            parser.error("--image is required with --mode image")
        run_image(args.image, device)
    else:
        run_smoke(device)


if __name__ == "__main__":
    main()
