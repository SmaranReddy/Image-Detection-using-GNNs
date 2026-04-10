import os
import json
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# Foreground class names in detection order.
# Label 0 is reserved for background in the model.
# These classes map to labels 1–5 in the returned targets.
DETECTION_CLASSES: List[str] = ["person", "car", "dog", "cat", "bicycle"]

# class name → 1-based label index  (0 = background, reserved for model)
CLASS_TO_LABEL: Dict[str, int] = {
    cls: idx + 1 for idx, cls in enumerate(DETECTION_CLASSES)
}

# ImageNet normalization statistics
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class COCODetectionDataset(Dataset):
    """
    COCO detection dataset filtered to a fixed set of five classes.

    Parsing is done from raw JSON (no pycocotools dependency).
    Images with no qualifying annotations are excluded entirely.

    Returns per sample:
        image  : Tensor (3, 224, 224)  — ImageNet-normalized RGB
        target : dict
            "boxes"  → FloatTensor (N, 4)  [x_min, y_min, x_max, y_max] in [0, 1]
            "labels" → LongTensor  (N,)    1-indexed foreground label (0 = background)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        image_size: int = 224,
    ) -> None:
        """
        Args:
            image_dir:       Directory that contains COCO image files.
            annotation_file: Path to instances_train2017.json or instances_val2017.json.
            image_size:      Target square size for all images after resize.
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        # ── Map COCO category_id → our 1-based label ────────────────────────
        coco_id_to_label: Dict[int, int] = {}
        for cat in coco["categories"]:
            if cat["name"] in CLASS_TO_LABEL:
                coco_id_to_label[cat["id"]] = CLASS_TO_LABEL[cat["name"]]

        # ── image_id → image metadata ────────────────────────────────────────
        id_to_image: Dict[int, dict] = {img["id"]: img for img in coco["images"]}

        # ── Group valid annotations by image_id ──────────────────────────────
        anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann["category_id"] not in coco_id_to_label:
                continue
            img_id = ann["image_id"]
            anns_by_image.setdefault(img_id, []).append({
                "bbox":  ann["bbox"],                         # COCO: [x, y, w, h] pixels
                "label": coco_id_to_label[ann["category_id"]],
            })

        # ── Build sample list; skip images with no valid annotations ─────────
        self.samples: List[dict] = []
        for img_id, anns in anns_by_image.items():
            if img_id not in id_to_image:
                continue
            meta = id_to_image[img_id]
            self.samples.append({
                "file_name":   meta["file_name"],
                "orig_width":  meta["width"],
                "orig_height": meta["height"],
                "annotations": anns,
            })

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        sample = self.samples[idx]

        # ── Load image and resize to target size ─────────────────────────────
        img_path = os.path.join(self.image_dir, sample["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # ── Scale factors: original pixel space → resized pixel space ────────
        # Boxes are first mapped to resized (image_size × image_size) pixel
        # coordinates, clipped to valid image bounds, then normalized to [0, 1]
        # by dividing by image_size.  Two explicit steps avoids the subtle error
        # of clipping at (0, 1) before accounting for the resize ratio.
        scale_x = self.image_size / sample["orig_width"]
        scale_y = self.image_size / sample["orig_height"]

        # ── Convert and filter annotations ───────────────────────────────────
        boxes:  List[List[float]] = []
        labels: List[int]         = []

        for ann in sample["annotations"]:
            x, y, w, h = ann["bbox"]  # COCO top-left origin, pixel units

            # Step 1 — map to resized pixel space and clip to image boundary
            x_min_px = max(0.0,                   x * scale_x)
            y_min_px = max(0.0,                   y * scale_y)
            x_max_px = min(float(self.image_size), (x + w) * scale_x)
            y_max_px = min(float(self.image_size), (y + h) * scale_y)

            # Skip degenerate boxes (zero area after clipping)
            if x_max_px <= x_min_px or y_max_px <= y_min_px:
                continue

            # Step 2 — normalize to [0, 1] relative to the resized image
            boxes.append([
                x_min_px / self.image_size,
                y_min_px / self.image_size,
                x_max_px / self.image_size,
                y_max_px / self.image_size,
            ])
            labels.append(ann["label"])

        # ── Build image tensor ───────────────────────────────────────────────
        image_tensor = TF.to_tensor(image)        # (3, H, W), values in [0, 1]
        image_tensor = self.normalize(image_tensor)

        # ── Build target tensors ─────────────────────────────────────────────
        if boxes:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)  # (N, 4)
            labels_tensor = torch.tensor(labels, dtype=torch.long)     # (N,)
        else:
            # Rare: all annotations were degenerate after clipping
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.long)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image_tensor, target


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max).

    Both formats use the same coordinate space (e.g. normalized [0, 1]).
    Use this when comparing model outputs against dataset ground-truth boxes.

    Args:
        boxes: Tensor (..., 4)  in (cx, cy, w, h) format

    Returns:
        Tensor (..., 4)  in (x_min, y_min, x_max, y_max) format
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([
        cx - 0.5 * w,   # x_min
        cy - 0.5 * h,   # y_min
        cx + 0.5 * w,   # x_max
        cy + 0.5 * h,   # y_max
    ], dim=-1)


def collate_fn(
    batch: List[Tuple[torch.Tensor, dict]],
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Custom collate function required by DataLoader because detection targets
    have variable numbers of boxes per image (cannot be stacked into a tensor).

    Usage:
        DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    Returns:
        images:  Tensor (B, 3, H, W)  — stacked image batch
        targets: List[dict]           — one target dict per image, length B
    """
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
