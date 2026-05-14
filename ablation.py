"""
ablation.py — Four-stage ablation study for the image understanding pipeline.

Stage A: BLIP captioning only
Stage B: BLIP + YOLO object detection (COCO classes)
Stage C: BLIP + YOLO + MLP-based learned relationship prediction (Visual Genome)
Stage D: Final output with evidence gating / hallucination suppression
"""

import os
import sys
from pathlib import Path

from PIL import Image

from utils.yolo_detector import load_model as _load_yolo, run_inference, format_detections
from utils.blip_captioner import generate_blip_caption, gate_caption
from relation_prediction.predict import infer_relationships_learned


IMAGE_DIR        = "test_images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# YOLO singleton — loaded once on first call to detect_objects()
# ---------------------------------------------------------------------------

_yolo_model = None


def _ensure_yolo() -> None:
    global _yolo_model
    if _yolo_model is None:
        print("[ablation] Loading YOLOv11 (COCO classes) …")
        _yolo_model = _load_yolo()
        print("[ablation] YOLO ready.")


def detect_objects(image: Image.Image) -> list:
    """Run YOLOv11 on *image* and return CLIP-verified COCO detections."""
    _ensure_yolo()
    raw = run_inference(_yolo_model, image)
    detections = format_detections(raw)
    from utils.detection_verifier import verify_detections
    return verify_detections(detections, image, debug=False)


def _to_pil(image) -> Image.Image:
    """Convert torch tensor or PIL to PIL RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    from torchvision import transforms as T
    return T.ToPILImage()(image.float().cpu().clamp(0, 1))


# ---------------------------------------------------------------------------
# Per-image ablation
# ---------------------------------------------------------------------------

def run_ablation(image_path: str) -> None:
    image_name = os.path.basename(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[skip] {image_name}: cannot open image ({exc})\n")
        return

    print(f"=== {image_name} ===\n")

    # ------------------------------------------------------------------
    # Stage A — BLIP captioning only (no detection, no relation context)
    # ------------------------------------------------------------------
    try:
        caption_A = generate_blip_caption(image, [], [])
    except Exception as exc:
        caption_A = f"[error: {exc}]"

    print("A (BLIP only):")
    print(caption_A)
    print()

    # ------------------------------------------------------------------
    # Stage B — BLIP + YOLO object detection
    # YOLOv11 runs on 80 COCO classes; detections are passed as grounding
    # context to the BLIP caption generator.
    # ------------------------------------------------------------------
    try:
        detections = detect_objects(image)
        caption_B  = generate_blip_caption(image, detections, [])
    except Exception as exc:
        detections = []
        caption_B  = f"[error: {exc}]"

    print("B (+ YOLO objects):")
    print(caption_B)
    print()

    # ------------------------------------------------------------------
    # Stage C — BLIP + YOLO + MLP-based learned relationship prediction
    # The MLP was trained on Visual Genome and predicts spatial/semantic
    # predicates from a filtered set of ~16–20 COCO-aligned relation types.
    # When using a visual-semantic model, the image is passed for CLIP
    # feature extraction from object crops.
    # ------------------------------------------------------------------
    try:
        relations = infer_relationships_learned(
            detections, image=image,
        )
        caption_C = generate_blip_caption(image, detections, relations)
    except Exception as exc:
        relations = []
        caption_C = f"[error: {exc}]"

    print("C (+ MLP learned relations):")
    print(caption_C)
    print()

    # ------------------------------------------------------------------
    # Stage D — Final output with evidence gating / hallucination suppression
    # gate_caption hedges low-confidence mentions and replaces captions that
    # contain too many unsupported entity claims with a safe fallback.
    # ------------------------------------------------------------------
    try:
        caption_D = gate_caption(caption_C, detections)
    except Exception:
        caption_D = caption_C

    print("D (final validated output):")
    print(caption_D)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    image_dir = Path(IMAGE_DIR)

    if not image_dir.is_dir():
        print(f"[error] Image directory not found: {image_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"[error] No images found in {image_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Running ablation on {len(image_paths)} image(s) from '{image_dir}/' …\n")

    for path in image_paths:
        run_ablation(str(path))


if __name__ == "__main__":
    main()
