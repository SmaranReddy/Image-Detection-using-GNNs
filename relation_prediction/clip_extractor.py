"""
Frozen CLIP visual feature extractor for object crops.

Provides visual appearance embeddings that enable the relation MLP
to move beyond geometry-only prediction toward semantic interaction
understanding.

Architecture:
    CLIPVisionModel (frozen) → 512-dim visual embedding per crop
    (openai/clip-vit-base-patch32)

Usage:
    extractor = CLIPExtractor()
    emb = extractor.extract_crop(image_pil, box_xyxy)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor


CLIP_DIM = 768  # openai/clip-vit-base-patch32 vision encoder hidden size
                # NOTE: this is CLIPVisionModel.pooler_output (the 768-d
                # pre-projection hidden state), NOT the 512-d joint
                # image-text embedding produced by CLIPModel.get_image_features.


def _clamp_box(box, img_w: int, img_h: int):
    """Clip a box to the image and reject degenerate ones.

    VG annotation boxes routinely run past the image edge. PIL's ``crop`` pads
    the overflow with black instead of clipping, so an unclamped crop feeds the
    encoder a partly-black patch. Both the single-crop and the batched path go
    through here so training-time and inference-time crops are built
    identically; before this, only the single-crop path had any box handling at
    all, and it clamped nothing.

    Returns the clipped (x1, y1, x2, y2) tuple, or None if the box is empty.
    """
    x1 = max(0.0, min(float(box[0]), float(img_w)))
    y1 = max(0.0, min(float(box[1]), float(img_h)))
    x2 = max(0.0, min(float(box[2]), float(img_w)))
    y2 = max(0.0, min(float(box[3]), float(img_h)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1, y1, x2, y2)


class CLIPExtractor:
    """
    Lightweight wrapper around frozen CLIP vision encoder.

    Caches the model and processor as class-level singletons so they
    are loaded only once per process — safe to create in __getitem__.
    """

    _model: Optional[CLIPVisionModel] = None
    _processor: Optional[CLIPImageProcessor] = None
    _device: Optional[torch.device] = None

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._lazy_load()

    @classmethod
    def _lazy_load(cls) -> None:
        if cls._model is not None:
            return
        print("[CLIPExtractor] Loading CLIP vision encoder (frozen) …")
        cls._processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        cls._model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._model.to(cls._device)
        cls._model.eval()
        print(f"[CLIPExtractor] Ready on {cls._device}")

    @torch.no_grad()
    def extract_crop(self, image: Image.Image, box: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Extract CLIP embedding for a single crop.

        Args:
            image: Full PIL image (RGB).
            box:   Crop region (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Tensor (CLIP_DIM,) — L2-normalised CLIP visual embedding.
        """
        box = _clamp_box(box, image.width, image.height)
        if box is None:
            print(f"[CLIPExtractor] WARNING: degenerate crop box, box={box}")
            return torch.zeros(CLIP_DIM)
        crop = image.crop(box)
        if crop.width == 0 or crop.height == 0:
            print(f"[CLIPExtractor] WARNING: empty crop after PIL crop, box={box}")
            return torch.zeros(CLIP_DIM)
        inputs = self._processor(images=crop, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        emb = outputs.pooler_output[0]  # (CLIP_DIM,)
        return F.normalize(emb, dim=-1).cpu()

    @torch.no_grad()
    def extract_crops(
        self,
        image: Image.Image,
        boxes: List[Tuple[float, float, float, float]],
    ) -> torch.Tensor:
        """
        Extract CLIP embeddings for multiple crops from the same image.

        Batches the processor call for efficiency.

        Args:
            image: Full PIL image (RGB).
            boxes: List of (x1, y1, x2, y2) pixel-coordinate boxes.

        Returns:
            Tensor (num_boxes, CLIP_DIM) — L2-normalised embeddings.
        """
        # Same clamping and same degenerate-box handling as extract_crop: this
        # is the path that builds the TRAINING cache while extract_crop is the
        # path used at INFERENCE, so any divergence here is a train/inference
        # feature mismatch. Degenerate boxes yield a zero vector in both, which
        # is what require_visual filtering looks for.
        clamped = [_clamp_box(b, image.width, image.height) for b in boxes]
        keep = [i for i, b in enumerate(clamped) if b is not None]
        embs = torch.zeros(len(boxes), CLIP_DIM)
        if not keep:
            return embs
        crops = [image.crop(clamped[i]) for i in keep]
        embs[keep] = self.encode_crops(crops)
        return embs

    @torch.no_grad()
    def encode_crops(self, crops: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of already-cropped PIL images.

        The single lowest-level entry point: extract_crop (inference),
        extract_crops (dataset) and build_clip_cache.py (the offline GPU cache)
        all bottom out here, so the preprocessing and the L2 normalisation
        cannot drift between the features a model is trained on and the ones it
        is served. The cache builder needs it separately because it batches
        crops from *different* images together to keep the GPU busy, which the
        per-image signature above cannot express.

        Returns (len(crops), CLIP_DIM), L2-normalised, on CPU.
        """
        if not crops:
            return torch.zeros(0, CLIP_DIM)
        inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        return F.normalize(outputs.pooler_output, dim=-1).cpu()

    @torch.no_grad()
    def extract_union_embedding(
        self,
        image: Image.Image,
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> torch.Tensor:
        """
        Extract CLIP embedding for the union region covering two boxes.

        The union region captures interaction context between two objects
        (e.g. contact, posture, support relationships) that isolated crops miss.

        Args:
            image: Full PIL image (RGB).
            box_a: First bounding box (x1, y1, x2, y2) in pixel coordinates.
            box_b: Second bounding box (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Tensor (CLIP_DIM,) — L2-normalised CLIP visual embedding.
        """
        union_box = (
            min(box_a[0], box_b[0]),
            min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]),
            max(box_a[3], box_b[3]),
        )
        return self.extract_crop(image, union_box)

    @staticmethod
    def to_union_key(image_id, subj_object_id, obj_object_id) -> str:
        """Cache key for the union region spanning a specific ordered pair.

        Union features used to live in a plain Python list indexed by sample
        position, rebuilt in-process on every run and never written to disk.
        That coupled the features to the *order* of ``dataset.samples``, so any
        change that reorders or refilters samples (a different predicate
        scheme, a different min_pred_count) would silently pair every union
        embedding with the wrong label. Keying on the object ids that actually
        define the region makes the association explicit and order-independent.

        The pair is ordered: (subj, obj) and (obj, subj) span the same pixels
        but are different samples, and keeping them distinct keeps the key a
        faithful name for "the union feature of this sample".
        """
        return f"{image_id}_union_{subj_object_id}_{obj_object_id}"

    @classmethod
    def to_embedding_key(
        cls, image_id: int, object_id: Optional[int] = None, box: Optional[Tuple] = None
    ) -> str:
        """Generate a deterministic cache key for an object instance."""
        if object_id is not None:
            return f"{image_id}_obj_{object_id}"
        if box is not None:
            return f"{image_id}_box_{box[0]:.1f}_{box[1]:.1f}_{box[2]:.1f}_{box[3]:.1f}"
        return f"{image_id}"
