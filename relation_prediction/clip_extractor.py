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
        crop = image.crop(box)
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
        crops = [image.crop(b) for b in boxes]
        inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
        outputs = self._model(**inputs)
        embs = outputs.pooler_output  # (num_boxes, CLIP_DIM)
        return F.normalize(embs, dim=-1).cpu()

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
