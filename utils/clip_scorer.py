"""
Full CLIP image-text similarity scorer for caption reranking.

Uses the full CLIP model (vision + text encoders) to compute
semantic similarity between images and candidate captions.

This enables BLIP-2 + CLIP reranking:
1. Generate multiple caption candidates from BLIP-2
2. Score each with CLIP image-text similarity
3. Select the caption with highest semantic alignment

Architecture:
    CLIPModel (openai/clip-vit-base-patch32)
    - Vision encoder: 512-dim embedding
    - Text encoder: 512-dim embedding
    - Similarity: cosine similarity in joint embedding space
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_EMBED_DIM = 512


class CLIPSimilarityScorer:
    """
    Full CLIP-based image-text similarity scorer.

    Computes cosine similarity between image embeddings and text embeddings
    in CLIP's joint visual-semantic embedding space.

    Used for caption reranking: select the caption that is most semantically
    aligned with the image according to CLIP.
    """

    _model: Optional[CLIPModel] = None
    _processor: Optional[CLIPProcessor] = None
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
        print("[CLIPSimilarityScorer] Loading full CLIP model (vision + text) …")
        cls._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        cls._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls._model.to(cls._device)
        cls._model.eval()
        print(f"[CLIPSimilarityScorer] Ready on {cls._device}")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLIP vision embedding for an image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor (CLIP_EMBED_DIM,) — L2-normalized image embedding
        """
        inputs = self._processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._model.get_image_features(**inputs)
        return F.normalize(outputs.pooler_output[0], dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Extract CLIP text embedding for a caption.

        Args:
            text: Caption string

        Returns:
            Tensor (CLIP_EMBED_DIM,) — L2-normalized text embedding
        """
        inputs = self._processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self._model.get_text_features(**inputs)
        return F.normalize(outputs.pooler_output[0], dim=-1)


    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Extract CLIP text embeddings for multiple captions (batched).

        Args:
            texts: List of caption strings

        Returns:
            Tensor (num_texts, CLIP_EMBED_DIM) — L2-normalized text embeddings
        """
        if not texts:
            return torch.empty(0, CLIP_EMBED_DIM, device=self.device)

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self._model.get_text_features(**inputs)
        return F.normalize(outputs.pooler_output, dim=-1)

    @torch.no_grad()
    def compute_similarity(
        self,
        image: Image.Image,
        caption: str,
    ) -> float:
        """
        Compute CLIP image-text similarity for a single caption.

        Args:
            image: PIL Image
            caption: Caption string

        Returns:
            Similarity score (cosine similarity, range: [-1, 1])
        """
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(caption)
        return float(image_emb @ text_emb)

    @torch.no_grad()
    def compute_similarities(
        self,
        image: Image.Image,
        captions: List[str],
    ) -> List[float]:
        """
        Compute CLIP image-text similarity for multiple captions.

        More efficient than calling compute_similarity in a loop
        because text encoding is batched.

        Args:
            image: PIL Image
            captions: List of caption strings

        Returns:
            List of similarity scores, one per caption
        """
        if not captions:
            return []

        image_emb = self.encode_image(image)
        text_embs = self.encode_texts(captions)

        similarities = image_emb @ text_embs.T
        return [float(s) for s in similarities]

    @torch.no_grad()
    def rerank_captions(
        self,
        image: Image.Image,
        captions: List[str],
        return_scores: bool = False,
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Rerank captions by CLIP image-text similarity.

        Args:
            image: PIL Image
            captions: List of candidate captions
            return_scores: If True, return all scores

        Returns:
            Tuple of:
            - best_caption: The caption with highest similarity
            - ranked_list: List of (caption, score) sorted by score descending
        """
        if not captions:
            return "", []

        scores = self.compute_similarities(image, captions)

        scored = list(zip(captions, scores))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        best_caption = ranked[0][0]

        return best_caption, ranked


_clip_scorer: Optional[CLIPSimilarityScorer] = None


def get_clip_scorer() -> CLIPSimilarityScorer:
    """Get or create the singleton CLIP similarity scorer."""
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = CLIPSimilarityScorer()
    return _clip_scorer


def clip_rerank_captions(
    image: Image.Image,
    captions: List[str],
) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Convenience function: rerank captions by CLIP similarity.

    Args:
        image: PIL Image
        captions: List of candidate captions

    Returns:
        (best_caption, ranked_list) where ranked_list is [(caption, score), ...]
    """
    scorer = get_clip_scorer()
    return scorer.rerank_captions(image, captions, return_scores=True)
