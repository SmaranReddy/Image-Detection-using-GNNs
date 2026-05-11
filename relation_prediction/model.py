"""
MLP relation classifier with visual-semantic support.

Architecture
------------
    [subj_label_emb | obj_label_emb | geo | subj_visual | obj_visual]
    → Linear → ReLU → Dropout
    → Linear → ReLU → Dropout
    → Linear → logits (num_predicates)

Visual features (CLIP embeddings) are optional:
    - When clip_dim=0:  input = 2*embed_dim + GEO_DIM        (geometry-only)
    - When clip_dim>0:  input = 2*embed_dim + GEO_DIM + 2*clip_dim  (visual-semantic)

The label embeddings provide coarse class information.
The CLIP visual embeddings provide appearance-based interaction cues.
The combination enables the MLP to learn semantic interactions
(e.g. riding, sitting on, carrying) rather than just spatial correlations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vg_dataset import GEO_DIM


class RelationMLP(nn.Module):
    """
    Args:
        num_labels:     Vocabulary size for subject / object labels.
        num_predicates: Number of predicate classes to predict.
        embed_dim:      Embedding dimension for label tokens.
        hidden_dims:    Sequence of hidden layer widths.
        dropout:        Dropout probability applied after each hidden ReLU.
        clip_dim:       Dimension of CLIP visual embeddings (0 = no visual features).
    """

    def __init__(
        self,
        num_labels: int,
        num_predicates: int,
        embed_dim: int = 64,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
        clip_dim: int = 0,
    ) -> None:
        super().__init__()

        self.label_emb = nn.Embedding(num_labels, embed_dim, padding_idx=0)
        self.clip_dim = clip_dim

        in_dim = 2 * embed_dim + GEO_DIM + 2 * clip_dim
        layers: list = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_predicates))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        subj_idx: torch.Tensor,   # (B,) long
        obj_idx:  torch.Tensor,   # (B,) long
        geo:      torch.Tensor,   # (B, GEO_DIM) float
        subj_feat: torch.Tensor = None,  # (B, clip_dim) float or None
        obj_feat:  torch.Tensor = None,  # (B, clip_dim) float or None
    ) -> torch.Tensor:            # (B, num_predicates)
        se = self.label_emb(subj_idx)         # (B, embed_dim)
        oe = self.label_emb(obj_idx)          # (B, embed_dim)

        components = [se, oe, geo]

        if subj_feat is not None and obj_feat is not None:
            components.append(subj_feat)
            components.append(obj_feat)

        x = torch.cat(components, dim=-1)     # (B, in_dim)
        return self.mlp(x)
