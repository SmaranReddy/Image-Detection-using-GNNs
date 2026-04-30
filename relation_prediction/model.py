"""
MLP relation classifier.

Architecture
------------
    [subj_emb | obj_emb | geo_feats]  →  Linear → ReLU → Dropout
                                       →  Linear → ReLU → Dropout
                                       →  Linear → logits (num_predicates)

Input dim = 2 * embed_dim + GEO_DIM (5)
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
    """

    def __init__(
        self,
        num_labels: int,
        num_predicates: int,
        embed_dim: int = 64,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.label_emb = nn.Embedding(num_labels, embed_dim, padding_idx=0)

        in_dim = 2 * embed_dim + GEO_DIM
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
    ) -> torch.Tensor:            # (B, num_predicates)
        se = self.label_emb(subj_idx)          # (B, embed_dim)
        oe = self.label_emb(obj_idx)           # (B, embed_dim)
        x  = torch.cat([se, oe, geo], dim=-1)  # (B, in_dim)
        return self.mlp(x)
