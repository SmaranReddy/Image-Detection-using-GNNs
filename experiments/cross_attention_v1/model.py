from __future__ import annotations

import torch
import torch.nn as nn

from relation_prediction.vg_dataset import (
    GEO_DIM,
    POSE_FEATURE_DIM,
    POSE_OBJECT_FEATURE_DIM,
    UNION_FEATURE_DIM,
)


class RelationMLP(nn.Module):
    """
    Experimental Cross-Attention V1

    Features are converted into modality tokens:

        subj_token
        obj_token
        geo_token
        subj_clip_token
        obj_clip_token
        union_token
        pose_token
        pose_object_token

    Then:

        MultiheadAttention
            ↓
        Mean Pool
            ↓
        Classifier

    Research hypothesis:
        - subject attends to object
        - union attends to both
        - geometry influences semantics
        - pose improves interaction reasoning
    """

    def __init__(
        self,
        num_labels: int,
        num_predicates: int,
        embed_dim: int = 64,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
        clip_dim: int = 0,
        pose_dim: int = 0,
        pose_object_dim: int = 0,
        union_dim: int = 0,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.clip_dim = clip_dim
        self.pose_dim = pose_dim
        self.pose_object_dim = pose_object_dim
        self.union_dim = union_dim
        self.hidden_dim = hidden_dim

        self.label_emb = nn.Embedding(
            num_labels,
            embed_dim,
            padding_idx=0,
        )

        # --------------------------------------------------
        # Token projections
        # --------------------------------------------------

        self.subj_proj = nn.Linear(embed_dim, hidden_dim)
        self.obj_proj = nn.Linear(embed_dim, hidden_dim)
        self.geo_proj = nn.Linear(GEO_DIM, hidden_dim)

        if clip_dim > 0:
            self.subj_clip_proj = nn.Linear(
                clip_dim,
                hidden_dim,
            )

            self.obj_clip_proj = nn.Linear(
                clip_dim,
                hidden_dim,
            )

        if union_dim > 0:
            self.union_proj = nn.Linear(
                union_dim,
                hidden_dim,
            )

        if pose_dim > 0:
            self.pose_proj = nn.Linear(
                pose_dim,
                hidden_dim,
            )

        if pose_object_dim > 0:
            self.pose_object_proj = nn.Linear(
                pose_object_dim,
                hidden_dim,
            )

        # --------------------------------------------------
        # Modality embeddings
        # --------------------------------------------------

        self.modality_emb = nn.Parameter(
            torch.randn(8, hidden_dim)
        )

        # --------------------------------------------------
        # Cross Attention
        # --------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dims[0],
                hidden_dims[1],
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dims[1],
                num_predicates,
            ),
        )

    def forward(
        self,
        subj_idx,
        obj_idx,
        geo,
        subj_feat=None,
        obj_feat=None,
        union_feat=None,
        pose_feat=None,
        pose_object_feat=None,
        geo_dropout_prob: float = 0.0,
    ):

        se = self.label_emb(subj_idx)
        oe = self.label_emb(obj_idx)

        if self.training and geo_dropout_prob > 0:
            if torch.rand(1).item() < geo_dropout_prob:
                geo = torch.zeros_like(geo)

        tokens = []

        # ------------------------------------------
        # Subject label token
        # ------------------------------------------

        tokens.append(
            self.subj_proj(se)
            + self.modality_emb[0]
        )

        # ------------------------------------------
        # Object label token
        # ------------------------------------------

        tokens.append(
            self.obj_proj(oe)
            + self.modality_emb[1]
        )

        # ------------------------------------------
        # Geometry token
        # ------------------------------------------

        tokens.append(
            self.geo_proj(geo)
            + self.modality_emb[2]
        )

        # ------------------------------------------
        # Subject CLIP token
        # ------------------------------------------

        if (
            self.clip_dim > 0
            and subj_feat is not None
        ):
            tokens.append(
                self.subj_clip_proj(subj_feat)
                + self.modality_emb[3]
            )

        # ------------------------------------------
        # Object CLIP token
        # ------------------------------------------

        if (
            self.clip_dim > 0
            and obj_feat is not None
        ):
            tokens.append(
                self.obj_clip_proj(obj_feat)
                + self.modality_emb[4]
            )

        # ------------------------------------------
        # Union token
        # ------------------------------------------

        if (
            self.union_dim > 0
            and union_feat is not None
        ):
            tokens.append(
                self.union_proj(union_feat)
                + self.modality_emb[5]
            )

        # ------------------------------------------
        # Pose token
        # ------------------------------------------

        if (
            self.pose_dim > 0
            and pose_feat is not None
        ):
            tokens.append(
                self.pose_proj(pose_feat)
                + self.modality_emb[6]
            )

        # ------------------------------------------
        # Pose-object token
        # ------------------------------------------

        if (
            self.pose_object_dim > 0
            and pose_object_feat is not None
        ):
            tokens.append(
                self.pose_object_proj(
                    pose_object_feat
                )
                + self.modality_emb[7]
            )

        # Shape:
        # [B, num_tokens, hidden_dim]

        tokens = torch.stack(tokens, dim=1)

        attended_tokens, _ = self.cross_attention(
            tokens,
            tokens,
            tokens,
        )

        pooled = attended_tokens.mean(dim=1)

        logits = self.classifier(pooled)

        return logits

