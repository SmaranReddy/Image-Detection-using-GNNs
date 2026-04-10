"""
HybridDetector: CNN + ViT + GNN object detection model.

Full tensor shape trace for default config
(B=batch, Q=20 queries, N=196 tokens, D=256 embed_dim, C=num_classes):

  Input              (B, 3,   224, 224)
  CNNBackbone        (B, 256,  28,  28)   3 × [Conv+BN+ReLU, MaxPool(2)]
  VisionTransformer  (B, 196,       256)  patch_size=2 → 14×14 = 196 tokens
  build_graph        (B, 196, 196)        sparse k-NN adjacency (k=4)
  GNNEncoder         (B, 196,       256)  per-token; no global pooling
  query_embed        (B,  20,       256)  20 learned query embeddings
  QueryDecoderLayer  (B,  20,       256)  cross-attn + dropout + FFN + dropout
  box_head           (B,  20,         4)  sigmoid → (cx, cy, w, h) in [0, 1]
  class_head         (B,  20,       C+1)  raw logits; index 0 = background
"""

from typing import Tuple

import torch
import torch.nn as nn

from models.cnn import CNNBackbone
from models.vit import VisionTransformer
from models.gnn import GCNLayer          # reused directly; GNNModel untouched
from graph.build_graph import build_graph


# ── GNN encoder (per-token output) ───────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    Graph convolutional encoder that preserves spatial structure.

    Identical to GNNModel (models/gnn.py) except the global mean-pool is
    omitted so all N token features are passed downstream to the decoder.
    GNNModel itself is not modified — the classification pipeline is unaffected.

    Shapes:
        x   : (B, N, D)
        adj : (B, N, N)  sparse k-NN adjacency from build_graph()
        out : (B, N, D)  per-token relational features
    """

    def __init__(
        self,
        in_dim:     int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(GCNLayer(in_dim if i == 0 else hidden_dim, hidden_dim))
        self.layers     = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, D)
        adj: (B, N, N)
        returns: (B, N, D)   all N token features retained
        """
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x, adj)))
        return x   # ← no x.mean(dim=1); spatial structure preserved


# ── Query decoder layer ───────────────────────────────────────────────────────

class QueryDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with cross-attention and feed-forward block.
    Self-attention among queries is omitted to keep the design minimal.

    Sub-layers (pre-norm residual pattern):
        1. Cross-attention  Q=queries, K/V=memory  → dropout → residual → LN
        2. Feed-forward     Linear → ReLU → Dropout → Linear → dropout → residual → LN

    Shapes:
        queries : (B, Q, D)
        memory  : (B, N, D)   GNN-encoded spatial tokens
        out     : (B, Q, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim:   int,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        # ── Cross-attention ───────────────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,       # (B, seq, D) convention throughout
        )
        self.norm1     = nn.LayerNorm(embed_dim)
        self.dropout1  = nn.Dropout(dropout)

        # ── Feed-forward block ────────────────────────────────────────────────
        # Bottleneck: D → ffn_dim → D   (typical ratio ffn_dim = 4 × embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2    = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,   # (B, Q, D)
        memory:  torch.Tensor,   # (B, N, D)
    ) -> torch.Tensor:           # (B, Q, D)
        """
        Each of the Q queries reads the full N-token memory via cross-attention,
        then passes through a position-wise FFN. Dropout is applied before
        each residual addition for regularisation.
        """
        # ── Cross-attention ───────────────────────────────────────────────────
        # attn_weights shape: (B, Q, N)  — each query attends over all N tokens
        attn_out, _ = self.cross_attn(query=queries, key=memory, value=memory)
        queries = self.norm1(queries + self.dropout1(attn_out))   # (B, Q, D)

        # ── Feed-forward ──────────────────────────────────────────────────────
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + self.dropout2(ffn_out))    # (B, Q, D)

        return queries


# ── Detection head ────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Produces per-query bounding box predictions and class logits.

    Box regression  : MLP → Sigmoid  →  (cx, cy, w, h)  ∈ [0, 1]
    Classification  : Linear          →  num_classes + 1 logits
                                         index 0 = background

    Shapes:
        in     : (B, Q, D)
        boxes  : (B, Q, 4)       (cx, cy, w, h), all values in [0, 1]
        logits : (B, Q, C+1)     raw scores; apply softmax/argmax at inference
    """

    def __init__(
        self,
        embed_dim:      int,
        num_classes:    int,
        box_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Two-layer MLP for box regression; sigmoid keeps output in [0, 1]
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, box_hidden_dim),
            nn.ReLU(),
            nn.Linear(box_hidden_dim, 4),
            nn.Sigmoid(),
        )

        # Linear classifier; +1 accounts for the background class at index 0
        self.class_head = nn.Linear(embed_dim, num_classes + 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, Q, D)
        returns:
            boxes  : (B, Q, 4)    (cx, cy, w, h) in [0, 1]
            logits : (B, Q, C+1)  class scores including background
        """
        boxes  = self.box_head(x).clamp(0, 1)  # (B, Q, 4) — clamp guards against fp edge cases
        logits = self.class_head(x)            # (B, Q, C+1)
        return boxes, logits


# ── Full detector ─────────────────────────────────────────────────────────────

class HybridDetector(nn.Module):
    """
    Multi-object detector: CNNBackbone → VisionTransformer → GNNEncoder
                           → QueryDecoderLayer → DetectionHead.

    Multi-object prediction is handled via Q=20 learned object queries.
    Each query independently attends to all GNN-encoded spatial tokens
    and predicts one (box, class) pair. Queries specialize over training
    without any NMS or anchor priors.

    Default tensor shapes (B = batch size):
        Input      (B,  3, 224, 224)
        CNN        (B, 256,  28,  28)
        ViT        (B, 196,       256)   patch_size=2 → 14×14 tokens
        GNN        (B, 196,       256)   per-token, sparse k-NN graph
        Queries    (B,  20,       256)   expanded from nn.Embedding
        Decoder    (B,  20,       256)   after cross-attn + FFN
        boxes      (B,  20,         4)   (cx, cy, w, h) in [0, 1]
        logits     (B,  20,       C+1)   C foreground + 1 background
    """

    def __init__(
        self,
        num_classes:     int,
        num_queries:     int   = 20,
        embed_dim:       int   = 256,
        vit_depth:       int   = 4,
        vit_heads:       int   = 8,
        gnn_layers:      int   = 2,
        decoder_ffn_dim: int   = 1024,   # 4 × embed_dim is the standard ratio
        dropout:         float = 0.1,
        feature_map_size: Tuple[int, int] = (28, 28),
        patch_size:      int   = 2,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries

        # ── CNN backbone ─────────────────────────────────────────────────────
        # Input : (B, 3, H, W)
        # Output: (B, 256, H/8, W/8)   with H=W=224 → (B, 256, 28, 28)
        self.cnn = CNNBackbone()
        cnn_out_channels = self.cnn.out_channels  # 256

        # ── Vision Transformer ────────────────────────────────────────────────
        # Input : CNN feature map  (B, 256, 28, 28)
        # Output: patch tokens     (B, N, D)   N = (28/patch_size)^2 = 196
        self.vit = VisionTransformer(
            in_channels      = cnn_out_channels,
            embed_dim        = embed_dim,
            depth            = vit_depth,
            num_heads        = vit_heads,
            patch_size       = patch_size,
            feature_map_size = feature_map_size,
        )

        # ── GNN encoder (per-token) ───────────────────────────────────────────
        # Input : vit_tokens (B, N, D) + k-NN adjacency (B, N, N)
        # Output: gnn_tokens (B, N, D)   — spatial structure preserved
        self.gnn = GNNEncoder(
            in_dim     = embed_dim,
            hidden_dim = embed_dim,
            num_layers = gnn_layers,
        )

        # ── Object query embeddings ───────────────────────────────────────────
        # Learned weight matrix: (Q, D) — each row is one query's initial state.
        # Expanded to (B, Q, D) in forward(); uses .expand() for zero memory copy.
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # ── Query decoder layer ───────────────────────────────────────────────
        # Cross-attention: queries (B, Q, D) attend to gnn_tokens (B, N, D)
        # Followed by FFN, both with dropout + residual + LayerNorm
        self.decoder = QueryDecoderLayer(
            embed_dim = embed_dim,
            num_heads = vit_heads,
            ffn_dim   = decoder_ffn_dim,
            dropout   = dropout,
        )

        # ── Detection head ────────────────────────────────────────────────────
        # box_head  : (B, Q, D) → (B, Q, 4)     (cx, cy, w, h) via sigmoid
        # class_head: (B, Q, D) → (B, Q, C+1)   includes background at index 0
        self.detection_head = DetectionHead(
            embed_dim   = embed_dim,
            num_classes = num_classes,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, 3, 224, 224)   input image batch

        Returns:
            logits : (B, num_queries, num_classes + 1)
                     Raw class scores per query. Index 0 = background.
                     Apply softmax for probabilities; argmax for predicted class.

            boxes  : (B, num_queries, 4)
                     Predicted bounding boxes in (cx, cy, w, h) format,
                     all values normalized to [0, 1] relative to image size.
        """
        B = x.shape[0]

        # ── CNN ──────────────────────────────────────────────────────────────
        cnn_feat = self.cnn(x)                             # (B, 256, 28, 28)

        # ── ViT ──────────────────────────────────────────────────────────────
        vit_tokens = self.vit(cnn_feat)                    # (B, 196, 256)

        # ── k-NN graph + GNN ─────────────────────────────────────────────────
        # build_graph: cosine similarity, k=4 neighbors → (B, 196, 196)
        # GNNEncoder:  2-layer GCN, no pooling            → (B, 196, 256)
        adj        = build_graph(vit_tokens)               # (B, 196, 196)
        # Residual from vit_tokens preserves positional encoding baked in by
        # the ViT's PositionalEncoding step.  Message-passing alone can smooth
        # spatial distinctiveness across layers; the residual guarantees that
        # the original position-aware token values are always present in the
        # features forwarded to the decoder — no new parameters required.
        gnn_tokens = self.gnn(vit_tokens, adj) + vit_tokens  # (B, 196, 256)

        # ── Object queries ────────────────────────────────────────────────────
        # .weight: (Q, D) → unsqueeze → (1, Q, D) → expand → (B, Q, D)
        # expand() is a zero-copy view; .contiguous() ensures safe downstream use
        queries = (
            self.query_embed.weight
            .unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
        )                                                  # (B, 20, 256)

        # ── Query decoder: cross-attn over GNN tokens + FFN ──────────────────
        query_feat = self.decoder(queries, gnn_tokens)     # (B, 20, 256)

        # ── Detection head ────────────────────────────────────────────────────
        boxes, logits = self.detection_head(query_feat)    # (B,20,4), (B,20,C+1)

        return logits, boxes
