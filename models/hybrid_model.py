import torch
import torch.nn as nn

from models.cnn import CNNBackbone
from models.vit import VisionTransformer
from models.gnn import GNNModel
from graph.build_graph import build_graph


class CNNViTGNN(nn.Module):
    """
    Hybrid model supporting:
    - CNN only
    - CNN + ViT
    - CNN + ViT + GNN
    """

    def __init__(
        self,
        num_classes,
        mode="cnn_vit_gnn",   # "cnn", "cnn_vit", "cnn_vit_gnn"
        cnn_out_channels=256,
        vit_embed_dim=256,
        vit_depth=4,
        vit_heads=8,
        gnn_hidden_dim=256,
        gnn_layers=2,
        feature_map_size=(28, 28)
    ):
        super().__init__()

        self.mode = mode

        # CNN backbone
        self.cnn = CNNBackbone()

        # Vision Transformer
        self.vit = VisionTransformer(
            in_channels=cnn_out_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            patch_size=2,  # important for reducing tokens
            feature_map_size=feature_map_size
        )

        # GNN
        self.gnn = GNNModel(
            in_dim=vit_embed_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers
        )

        # Classifiers
        self.cnn_classifier = nn.Linear(cnn_out_channels, num_classes)
        self.classifier = nn.Linear(gnn_hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """

        # ===== CNN =====
        cnn_features = self.cnn(x)   # (B, C, H, W)

        # ===== MODE 1: CNN ONLY =====
        if self.mode == "cnn":
            x = cnn_features.mean(dim=[2, 3])   # Global Avg Pool
            return self.cnn_classifier(x)

        # ===== ViT =====
        vit_tokens = self.vit(cnn_features)     # (B, N, D)

        # ===== MODE 2: CNN + ViT =====
        if self.mode == "cnn_vit":
            x = vit_tokens.mean(dim=1)
            return self.classifier(x)

        # ===== MODE 3: CNN + ViT + GNN =====
        adj = build_graph(vit_tokens)           # (B, N, N)
        gnn_out = self.gnn(vit_tokens, adj)     # (B, D)

        return self.classifier(gnn_out)