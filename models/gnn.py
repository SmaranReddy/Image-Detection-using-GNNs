import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    Basic Graph Convolution Layer
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x: (B, N, D)
        adj: (B, N, N)
        """
        deg = adj.sum(dim=-1, keepdim=True)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float("inf")] = 0

        adj_norm = adj * deg_inv
        out = torch.bmm(adj_norm, x)
        out = self.linear(out)
        return out


class GNNModel(nn.Module):
    """
    GNN for relational reasoning on ViT tokens
    """
    def __init__(self, in_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                GCNLayer(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim
                )
            )

        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        """
        x: (B, N, D)
        adj: (B, N, N)
        returns: (B, D)
        """
        for layer in self.layers:
            x = self.activation(layer(x, adj))

        # Global pooling
        x = x.mean(dim=1)
        return x
