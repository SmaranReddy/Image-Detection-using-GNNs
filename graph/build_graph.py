import torch


def build_graph(token_embeddings, k=4):
    """
    Build adjacency matrix using k-NN in feature space.

    Args:
        token_embeddings: (B, N, D)
        k: number of neighbors per node

    Returns:
        adjacency matrices: (B, N, N)
    """
    B, N, D = token_embeddings.shape
    adj_matrices = []

    for b in range(B):
        x = token_embeddings[b]  # (N, D)

        # Pairwise cosine similarity
        norm = x / x.norm(dim=1, keepdim=True)
        sim = torch.matmul(norm, norm.T)  # (N, N)

        # Top-k neighbors
        _, indices = torch.topk(sim, k=k + 1, dim=-1)

        adj = torch.zeros(N, N, device=x.device)
        for i in range(N):
            adj[i, indices[i]] = 1

        adj.fill_diagonal_(1)  # self-loops
        adj_matrices.append(adj)

    return torch.stack(adj_matrices)
