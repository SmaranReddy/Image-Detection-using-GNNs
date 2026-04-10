import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.
    Call once at the top of main() before any dataset or model creation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic convolutions — slightly slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
