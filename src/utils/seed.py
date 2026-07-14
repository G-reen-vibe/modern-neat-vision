"""Reproducibility utilities: seed all RNGs deterministically."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    On CPU there is no cuDNN, but we still set the deterministic flag for
    forward-compatibility with GPU runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU
    # Determinism (slight perf cost; acceptable for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Limit threads to available CPUs (we have 2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
