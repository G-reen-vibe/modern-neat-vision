"""Stability regularizer for D-NEAT developmental programs.

The research bet of D-NEAT (see DECISION.md §1): a denoising regularizer
can make developmental programs stable enough for evolution. This file
contains the regularizer's loss computation.

Concept:
  - Run the developmental program twice with independent Gaussian noise
    added to the CPPN's outputs.
  - The two resulting phenotypes should be similar.
  - Minimize the KL divergence between the two phenotypes' decision
    distributions (which primitives to instantiate, where to connect).

This is a SCAFFOLD in Phase 2. The actual loss will be implemented in
Phase 4 alongside the developmental program.
"""
from __future__ import annotations
import torch


def stability_loss(genome, noise_sigma: float = 0.1, n_samples: int = 2) -> torch.Tensor:
    """Compute the stability regularizer loss.

    SCAFFOLD: returns 0 in Phase 2.
    """
    return torch.tensor(0.0)
