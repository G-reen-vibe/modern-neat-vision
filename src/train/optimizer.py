"""Optimizer + scheduler factory.

We use:
  - AdamW with cosine LR schedule and linear warmup.
  - Weight decay 5e-4 (standard CIFAR setting).
  - Label smoothing 0.1.

This matches the recipe used by DeiT / ConvNeXt on CIFAR.
"""
from __future__ import annotations
import math
import torch
import torch.optim as optim
from typing import Callable


def build_optimizer(model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 5e-4):
    """AdamW optimizer. Bias/BN params excluded from weight decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=(0.9, 0.999), eps=1e-8,
    )


class CosineWarmupScheduler:
    """Linear warmup + cosine decay. Step-based, not epoch-based.

    Usage:
      sched = CosineWarmupScheduler(opt, warmup_steps, total_steps)
      for step in range(total_steps):
          sched.step()  # call BEFORE optimizer.step() if warmup is positive
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup:
            frac = self.step_count / self.warmup
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base * frac
        else:
            frac = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            frac = min(1.0, max(0.0, frac))
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                cos = 0.5 * (1 + math.cos(math.pi * frac))
                g["lr"] = self.min_lr + (base - self.min_lr) * cos

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]
