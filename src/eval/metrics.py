"""Metrics: accuracy, parameter count, FLOPs, inference latency.

These are computed on the trained model for the final summary table.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
from typing import Optional


@torch.no_grad()
def compute_accuracy(model: nn.Module, loader, device: str = "cpu") -> tuple[float, Optional[float]]:
    """Return (top-1, top-5) accuracy. top-5 is None if num_classes < 10."""
    model.eval()
    total, correct, top5_correct = 0, 0, 0
    num_classes = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        if num_classes == 0:
            num_classes = out.size(1)
        if num_classes >= 10:
            top5 = out.topk(min(5, num_classes), dim=1).indices
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
    top1 = correct / max(1, total)
    top5 = top5_correct / max(1, total) if num_classes >= 10 else None
    return top1, top5


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> tuple[int, int]:
    """Return (MACs, params). Uses thop. Returns (0, params) if thop fails."""
    try:
        from thop import profile
        x = torch.zeros(input_size)
        macs, params = profile(model, inputs=(x,), verbose=False)
        return int(macs), int(params)
    except Exception:
        return 0, count_parameters(model)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    device: str = "cpu",
    warmup: int = 10,
    n_runs: int = 100,
) -> dict:
    """Measure inference latency. Returns median and mean (ms)."""
    model.eval()
    x = torch.randn(input_size).to(device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return {"latency_ms_median": median, "latency_ms_mean": mean}
