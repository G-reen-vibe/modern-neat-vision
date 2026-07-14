"""Greedy complexification search with random policy.

This is the baseline: at each step, try K random growth operations, keep
the best. No learning. The learned policy (Round 28+) should beat this.
"""
from __future__ import annotations
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import (
    initial_graph, apply_operation, graph_to_phenotype,
    graph_features, OPS, GrowthGraph,
)
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer


def evaluate_graph(graph: GrowthGraph, train_loader, val_loader,
                   num_classes: int, in_channels: int, image_size: int,
                   epochs: int = 2) -> Tuple[float, int, float]:
    """Evaluate a graph. Returns (accuracy, param_count, time_s)."""
    t0 = time.time()
    p = graph_to_phenotype(graph)
    if p is None or not p.is_valid():
        return 0.0, 0, time.time() - t0
    try:
        model = compile_phenotype(p, in_channels=in_channels,
                                  num_classes=num_classes, image_size=image_size)
        # Quick forward-pass validation
        x = torch.randn(2, in_channels, image_size, image_size)
        model(x)
        params = sum(pp.numel() for pp in model.parameters())
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
            warmup_epochs=1, total_epochs=epochs,
            label_smoothing=0.1, grad_clip=1.0, device="cpu",
        )
        result = trainer.fit(epochs, logger=None, eval_every=1)
        return result["best_acc"], params, time.time() - t0
    except Exception:
        return 0.0, 0, time.time() - t0


def greedy_search_random(
    train_loader, val_loader, num_classes: int, in_channels: int, image_size: int,
    n_steps: int = 5, candidates_per_step: int = 3, epochs_per_eval: int = 2,
    seed: int = 0, verbose: bool = True,
) -> Tuple[GrowthGraph, List[dict]]:
    """Greedy complexification with random operation selection.

    At each step:
      1. Try K random growth operations on the current best graph
      2. Evaluate each candidate
      3. Keep the best one if it improves over current

    Returns (best_graph, history).
    """
    rng = random.Random(seed)
    current = initial_graph()
    # Evaluate the initial graph
    cur_acc, cur_params, cur_time = evaluate_graph(
        current, train_loader, val_loader, num_classes, in_channels, image_size, epochs_per_eval
    )
    if verbose:
        print(f"  Step 0: initial graph acc={cur_acc:.4f} params={cur_params} ({cur_time:.0f}s)")

    history = [{"step": 0, "acc": cur_acc, "params": cur_params, "n_nodes": len(current.nodes),
                "op": "init", "accepted": True}]

    for step in range(1, n_steps + 1):
        # Generate K candidates
        candidates = []
        for _ in range(candidates_per_step):
            op = rng.choice(OPS)
            new_graph = apply_operation(current, op, rng)
            candidates.append((op, new_graph))

        # Evaluate candidates
        best_op = None
        best_acc = cur_acc
        best_graph = None
        best_params = cur_params
        for op, graph in candidates:
            acc, params, t = evaluate_graph(
                graph, train_loader, val_loader, num_classes, in_channels, image_size, epochs_per_eval
            )
            if verbose:
                print(f"  Step {step} candidate ({op}): acc={acc:.4f} params={params} ({t:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                best_op = op
                best_graph = graph
                best_params = params

        # Accept if improved
        if best_graph is not None:
            current = best_graph
            cur_acc = best_acc
            cur_params = best_params
            history.append({"step": step, "acc": cur_acc, "params": cur_params,
                            "n_nodes": len(current.nodes), "op": best_op, "accepted": True})
            if verbose:
                print(f"  Step {step}: ACCEPTED {best_op} -> acc={cur_acc:.4f} params={cur_params}")
        else:
            history.append({"step": step, "acc": cur_acc, "params": cur_params,
                            "n_nodes": len(current.nodes), "op": "none", "accepted": False})
            if verbose:
                print(f"  Step {step}: no improvement, keeping current (acc={cur_acc:.4f})")

    return current, history


def main():
    print("=== Greedy Complexification (Random Policy) ===")
    train, val, nc = get_datasets("fashionmnist")
    train_sub = Subset(train, list(range(3000)))
    val_sub = Subset(val, list(range(1000)))
    tl = DataLoader(train_sub, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=64, shuffle=False)
    spec = get_spec("fashionmnist")

    t0 = time.time()
    best_graph, history = greedy_search_random(
        tl, vl, nc, spec.in_channels, spec.image_size,
        n_steps=4, candidates_per_step=3, epochs_per_eval=2,
        seed=0, verbose=True,
    )
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Best acc: {history[-1]['acc']:.4f} ({len(best_graph.nodes)} nodes, {history[-1]['params']} params)")
    print(f"History: {[(h['step'], h['op'], h['acc']) for h in history]}")


if __name__ == "__main__":
    main()
