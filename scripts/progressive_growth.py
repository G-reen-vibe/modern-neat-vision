"""Progressive growth: grow the graph while training, keeping weights.

Instead of evaluating each candidate from scratch, this mode:
1. Trains the current graph for E epochs
2. Applies a growth operation (keeping weights via transfer)
3. Trains for E more epochs
4. Repeats

This is much faster than re-evaluating from scratch each time, because
the weights from the previous step warm-start the next.
"""
from __future__ import annotations
import sys
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import initial_graph, apply_operation, graph_to_phenotype, OPS, GrowthGraph
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer


def progressive_growth(
    train_loader, val_loader, num_classes, in_channels, image_size,
    n_steps: int = 5, epochs_per_step: int = 2, seed: int = 0, verbose: bool = True,
):
    """Grow the graph progressively while training."""
    rng = random.Random(seed)
    torch.manual_seed(seed)

    current = initial_graph()
    # Compile and train initial graph
    p = graph_to_phenotype(current)
    model = compile_phenotype(p, in_channels=in_channels, num_classes=num_classes, image_size=image_size)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
                      warmup_epochs=1, total_epochs=epochs_per_step,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    result = trainer.fit(epochs_per_step, logger=None, eval_every=1)
    best_acc = result["best_acc"]
    if verbose:
        print(f"  Step 0: initial acc={best_acc:.4f} params={sum(pp.numel() for pp in model.parameters())}")

    cur_state = model.state_dict()

    for step in range(1, n_steps + 1):
        # Try a few operations, pick the one that gives best forward pass
        best_step_acc = best_acc
        best_step_graph = None
        best_step_state = None

        for _ in range(3):  # try 3 random ops
            op = rng.choice(OPS)
            new_graph = apply_operation(current, op, rng)
            new_p = graph_to_phenotype(new_graph)
            if new_p is None or not new_p.is_valid():
                continue
            try:
                new_model = compile_phenotype(new_p, in_channels=in_channels,
                                              num_classes=num_classes, image_size=image_size)
                # Transfer weights
                new_state = new_model.state_dict()
                transferred = 0
                for k, v in cur_state.items():
                    if k in new_state and new_state[k].shape == v.shape:
                        new_state[k] = v
                        transferred += 1
                if transferred > 0:
                    new_model.load_state_dict(new_state)
                # Quick forward validation
                x = torch.randn(2, in_channels, image_size, image_size)
                new_model(x)
                # Train briefly
                new_trainer = Trainer(model=new_model, train_loader=train_loader, val_loader=val_loader,
                                      num_classes=num_classes, lr=5e-4, weight_decay=5e-4,
                                      warmup_epochs=1, total_epochs=epochs_per_step,
                                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
                result = new_trainer.fit(epochs_per_step, logger=None, eval_every=1)
                acc = result["best_acc"]
                if verbose:
                    print(f"  Step {step} ({op}): acc={acc:.4f} (transferred {transferred} weights)")
                if acc > best_step_acc:
                    best_step_acc = acc
                    best_step_graph = new_graph
                    best_step_state = new_model.state_dict()
            except Exception as e:
                if verbose:
                    print(f"  Step {step} ({op}): FAIL {str(e)[:60]}")
                continue

        if best_step_graph is not None:
            current = best_step_graph
            cur_state = best_step_state
            best_acc = best_step_acc
            if verbose:
                print(f"  Step {step}: ACCEPTED -> acc={best_acc:.4f}")
        else:
            if verbose:
                print(f"  Step {step}: no improvement (acc={best_acc:.4f})")

    return current, best_acc


def main():
    print("=== Progressive Growth ===")
    train, val, nc = get_datasets("fashionmnist")
    train_sub = Subset(train, list(range(5000)))
    val_sub = Subset(val, list(range(1000)))
    tl = DataLoader(train_sub, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=64, shuffle=False)
    spec = get_spec("fashionmnist")

    t0 = time.time()
    best_graph, best_acc = progressive_growth(
        tl, vl, nc, spec.in_channels, spec.image_size,
        n_steps=4, epochs_per_step=2, seed=0, verbose=True,
    )
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Best accuracy: {best_acc:.4f} ({len(best_graph.nodes)} nodes)")


if __name__ == "__main__":
    main()
