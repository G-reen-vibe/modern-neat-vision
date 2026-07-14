"""Weight inheritance ablation: does transferring weights help?

Compares:
1. Train from scratch (no inheritance)
2. Train with weight inheritance from parent

Both use the same growth sequence: initial -> add_pool -> add_bn_relu
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import random

from src.data.datasets import get_datasets, get_spec
from src.search.growth import initial_graph, apply_operation, graph_to_phenotype, OPS
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer
from src.utils.seed import seed_everything
from src.models.baselines import count_parameters


def train_with_inheritance(graph, parent_state, tl, vl, nc, spec, epochs, seed, lr=1e-3):
    """Train a graph, optionally inheriting weights from parent."""
    seed_everything(seed)
    p = graph_to_phenotype(graph)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    # Transfer weights
    transferred = 0
    if parent_state is not None:
        model_state = model.state_dict()
        for k, v in parent_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                transferred += 1
        if transferred > 0:
            model.load_state_dict(model_state)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=lr, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"], model.state_dict(), transferred


def train_no_inheritance(graph, tl, vl, nc, spec, epochs, seed, lr=1e-3):
    """Train a graph from scratch."""
    seed_everything(seed)
    p = graph_to_phenotype(graph)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=lr, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"], model.state_dict()


def main():
    dataset = "fashionmnist"
    train_size = 3000
    val_size = 1000
    epochs = 2
    seed = 0

    print(f"=== Weight Inheritance Ablation ===")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    rng = random.Random(42)
    # Growth sequence: initial -> add_pool -> add_bn_relu
    g0 = initial_graph()
    g1 = apply_operation(g0, "add_pool", rng)
    g2 = apply_operation(g1, "add_bn_relu", rng)

    # --- No inheritance ---
    print("\n[No inheritance]")
    acc0, state0 = train_no_inheritance(g0, tl, vl, nc, spec, epochs, seed)
    print(f"  Step 0 (initial): {acc0:.4f}")
    acc1, state1 = train_no_inheritance(g1, tl, vl, nc, spec, epochs, seed)
    print(f"  Step 1 (add_pool): {acc1:.4f}")
    acc2, state2 = train_no_inheritance(g2, tl, vl, nc, spec, epochs, seed)
    print(f"  Step 2 (add_bn_relu): {acc2:.4f}")
    no_inh_final = acc2

    # --- With inheritance (lower LR for fine-tuning) ---
    print("\n[With inheritance]")
    acc0_i, state0_i = train_no_inheritance(g0, tl, vl, nc, spec, epochs, seed)
    print(f"  Step 0 (initial): {acc0_i:.4f}")
    acc1_i, state1_i, t1 = train_with_inheritance(g1, state0_i, tl, vl, nc, spec, epochs, seed, lr=5e-4)
    print(f"  Step 1 (add_pool): {acc1_i:.4f} (transferred {t1} weights)")
    acc2_i, state2_i, t2 = train_with_inheritance(g2, state1_i, tl, vl, nc, spec, epochs, seed, lr=5e-4)
    print(f"  Step 2 (add_bn_relu): {acc2_i:.4f} (transferred {t2} weights)")
    inh_final = acc2_i

    print(f"\n=== Summary ===")
    print(f"  No inheritance:     {no_inh_final:.4f}")
    print(f"  With inheritance:   {inh_final:.4f}")
    print(f"  Difference:         {inh_final - no_inh_final:+.4f}")

    results = {
        "no_inheritance": {"step0": acc0, "step1": acc1, "step2": acc2},
        "with_inheritance": {"step0": acc0_i, "step1": acc1_i, "step2": acc2_i, "transferred": [t1, t2]},
    }
    out = Path("results/analysis/10_weight_inheritance.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
