"""Epoch budget analysis: how many epochs for a stable fitness signal?

Trains the initial graph with 1, 2, 3, 5 epochs on 3k samples, 3 seeds each.
Reports mean ± std and the coefficient of variation (std/mean).
Lower CV = more stable signal for the search.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import initial_graph, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer
from src.utils.seed import seed_everything
from src.models.baselines import count_parameters


def train_one(graph, tl, vl, nc, spec, epochs, seed):
    seed_everything(seed)
    p = graph_to_phenotype(graph)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"]


def main():
    dataset = "fashionmnist"
    train_size = 3000
    val_size = 1000
    epoch_counts = [1, 2, 3, 5]
    seeds = [0, 1, 2]

    print(f"=== Epoch Budget Analysis ===")
    print(f"Dataset: {dataset}, train={train_size}, seeds={seeds}")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    results = {}
    for epochs in epoch_counts:
        accs = []
        t0 = time.time()
        for seed in seeds:
            acc = train_one(initial_graph(), tl, vl, nc, spec, epochs, seed)
            accs.append(acc)
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        cv = std / mean if mean > 0 else float('inf')
        elapsed = time.time() - t0
        results[epochs] = {"mean": mean, "std": std, "cv": cv, "accs": accs, "time": elapsed}
        print(f"  {epochs} epoch(s): {mean:.4f} ± {std:.4f}  CV={cv:.4f}  time={elapsed:.0f}s  {accs}")

    out = Path("results/analysis/06_epoch_budget.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Find minimum epochs where CV < 0.05 (5% relative std)
    print(f"\n=== Minimum viable epochs (CV < 0.05) ===")
    for epochs, r in results.items():
        stable = "✓ STABLE" if r["cv"] < 0.05 else "✗ noisy"
        print(f"  {epochs} epochs: CV={r['cv']:.4f} {stable}")


if __name__ == "__main__":
    main()
