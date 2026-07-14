"""Multi-seed evaluation of the growth graph vs Simple CNN.

Trains each model with 3 different seeds and reports mean ± std.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import initial_graph, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.models.baselines import build_model
from src.train.trainer import Trainer
from src.utils.seed import seed_everything


def train_and_eval(model, tl, vl, nc, epochs=2, seed=0):
    seed_everything(seed)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    result = trainer.fit(epochs, logger=None, eval_every=1)
    return result["best_acc"]


def main():
    print("=== Multi-seed Evaluation (3 seeds) ===")
    train, val, nc = get_datasets("fashionmnist")
    tl = DataLoader(Subset(train, range(5000)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(1000)), batch_size=64, shuffle=False)
    spec = get_spec("fashionmnist")

    results = {"simple_cnn": [], "growth_graph": []}
    for seed in range(3):
        print(f"\nSeed {seed}:")
        # Simple CNN
        model = build_model("simple_cnn", num_classes=nc, in_channels=spec.in_channels)
        acc = train_and_eval(model, tl, vl, nc, epochs=2, seed=seed)
        results["simple_cnn"].append(acc)
        print(f"  Simple CNN: {acc:.4f}")

        # Growth graph
        g = initial_graph()
        p = graph_to_phenotype(g)
        model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
        acc = train_and_eval(model, tl, vl, nc, epochs=2, seed=seed)
        results["growth_graph"].append(acc)
        print(f"  Growth graph: {acc:.4f}")

    # Summary
    print("\n=== Summary (3 seeds) ===")
    import numpy as np
    for name, accs in results.items():
        m = np.mean(accs)
        s = np.std(accs)
        print(f"  {name}: {m:.4f} ± {s:.4f}  (seeds: {[f'{a:.4f}' for a in accs]})")


if __name__ == "__main__":
    main()
