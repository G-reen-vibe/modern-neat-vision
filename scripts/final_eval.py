"""Final evaluation: compare all approaches on Fashion-MNIST.

Approaches:
  1. Simple CNN baseline (hand-designed, 94K params)
  2. Initial growth graph (no search)
  3. Progressive growth (4 steps)
  4. Policy search best (if saved)

All trained on 5000 train / 1000 val for 3 epochs.
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


def get_loaders(dataset="fashionmnist", train_size=5000, val_size=1000, batch_size=64):
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=batch_size, shuffle=False)
    return tl, vl, nc


def train_and_eval(model, tl, vl, nc, epochs=2):
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    t0 = time.time()
    result = trainer.fit(epochs, logger=None, eval_every=1)
    return result["best_acc"], time.time() - t0, sum(p.numel() for p in model.parameters())


def main():
    print("=" * 60)
    print("FINAL EVALUATION: Growth Search vs Baselines")
    print("=" * 60)
    all_results = {}

    for dataset in ["fashionmnist", "cifar10"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset}")
        print(f"{'='*40}")
        tl, vl, nc = get_loaders(dataset=dataset)
        spec = get_spec(dataset)
        results = {}

        # 1. Simple CNN
        print(f"\n[1] Simple CNN ({dataset})")
        model = build_model("simple_cnn", num_classes=nc, in_channels=spec.in_channels)
        acc, t, params = train_and_eval(model, tl, vl, nc, epochs=2)
        results["simple_cnn"] = (acc, params, t)
        print(f"  acc={acc:.4f} params={params:,} time={t:.0f}s")

        # 2. Initial growth graph
        print(f"\n[2] Initial growth graph ({dataset})")
        g = initial_graph()
        p = graph_to_phenotype(g)
        model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
        acc, t, params = train_and_eval(model, tl, vl, nc, epochs=2)
        results["initial_graph"] = (acc, params, t)
        print(f"  acc={acc:.4f} params={params:,} time={t:.0f}s")

        all_results[dataset] = results

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        print(f"{'Method':<25} {'Accuracy':>10} {'Params':>10} {'Time':>8}")
        print("-" * 55)
        for name, (acc, params, t) in results.items():
            print(f"{name:<25} {acc:>10.4f} {params:>10,} {t:>7.0f}s")


if __name__ == "__main__":
    main()
