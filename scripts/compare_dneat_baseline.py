"""Compare D-NEAT discovered topologies against a fixed baseline.

Trains:
  1. A fixed simple CNN (conv -> pool -> head) — the "fallback" phenotype
  2. The best D-NEAT topology found in a short search

Both trained on the same data subset for the same number of epochs.
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
from src.search.neat import run_dneat, DNeatConfig
from src.models.dneat.developmental import develop, DevelopmentalConfig, _fallback_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer


def get_small_loaders(dataset_name="fashionmnist", train_size=3000, val_size=1000, batch_size=64):
    train, val, nc = get_datasets(dataset_name)
    train_sub = Subset(train, list(range(min(train_size, len(train)))))
    val_sub = Subset(val, list(range(min(val_size, len(val)))))
    tl = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    return tl, vl, nc


def train_phenotype(phenotype, train_loader, val_loader, num_classes, in_channels, image_size, epochs=2):
    model = compile_phenotype(phenotype, in_channels=in_channels, num_classes=num_classes, image_size=image_size)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
                      warmup_epochs=1, total_epochs=epochs, label_smoothing=0.1,
                      grad_clip=1.0, device="cpu")
    t0 = time.time()
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"], time.time() - t0, sum(p.numel() for p in model.parameters())


def main():
    print("=== D-NEAT vs Fixed Baseline ===")
    train_loader, val_loader, nc = get_small_loaders("fashionmnist", 3000, 1000, 64)
    spec = get_spec("fashionmnist")

    # 1. Fixed baseline (fallback chain)
    print("\n[1] Fixed baseline (conv -> pool -> head):")
    baseline_p = _fallback_phenotype()
    # Patch the fallback to use 16 channels for fair comparison
    from src.models.dneat.developmental import PhenotypeNode
    baseline_p.nodes[1] = PhenotypeNode(1, "conv_bn_relu",
        {"out_channels": 16, "kernel_size": 3, "stride": 1, "groups": 1}, (0, 0))
    acc, t, params = train_phenotype(baseline_p, train_loader, val_loader, nc, spec.in_channels, spec.image_size, epochs=2)
    print(f"  acc={acc:.4f}, time={t:.0f}s, params={params}")

    # 2. D-NEAT search
    print("\n[2] D-NEAT search (3 pop, 2 gen, 2 epochs each):")
    config = DNeatConfig(
        population_size=3, generations=2, inner_train_epochs=2,
        batch_size=64, dev_config=DevelopmentalConfig(noise_sigma=0.0),
    )
    t0 = time.time()
    population = run_dneat(config, train_loader, val_loader, nc, spec.in_channels, spec.image_size, seed=0, verbose=True)
    print(f"  Search time: {time.time()-t0:.0f}s")
    best = population[0]
    print(f"  Best D-NEAT: acc={best.val_acc:.4f}, nodes={best.phenotype_node_count}")

    # 3. Retrain best for same budget
    print("\n[3] Retrain best D-NEAT topology with fresh weights:")
    best_p = develop(best.genome, config.dev_config, seed=0)
    acc, t, params = train_phenotype(best_p, train_loader, val_loader, nc, spec.in_channels, spec.image_size, epochs=2)
    print(f"  acc={acc:.4f}, time={t:.0f}s, params={params}")

    print(f"\n=== Summary ===")
    print(f"Fixed baseline:  {baseline_p is not None and 'trained'}")
    print(f"D-NEAT best:     {best.val_acc:.4f} (during search)")
    print(f"D-NEAT retrained:{acc:.4f}")


if __name__ == "__main__":
    main()
