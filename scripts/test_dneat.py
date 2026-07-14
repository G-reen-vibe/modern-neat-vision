"""Quick D-NEAT search test. Uses a small Fashion-MNIST subset for speed.

Tests the end-to-end search loop with a tiny population and few generations.
Should complete in a few minutes.
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
from src.models.dneat.developmental import DevelopmentalConfig


def get_small_loaders(dataset_name: str = "fashionmnist",
                      train_size: int = 5000, val_size: int = 1000,
                      batch_size: int = 128):
    train, val, nc = get_datasets(dataset_name)
    train_sub = Subset(train, list(range(min(train_size, len(train)))))
    val_sub = Subset(val, list(range(min(val_size, len(val)))))
    tl = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    return tl, vl, nc


def main():
    print("=== D-NEAT quick search test ===")
    print("Loading data...")
    train_loader, val_loader, nc = get_small_loaders("fashionmnist", 3000, 1000, 64)
    spec = get_spec("fashionmnist")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    config = DNeatConfig(
        population_size=3,
        generations=2,
        inner_train_epochs=2,
        stability_weight=0.5,
        train_subset_size=3000,
        val_subset_size=1000,
        batch_size=64,
        dev_config=DevelopmentalConfig(noise_sigma=0.0),
    )
    print(f"Config: pop={config.population_size}, gens={config.generations}, "
          f"epochs={config.inner_train_epochs}")

    t0 = time.time()
    population = run_dneat(config, train_loader, val_loader,
                           num_classes=nc, in_channels=spec.in_channels,
                           image_size=spec.image_size, seed=0, verbose=True)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Best individual: val_acc={population[0].val_acc:.4f}, "
          f"nodes={population[0].phenotype_node_count}, "
          f"time={population[0].train_time_s:.0f}s")
    print(f"Best genome: {len(population[0].genome.nodes)} nodes, "
          f"{len(population[0].genome.edges)} edges")


if __name__ == "__main__":
    main()
