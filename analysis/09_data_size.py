"""Single-seed data size effect (fast version)."""
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


def train_one(tl, vl, nc, spec, epochs, seed):
    seed_everything(seed)
    g = initial_graph()
    p = graph_to_phenotype(g)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"]


def main():
    dataset = "fashionmnist"
    data_sizes = [500, 1000, 2000, 5000]
    epochs = 2
    seed = 0

    print(f"=== Data Size Effect (single seed) ===")
    train, val, nc = get_datasets(dataset)
    spec = get_spec(dataset)
    vl = DataLoader(Subset(val, range(1000)), batch_size=64, shuffle=False)

    results = {}
    for size in data_sizes:
        tl = DataLoader(Subset(train, range(size)), batch_size=64, shuffle=True, drop_last=True)
        t0 = time.time()
        acc = train_one(tl, vl, nc, spec, epochs, seed)
        elapsed = time.time() - t0
        results[size] = {"acc": acc, "time": elapsed}
        print(f"  {size:5d} samples: acc={acc:.4f}  time={elapsed:.0f}s")

    out = Path("results/analysis/09_data_size.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
