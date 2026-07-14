"""Fast operation ablation: 1 seed, 1 epoch, 1000 samples.
Tests each operation's effect on accuracy vs baseline.
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


def train_one(graph, tl, vl, nc, spec, epochs, seed):
    seed_everything(seed)
    p = graph_to_phenotype(graph)
    if p is None or not p.is_valid():
        return 0.0, 0
    try:
        model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
        x = torch.randn(2, spec.in_channels, spec.image_size, spec.image_size)
        model(x)
        params = count_parameters(model)
        trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                          lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                          label_smoothing=0.1, grad_clip=1.0, device="cpu")
        res = trainer.fit(epochs, logger=None, eval_every=1)
        return res["best_acc"], params
    except Exception:
        return 0.0, 0


def main():
    dataset = "fashionmnist"
    train_size = 1000
    val_size = 500
    epochs = 1
    seeds = [0]

    print(f"=== Fast Operation Ablation ===")
    print(f"Dataset: {dataset}, train={train_size}, epochs={epochs}, seeds={seeds}")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    # Baseline
    base_acc, base_params = train_one(initial_graph(), tl, vl, nc, spec, epochs, 0)
    print(f"\nBaseline: acc={base_acc:.4f} params={base_params}")

    results = {"baseline": {"mean": base_acc, "params": base_params, "delta": 0.0}}
    for op in OPS:
        rng = random.Random(42)
        g = initial_graph()
        g = apply_operation(g, op, rng)
        acc, params = train_one(g, tl, vl, nc, spec, epochs, 0)
        delta = acc - base_acc
        results[op] = {"mean": acc, "params": params, "delta": delta}
        print(f"  {op:15s}: acc={acc:.4f} delta={delta:+.4f} params={params}")

    out = Path("results/analysis/05_operation_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    print(f"\n=== Operation ranking ===")
    ranked = sorted(results.items(), key=lambda x: x[1]["delta"], reverse=True)
    for name, r in ranked:
        print(f"  {name:15s}: delta={r['delta']:+.4f} params={r['params']}")


if __name__ == "__main__":
    main()
