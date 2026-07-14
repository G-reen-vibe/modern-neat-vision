"""CIFAR-10 controlled baseline: Simple CNN vs Growth Graph, 2 seeds.

Tests whether the findings from Fashion-MNIST transfer to CIFAR-10
(3-channel 32×32 images, 10 classes, harder).
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
from src.models.baselines import build_model, count_parameters
from src.train.trainer import Trainer
from src.utils.seed import seed_everything


def run_one(model_fn, tl, vl, nc, epochs, seed):
    seed_everything(seed)
    model = model_fn()
    params = count_parameters(model)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    t0 = time.time()
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"], params, time.time() - t0


def main():
    dataset = "cifar10"
    train_size = 3000
    val_size = 1000
    epochs = 2
    seeds = [0]

    print(f"=== CIFAR-10 Controlled Baseline ===")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    def make_simple_cnn():
        return build_model("simple_cnn", num_classes=nc, in_channels=spec.in_channels)
    def make_growth_graph():
        g = initial_graph()
        p = graph_to_phenotype(g)
        return compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)

    results = {}
    for name, fn in [("simple_cnn", make_simple_cnn), ("growth_graph", make_growth_graph)]:
        accs, params_list, times = [], [], []
        for seed in seeds:
            acc, params, t = run_one(fn, tl, vl, nc, epochs, seed)
            accs.append(acc)
            params_list.append(params)
            times.append(t)
            print(f"  {name} seed={seed}: acc={acc:.4f} params={params} time={t:.0f}s")
        results[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                         "params": params_list[0], "accs": accs}

    print(f"\n=== Summary ({dataset}, {len(seeds)} seeds) ===")
    for name, r in results.items():
        print(f"  {name}: {r['mean']:.4f} ± {r['std']:.4f}  (params={r['params']})")

    out = Path("results/analysis/15_cifar10_baseline.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
