"""Controlled baseline: Simple CNN vs Initial Growth Graph, 3 seeds, same budget.

Both models have identical architecture (3 convs 32->64->128 + 2 maxpools + GAP + linear).
The only difference is how they're constructed: Simple CNN is hand-written in PyTorch;
the Growth Graph is compiled from our graph representation.

This tests whether our compilation pipeline introduces any performance difference.
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
from src.eval.metrics import compute_accuracy, count_flops


def run_one(model_fn, tl, vl, nc, epochs, seed):
    seed_everything(seed)
    model = model_fn()
    params = count_parameters(model)
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    t0 = time.time()
    res = trainer.fit(epochs, logger=None, eval_every=1)
    t = time.time() - t0
    # Per-epoch accuracies from history
    epoch_accs = [h.get("val_acc", 0) for h in res["history"]]
    return res["best_acc"], params, t, epoch_accs


def main():
    dataset = "fashionmnist"
    train_size = 5000
    val_size = 1000
    epochs = 3
    seeds = [0, 1, 2]

    print(f"=== Controlled Baseline: Simple CNN vs Growth Graph ===")
    print(f"Dataset: {dataset}, train={train_size}, val={val_size}, epochs={epochs}, seeds={seeds}")
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
        accs, params_list, times, epoch_accs_list = [], [], [], []
        for seed in seeds:
            acc, params, t, epoch_accs = run_one(fn, tl, vl, nc, epochs, seed)
            accs.append(acc)
            params_list.append(params)
            times.append(t)
            epoch_accs_list.append(epoch_accs)
            print(f"  {name} seed={seed}: acc={acc:.4f} params={params} time={t:.0f}s epoch_accs={[f'{a:.3f}' for a in epoch_accs]}")
        results[name] = {
            "accs": accs,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "params": params_list[0],
            "mean_time": float(np.mean(times)),
            "epoch_accs": epoch_accs_list,
        }

    print(f"\n=== Summary (mean ± std over {len(seeds)} seeds) ===")
    for name, r in results.items():
        print(f"  {name}: {r['mean']:.4f} ± {r['std']:.4f}  (params={r['params']}, time={r['mean_time']:.0f}s)")

    # Save
    out = Path("results/analysis/01_controlled_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
