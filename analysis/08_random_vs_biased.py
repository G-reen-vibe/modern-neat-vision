"""Policy vs Random search comparison.

Runs both search strategies with the same budget and compares final accuracy.
This tests whether the learned policy provides any benefit over random search.
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


def random_search(tl, vl, nc, spec, n_steps, candidates_per_step, epochs, seed):
    """Pure random search: try K random ops per step, keep best."""
    rng = random.Random(seed)
    current = initial_graph()
    cur_acc, cur_params = train_one(current, tl, vl, nc, spec, epochs, seed)
    history = [{"step": 0, "acc": cur_acc, "op": "init"}]
    for step in range(1, n_steps + 1):
        best_acc = cur_acc
        best_graph = None
        for _ in range(candidates_per_step):
            op = rng.choice(OPS)
            g = apply_operation(current, op, rng)
            acc, _ = train_one(g, tl, vl, nc, spec, epochs, seed)
            if acc > best_acc:
                best_acc = acc
                best_graph = g
        if best_graph is not None:
            current = best_graph
            cur_acc = best_acc
            history.append({"step": step, "acc": cur_acc, "op": "random_best"})
        else:
            history.append({"step": step, "acc": cur_acc, "op": "none"})
    return current, cur_acc, history


def biased_search(tl, vl, nc, spec, n_steps, candidates_per_step, epochs, seed):
    """Biased search: only use the top-3 operations from ablation (add_pool, add_bn_relu, add_block)."""
    rng = random.Random(seed)
    GOOD_OPS = ["add_pool", "add_bn_relu", "add_block", "add_skip"]
    current = initial_graph()
    cur_acc, cur_params = train_one(current, tl, vl, nc, spec, epochs, seed)
    history = [{"step": 0, "acc": cur_acc, "op": "init"}]
    for step in range(1, n_steps + 1):
        best_acc = cur_acc
        best_graph = None
        for _ in range(candidates_per_step):
            op = rng.choice(GOOD_OPS)
            g = apply_operation(current, op, rng)
            acc, _ = train_one(g, tl, vl, nc, spec, epochs, seed)
            if acc > best_acc:
                best_acc = acc
                best_graph = g
        if best_graph is not None:
            current = best_graph
            cur_acc = best_acc
            history.append({"step": step, "acc": cur_acc, "op": "biased_best"})
        else:
            history.append({"step": step, "acc": cur_acc, "op": "none"})
    return current, cur_acc, history


def main():
    dataset = "fashionmnist"
    train_size = 3000
    val_size = 1000
    epochs = 2
    n_steps = 3
    candidates_per_step = 2
    seeds = [0]

    print(f"=== Random vs Biased Search ===")
    print(f"train={train_size}, epochs={epochs}, steps={n_steps}, candidates={candidates_per_step}")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    results = {"random": [], "biased": []}
    for seed in seeds:
        print(f"\nSeed {seed}:")
        t0 = time.time()
        _, r_acc, r_hist = random_search(tl, vl, nc, spec, n_steps, candidates_per_step, epochs, seed)
        r_time = time.time() - t0
        results["random"].append({"seed": seed, "acc": r_acc, "time": r_time, "history": r_hist})
        print(f"  Random:  acc={r_acc:.4f} time={r_time:.0f}s  {[h['acc'] for h in r_hist]}")

        t0 = time.time()
        _, b_acc, b_hist = biased_search(tl, vl, nc, spec, n_steps, candidates_per_step, epochs, seed)
        b_time = time.time() - t0
        results["biased"].append({"seed": seed, "acc": b_acc, "time": b_time, "history": b_hist})
        print(f"  Biased:  acc={b_acc:.4f} time={b_time:.0f}s  {[h['acc'] for h in b_hist]}")

    print(f"\n=== Summary ===")
    for method in ["random", "biased"]:
        accs = [r["acc"] for r in results[method]]
        print(f"  {method}: {np.mean(accs):.4f} ± {np.std(accs):.4f}  {accs}")

    out = Path("results/analysis/08_random_vs_biased.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
