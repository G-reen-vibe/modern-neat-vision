"""Can growth search improve over the initial graph?

Now that init is fixed (Round 19), the initial graph matches Simple CNN.
This experiment tests whether a short growth search (3 steps) can improve
accuracy beyond the initial graph, with proper multi-seed evaluation.
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
from src.search.growth import initial_graph, apply_operation, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer
from src.utils.seed import seed_everything
from src.models.baselines import count_parameters


def train_one(graph, tl, vl, nc, spec, epochs, seed, parent_state=None):
    seed_everything(seed)
    p = graph_to_phenotype(graph)
    if p is None or not p.is_valid():
        return 0.0, 0, None
    try:
        model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
        x = torch.randn(2, spec.in_channels, spec.image_size, spec.image_size)
        model(x)
        if parent_state is not None:
            ms = model.state_dict()
            for k, v in parent_state.items():
                if k in ms and ms[k].shape == v.shape:
                    ms[k] = v
            model.load_state_dict(ms)
        params = count_parameters(model)
        trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                          lr=5e-4, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                          label_smoothing=0.1, grad_clip=1.0, device="cpu")
        res = trainer.fit(epochs, logger=None, eval_every=1)
        return res["best_acc"], params, model.state_dict()
    except Exception:
        return 0.0, 0, None


def growth_search(tl, vl, nc, spec, n_steps, epochs, seed):
    """Run a short growth search. Returns (best_acc, history)."""
    rng = random.Random(seed)
    GOOD_OPS = ["add_pool", "add_bn_relu", "add_skip"]
    current = initial_graph()
    cur_acc, _, cur_state = train_one(current, tl, vl, nc, spec, epochs, seed)
    history = [cur_acc]
    for step in range(n_steps):
        best_acc = cur_acc
        best_graph = None
        best_state = None
        for op in GOOD_OPS:
            g = apply_operation(current, op, rng)
            acc, _, state = train_one(g, tl, vl, nc, spec, epochs, seed, cur_state)
            if acc > best_acc:
                best_acc = acc
                best_graph = g
                best_state = state
        if best_graph is not None:
            current = best_graph
            cur_acc = best_acc
            cur_state = best_state
        history.append(cur_acc)
    return cur_acc, history


def main():
    dataset = "fashionmnist"
    train_size = 3000
    epochs = 2
    n_steps = 2
    seeds = [0, 1]

    print(f"=== Growth Search Improvement Test ===")
    print(f"train={train_size}, epochs={epochs}, steps={n_steps}, seeds={seeds}")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(1000)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    # Baseline: initial graph (no search)
    base_accs = []
    for seed in seeds:
        acc, _, _ = train_one(initial_graph(), tl, vl, nc, spec, epochs, seed)
        base_accs.append(acc)
    print(f"\nBaseline (no search): {np.mean(base_accs):.4f} ± {np.std(base_accs):.4f}  {base_accs}")

    # Growth search
    search_accs = []
    search_histories = []
    for seed in seeds:
        acc, hist = growth_search(tl, vl, nc, spec, n_steps, epochs, seed)
        search_accs.append(acc)
        search_histories.append(hist)
        print(f"  Search seed={seed}: {acc:.4f}  trajectory={[f'{h:.3f}' for h in hist]}")

    print(f"\nSearch result: {np.mean(search_accs):.4f} ± {np.std(search_accs):.4f}  {search_accs}")
    delta = np.mean(search_accs) - np.mean(base_accs)
    print(f"Improvement: {delta:+.4f}")

    results = {
        "baseline": {"mean": float(np.mean(base_accs)), "std": float(np.std(base_accs)), "accs": base_accs},
        "search": {"mean": float(np.mean(search_accs)), "std": float(np.std(search_accs)), "accs": search_accs, "histories": search_histories},
        "delta": float(delta),
    }
    out = Path("results/analysis/21_search_improvement.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
