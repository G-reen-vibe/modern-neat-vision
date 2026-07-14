"""Operation ablation with 2 epochs (stable budget).

Re-tests each operation with 2-epoch training, which Round 6 showed gives
CV=0.016 (stable signal). This should give a much clearer picture of which
operations actually help.
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
    train_size = 3000
    val_size = 1000
    epochs = 2
    seeds = [0, 1]

    print(f"=== Operation Ablation (2 epochs, {seeds} seeds) ===")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(val_size)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    # Baseline
    base_accs = []
    for seed in seeds:
        acc, _ = train_one(initial_graph(), tl, vl, nc, spec, epochs, seed)
        base_accs.append(acc)
    base_mean = float(np.mean(base_accs))
    print(f"\nBaseline: {base_mean:.4f} ± {float(np.std(base_accs)):.4f}  {base_accs}")

    results = {"baseline": {"mean": base_mean, "std": float(np.std(base_accs)), "accs": base_accs, "delta": 0.0}}
    for op in OPS:
        op_accs = []
        op_params = 0
        for seed in seeds:
            rng = random.Random(seed * 100 + 42)
            g = initial_graph()
            g = apply_operation(g, op, rng)
            acc, params = train_one(g, tl, vl, nc, spec, epochs, seed)
            op_accs.append(acc)
            op_params = params
        op_mean = float(np.mean(op_accs))
        delta = op_mean - base_mean
        results[op] = {"mean": op_mean, "std": float(np.std(op_accs)), "accs": op_accs,
                       "params": op_params, "delta": delta}
        print(f"  {op:15s}: {op_mean:.4f} ± {float(np.std(op_accs)):.4f}  delta={delta:+.4f} params={op_params}  {op_accs}")

    out = Path("results/analysis/07_operation_ablation_2ep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    print(f"\n=== Operation ranking (by mean delta) ===")
    ranked = sorted(results.items(), key=lambda x: x[1].get("delta", 0), reverse=True)
    for name, r in ranked:
        d = r.get("delta", 0)
        print(f"  {name:15s}: delta={d:+.4f}  params={r.get('params', '?')}")


if __name__ == "__main__":
    main()
