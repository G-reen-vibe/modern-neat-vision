"""Growth steps analysis: how many steps to reach best accuracy?

Runs growth search with 1, 2, 3, 4, 5 steps and reports accuracy at each.
Uses the good operations (add_pool, add_bn_relu, add_block, add_skip).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


def main():
    dataset = "fashionmnist"
    train_size = 3000
    epochs = 2
    seed = 0
    max_steps = 3

    print(f"=== Growth Steps Analysis ===")
    train, val, nc = get_datasets(dataset)
    tl = DataLoader(Subset(train, range(train_size)), batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(Subset(val, range(1000)), batch_size=64, shuffle=False)
    spec = get_spec(dataset)

    rng = random.Random(seed)
    GOOD_OPS = ["add_pool", "add_bn_relu", "add_block", "add_skip"]

    current = initial_graph()
    cur_acc, cur_params, cur_state = train_one(current, tl, vl, nc, spec, epochs, seed)
    history = [{"step": 0, "acc": cur_acc, "params": cur_params, "n_nodes": len(current.nodes)}]
    print(f"Step 0: acc={cur_acc:.4f} params={cur_params} nodes={len(current.nodes)}")

    for step in range(1, max_steps + 1):
        best_acc = cur_acc
        best_graph = None
        best_state = None
        best_op = None
        for op in GOOD_OPS:
            g = apply_operation(current, op, rng)
            acc, params, state = train_one(g, tl, vl, nc, spec, epochs, seed, cur_state)
            if acc > best_acc:
                best_acc = acc
                best_graph = g
                best_state = state
                best_op = op
        if best_graph is not None:
            current = best_graph
            cur_acc = best_acc
            cur_state = best_state
            history.append({"step": step, "acc": cur_acc, "params": best_state and count_parameters(compile_phenotype(graph_to_phenotype(current), in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)), "n_nodes": len(current.nodes), "op": best_op})
            print(f"Step {step}: ACCEPTED {best_op} -> acc={cur_acc:.4f} nodes={len(current.nodes)}")
        else:
            history.append({"step": step, "acc": cur_acc, "op": "none", "n_nodes": len(current.nodes)})
            print(f"Step {step}: no improvement (acc={cur_acc:.4f})")

    out = Path("results/analysis/13_growth_steps.json")
    with open(out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved to {out}")
    print(f"\n=== Accuracy trajectory ===")
    for h in history:
        print(f"  Step {h['step']}: acc={h['acc']:.4f} nodes={h['n_nodes']}")


if __name__ == "__main__":
    main()
