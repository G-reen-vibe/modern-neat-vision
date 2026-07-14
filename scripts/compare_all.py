"""Compare: random greedy search vs policy-guided search vs simple CNN baseline.

All trained on the same Fashion-MNIST subset (3000 train, 1000 val, 2 epochs)
for fair comparison. Reports best accuracy and total time.
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
from src.search.growth import initial_graph, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.models.baselines import build_model
from src.train.trainer import Trainer


def get_loaders():
    train, val, nc = get_datasets("fashionmnist")
    train_sub = Subset(train, list(range(3000)))
    val_sub = Subset(val, list(range(1000)))
    tl = DataLoader(train_sub, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=64, shuffle=False)
    return tl, vl, nc


def train_model(model, tl, vl, nc, epochs=2):
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=1, total_epochs=epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    t0 = time.time()
    res = trainer.fit(epochs, logger=None, eval_every=1)
    return res["best_acc"], time.time() - t0


def main():
    print("=== Comparison: Growth Search vs Baselines ===\n")
    tl, vl, nc = get_loaders()
    spec = get_spec("fashionmnist")

    # 1. Simple CNN baseline
    print("[1] Simple CNN baseline (3-layer):")
    model = build_model("simple_cnn", num_classes=nc, in_channels=spec.in_channels)
    acc, t = train_model(model, tl, vl, nc, epochs=2)
    params = sum(p.numel() for p in model.parameters())
    print(f"  acc={acc:.4f} params={params} time={t:.0f}s\n")

    # 2. Initial growth graph (no search)
    print("[2] Initial growth graph (2 convs + BN, no search):")
    g = initial_graph()
    p = graph_to_phenotype(g)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    acc, t = train_model(model, tl, vl, nc, epochs=2)
    params = sum(pp.numel() for pp in model.parameters())
    print(f"  acc={acc:.4f} params={params} time={t:.0f}s\n")

    # 3. Policy search (2 episodes, 4 steps, 2 candidates)
    print("[3] Policy-guided growth search (2 episodes, 4 steps):")
    from scripts.policy_search import policy_search
    t0 = time.time()
    best_graph, best_acc = policy_search(
        tl, vl, nc, spec.in_channels, spec.image_size,
        n_episodes=2, steps_per_episode=4, epochs_per_eval=2, seed=0, verbose=False,
    )
    print(f"  best_acc={best_acc:.4f} time={time.time()-t0:.0f}s\n")

    # 4. Random greedy search (4 steps, 3 candidates)
    print("[4] Random greedy search (4 steps, 3 candidates):")
    from scripts.greedy_search import greedy_search_random
    t0 = time.time()
    best_graph_r, history = greedy_search_random(
        tl, vl, nc, spec.in_channels, spec.image_size,
        n_steps=4, candidates_per_step=3, epochs_per_eval=2, seed=0, verbose=False,
    )
    print(f"  best_acc={history[-1]['acc']:.4f} time={time.time()-t0:.0f}s\n")

    print("=== Summary ===")
    print(f"Simple CNN:      {acc:.4f}")
    print(f"Initial graph:   (see above)")
    print(f"Policy search:   {best_acc:.4f}")
    print(f"Random search:   {history[-1]['acc']:.4f}")


if __name__ == "__main__":
    main()
