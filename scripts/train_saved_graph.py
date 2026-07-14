"""Train a saved graph topology with more epochs and on a larger dataset.

Usage:
  python3 scripts/train_saved_graph.py --graph results/best_graph.json --epochs 10 --dataset cifar10
"""
from __future__ import annotations
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import graph_from_dict, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer
from src.eval.metrics import compute_accuracy, count_parameters, count_flops, measure_latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="Path to saved graph JSON")
    ap.add_argument("--dataset", default="fashionmnist", choices=["fashionmnist", "cifar10", "cifar100"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train-size", type=int, default=10000, help="Train subset size (0 = full)")
    ap.add_argument("--val-size", type=int, default=2000, help="Val subset size (0 = full)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.utils.seed import seed_everything
    seed_everything(args.seed)

    # Load graph
    with open(args.graph) as f:
        graph_dict = json.load(f)
    graph = graph_from_dict(graph_dict)
    print(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Data
    train, val, nc = get_datasets(args.dataset)
    train_size = args.train_size if args.train_size > 0 else len(train)
    val_size = args.val_size if args.val_size > 0 else len(val)
    train_sub = Subset(train, list(range(min(train_size, len(train)))))
    val_sub = Subset(val, list(range(min(val_size, len(val)))))
    tl = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
    spec = get_spec(args.dataset)
    print(f"Dataset: {args.dataset} ({train_size} train, {val_size} val)")

    # Compile model
    p = graph_to_phenotype(graph)
    model = compile_phenotype(p, in_channels=spec.in_channels, num_classes=nc, image_size=spec.image_size)
    params = count_parameters(model)
    print(f"Params: {params:,}")

    # Train
    trainer = Trainer(model=model, train_loader=tl, val_loader=vl, num_classes=nc,
                      lr=1e-3, weight_decay=5e-4, warmup_epochs=2, total_epochs=args.epochs,
                      label_smoothing=0.1, grad_clip=1.0, device="cpu")
    t0 = time.time()
    result = trainer.fit(args.epochs, logger=None, eval_every=1)
    total_time = time.time() - t0

    # Final eval
    top1, top5 = compute_accuracy(model, vl, device="cpu")
    macs, _ = count_flops(model, input_size=(1, spec.in_channels, spec.image_size, spec.image_size))
    lat = measure_latency(model, input_size=(1, spec.in_channels, spec.image_size, spec.image_size), n_runs=50)

    print(f"\n=== Results ===")
    print(f"Best val acc: {result['best_acc']:.4f} (epoch {result['best_epoch']})")
    print(f"Final acc: {top1:.4f}")
    if top5 is not None:
        print(f"Top-5 acc: {top5:.4f}")
    print(f"Params: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"Latency: {lat['latency_ms_median']:.2f}ms (median)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
