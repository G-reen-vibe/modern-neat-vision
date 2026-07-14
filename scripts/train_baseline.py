"""Train a single baseline model on a single dataset with a single seed.

Usage:
  python3 scripts/train_baseline.py \
      --dataset cifar10 --model resnet18 --seed 0

CLI overrides in 'a.b.c=value' format are supported:
  --set train.epochs=50 train.lr=0.0005
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, parse_cli_overrides
from src.utils.seed import seed_everything
from src.utils.logging import RunLogger
from src.data.datasets import get_dataloaders, get_spec
from src.models.baselines import build_model, count_parameters
from src.train.trainer import Trainer
from src.eval.metrics import compute_accuracy, count_flops, measure_latency


CONFIG_DIR = PROJECT_ROOT / "configs"


def build_run_name(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}__{model}__seed{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["fashionmnist", "cifar10", "cifar100"])
    ap.add_argument("--model", required=True,
                    choices=["simple_cnn", "resnet18", "mobilenetv3_small",
                             "efficientnet_b0", "deit_tiny"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=None,
                    help="Config overrides as key=value pairs.")
    ap.add_argument("--run-dir", default=None,
                    help="Override the run output directory.")
    args = ap.parse_args()

    # Load and merge configs
    overrides = parse_cli_overrides(args.set)
    cfg = load_config(
        CONFIG_DIR / "base.yaml",
        CONFIG_DIR / "dataset" / f"{args.dataset}.yaml",
        CONFIG_DIR / "model" / f"{args.model}.yaml",
        overrides,
    )
    # Inject seed and dataset/model names back into cfg for the logger
    cfg["seed"] = args.seed
    cfg["dataset"]["name"] = args.dataset
    cfg["model"]["name"] = args.model

    # Seed
    seed_everything(args.seed)

    # Data
    spec = get_spec(args.dataset)
    train_loader, val_loader, num_classes = get_dataloaders(
        args.dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"].get("num_workers", 0),
    )

    # Model
    model = build_model(args.model, num_classes=num_classes, in_channels=spec.in_channels)
    n_params = count_parameters(model)

    # Logger
    run_name = args.run_dir or build_run_name(args.dataset, args.model, args.seed)
    run_dir = PROJECT_ROOT / "results" / "runs" / run_name
    logger = RunLogger(run_dir, cfg)
    logger.log(f"Run: {run_name}")
    logger.log(f"Dataset: {args.dataset}  Model: {args.model}  Seed: {args.seed}")
    logger.log(f"Params: {n_params:,}")
    logger.log(f"Train epochs: {cfg['train']['epochs']}  Batch size: {cfg['train']['batch_size']}")
    logger.log(f"LR: {cfg['train']['lr']}  WD: {cfg['train']['weight_decay']}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_epochs=cfg["train"]["warmup_epochs"],
        total_epochs=cfg["train"]["epochs"],
        label_smoothing=cfg["train"]["label_smoothing"],
        grad_clip=cfg["train"]["grad_clip"],
        device=cfg.get("device", "cpu"),
    )
    t0 = time.time()
    train_result = trainer.fit(cfg["train"]["epochs"], logger=logger)
    total_time = time.time() - t0

    # Final eval
    top1, top5 = compute_accuracy(model, val_loader, device=cfg.get("device", "cpu"))
    macs, _ = count_flops(model, input_size=(1, spec.in_channels, spec.image_size, spec.image_size))
    lat = measure_latency(
        model,
        input_size=(1, spec.in_channels, spec.image_size, spec.image_size),
        device=cfg.get("device", "cpu"),
        n_runs=cfg["eval"].get("latency_runs", 100),
    )

    summary = {
        "best_acc": train_result["best_acc"],
        "best_epoch": train_result["best_epoch"],
        "final_acc": top1,
        "top1_acc": top1,
        "top5_acc": top5 if top5 is not None else -1.0,
        "params": n_params,
        "flops": macs,
        "train_time_s_total": total_time,
        "latency_ms_median": lat["latency_ms_median"],
        "latency_ms_mean": lat["latency_ms_mean"],
    }
    logger.log_final(summary)
    logger.log(f"DONE. Best acc: {train_result['best_acc']:.4f} (epoch {train_result['best_epoch']})  Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
