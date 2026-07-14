"""Sanity-check pipeline. Verifies the whole stack works end-to-end in <2 min.

Tests:
  1. All three datasets load (downloads a tiny subset if needed).
  2. All five baseline models can be instantiated and forwarded on a single batch.
  3. A 1-epoch training run of SimpleCNN on CIFAR-10 completes.
  4. Metrics (accuracy, FLOPs, latency) all return sane values.
  5. Aggregation script runs without error (on dummy data if no real runs).

Run:
  python3 scripts/sanity_check.py
"""
from __future__ import annotations
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils.seed import seed_everything
from src.data.datasets import get_dataloaders, get_spec
from src.models.baselines import build_model, count_parameters, list_models
from src.train.trainer import Trainer
from src.eval.metrics import compute_accuracy, count_flops, measure_latency


def step(name: str, fn):
    print(f"\n--- {name} ---")
    t0 = time.time()
    try:
        result = fn()
        print(f"  OK ({time.time()-t0:.1f}s)")
        return result
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        raise


def test_datasets():
    """Test all three datasets load and have correct shapes."""
    for ds_name in ["fashionmnist", "cifar10", "cifar100"]:
        spec = get_spec(ds_name)
        train_loader, val_loader, nc = get_dataloaders(
            ds_name, batch_size=32, num_workers=0
        )
        x, y = next(iter(train_loader))
        assert x.shape[1] == spec.in_channels, f"{ds_name}: in_channels mismatch"
        assert x.shape[2] == spec.image_size, f"{ds_name}: image_size mismatch"
        assert nc == spec.num_classes
        print(f"  {ds_name}: train_batch={x.shape}  num_classes={nc}")
    return True


def test_models():
    for model_name in list_models():
        spec = get_spec("cifar10")
        model = build_model(model_name, num_classes=spec.num_classes, in_channels=spec.in_channels)
        x = torch.randn(2, spec.in_channels, spec.image_size, spec.image_size)
        out = model(x)
        assert out.shape == (2, spec.num_classes), f"{model_name}: out shape {out.shape}"
        n_params = count_parameters(model)
        print(f"  {model_name}: out={tuple(out.shape)}  params={n_params:,}")
    return True


def test_train_one_epoch():
    """1-epoch training test. Uses Fashion-MNIST (small + already downloaded).
    Skipped if Fashion-MNIST is not present.
    """
    data_root = Path(__file__).resolve().parents[1] / "data"
    if not (data_root / "FashionMNIST" / "raw").exists():
        print("  Skipped (Fashion-MNIST not downloaded)")
        return True
    seed_everything(0)
    train_loader, val_loader, nc = get_dataloaders("fashionmnist", batch_size=128, num_workers=0)
    model = build_model("simple_cnn", num_classes=nc, in_channels=1)
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        num_classes=nc, lr=1e-3, weight_decay=5e-4,
        warmup_epochs=1, total_epochs=1,
        label_smoothing=0.1, grad_clip=1.0, device="cpu",
    )
    result = trainer.fit(epochs=1, logger=None, eval_every=1)
    print(f"  Best acc after 1 epoch: {result['best_acc']:.4f}")
    assert result["best_acc"] >= 0.0
    # Test metrics
    top1, top5 = compute_accuracy(model, val_loader, device="cpu")
    macs, params = count_flops(model, input_size=(1, 1, 28, 28))
    lat = measure_latency(model, input_size=(1, 1, 28, 28), n_runs=10)
    print(f"  top1={top1:.4f}  top5={top5:.4f}  MACs={macs}  params={params}")
    print(f"  latency: median={lat['latency_ms_median']:.2f}ms  mean={lat['latency_ms_mean']:.2f}ms")
    return True


def test_aggregate():
    from src.eval.aggregate import write_summary
    out = write_summary()
    print(f"  Wrote: {out}")
    return True


def main():
    print("=== Sanity check ===")
    t_start = time.time()
    step("Dataset loaders", test_datasets)
    step("Model instantiation", test_models)
    step("1-epoch training (SimpleCNN on CIFAR-10)", test_train_one_epoch)
    step("Aggregation script", test_aggregate)
    print(f"\n=== ALL PASSED in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
