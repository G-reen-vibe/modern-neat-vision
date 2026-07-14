"""Run all baselines across all datasets and multiple seeds.

This is a thin orchestrator that calls scripts/train_baseline.py as a subprocess
so each run is fully isolated (no shared state, no in-process leaks).

By default we run 3 seeds per (dataset, model). Use --seeds to override.
Use --datasets and --models to subset.

Examples:
  # Full sweep (very long)
  python3 scripts/run_baselines.py --seeds 0 1 2

  # Quick smoke test (1 seed, fast dataset only)
  python3 scripts/run_baselines.py --datasets fashionmnist --models simple_cnn --seeds 0

  # ResNet on CIFAR-10 only, 5 seeds
  python3 scripts/run_baselines.py --datasets cifar10 --models resnet18 --seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["fashionmnist", "cifar10", "cifar100"])
    ap.add_argument("--models", nargs="*",
                    default=["simple_cnn", "resnet18", "mobilenetv3_small",
                             "efficientnet_b0", "deit_tiny"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip a run if its final.json already exists.")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--set", nargs="*", default=None,
                    help="Config overrides passed through to train_baseline.py")
    args = ap.parse_args()

    runs = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                run_name = f"{dataset}__{model}__seed{seed}"
                final_path = PROJECT_ROOT / "results" / "runs" / run_name / "final.json"
                if args.skip_existing and final_path.exists():
                    print(f"[SKIP] {run_name} (already done)")
                    continue
                runs.append((dataset, model, seed, run_name))

    if not runs:
        print("Nothing to run (all skipped).")
        return

    print(f"=== Running {len(runs)} experiments ===")
    t_start = time.time()
    for i, (ds, model, seed, name) in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {name}")
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "train_baseline.py"),
            "--dataset", ds, "--model", model, "--seed", str(seed),
        ]
        if args.set:
            cmd += ["--set"] + args.set
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        elapsed = time.time() - t0
        status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
        print(f"  -> {status} in {elapsed:.0f}s")
    total = time.time() - t_start
    print(f"\n=== All done in {total:.0f}s ({total/3600:.1f}h) ===")
    # Auto-aggregate
    print("Aggregating results...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_results.py")])


if __name__ == "__main__":
    main()
