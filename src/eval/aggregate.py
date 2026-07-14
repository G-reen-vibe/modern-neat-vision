"""Aggregate per-run results across seeds.

Walks results/runs/<dataset>/<model>/<seed>/ directories, reads final.json
from each, and produces a summary table with mean ± std and 95% CI.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Optional
import pandas as pd
from scipy import stats


RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "runs"
SUMMARY_ROOT = Path(__file__).resolve().parents[2] / "results" / "summaries"


def _gather_runs() -> list[dict]:
    """Walk results/runs/ and collect final.json from each run."""
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        final_path = run_dir / "final.json"
        config_path = run_dir / "config.json"
        if not final_path.exists():
            continue
        try:
            with open(final_path) as f:
                final = json.load(f)
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception:
            continue
        run = {
            "run_name": run_dir.name,
            "dataset": cfg.get("dataset", {}).get("name", "unknown"),
            "model": cfg.get("model", {}).get("name", "unknown"),
            "seed": cfg.get("seed", -1),
            **final,
        }
        runs.append(run)
    return runs


def _ci(mean: float, std: float, n: int, alpha: float = 0.05) -> Optional[float]:
    """Half-width of the 95% CI (Student's t). Returns None if n < 2."""
    if n < 2:
        return None
    se = std / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return se * t_crit


def aggregate() -> pd.DataFrame:
    """Aggregate runs into a per-(dataset, model) summary DataFrame."""
    runs = _gather_runs()
    if not runs:
        return pd.DataFrame()
    df = pd.DataFrame(runs)
    # Group by (dataset, model)
    metrics_cols = [c for c in df.columns if c not in
                    ("run_name", "dataset", "model", "seed")]
    agg = df.groupby(["dataset", "model"]).agg(
        {c: ["mean", "std", "count"] for c in metrics_cols if pd.api.types.is_numeric_dtype(df[c])}
    ).reset_index()
    # Flatten columns
    agg.columns = ["_".join([str(x) for x in c if x]).strip("_") for c in agg.columns.values]

    # Add 95% CI half-widths for key metrics
    for c in ("best_acc", "final_acc", "top1_acc"):
        mean_col = f"{c}_mean"
        std_col = f"{c}_std"
        count_col = f"{c}_count"
        if mean_col in agg.columns and std_col in agg.columns and count_col in agg.columns:
            ci_col = f"{c}_ci95"
            agg[ci_col] = agg.apply(
                lambda r: _ci(r[mean_col], r[std_col], int(r[count_col])) or float("nan"),
                axis=1,
            )
    return agg


def write_summary() -> Path:
    """Aggregate and write to results/summaries/summary.csv. Returns path."""
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    df = aggregate()
    out = SUMMARY_ROOT / "summary.csv"
    df.to_csv(out, index=False)
    # Also write a pretty-printed markdown table for human reading
    md = SUMMARY_ROOT / "summary.md"
    if not df.empty:
        # Pick a few useful columns for the markdown summary
        keep = []
        for c in ["dataset", "model",
                  "best_acc_mean", "best_acc_std", "best_acc_ci95",
                  "top1_acc_mean", "top1_acc_std", "top1_acc_ci95",
                  "params_mean", "flops_mean",
                  "train_time_s_mean"]:
            if c in df.columns:
                keep.append(c)
        df_md = df[keep].copy()
        # Format floats
        for c in df_md.columns:
            if df_md[c].dtype.kind in "fc":
                df_md[c] = df_md[c].map(lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "")
        with open(md, "w") as f:
            f.write("# Aggregated results\n\n")
            f.write(df_md.to_markdown(index=False))
            f.write("\n")
    else:
        md.write_text("# Aggregated results\n\nNo runs found.\n")
    return out


if __name__ == "__main__":
    out = write_summary()
    print(f"Wrote summary to {out}")
