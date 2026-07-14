"""Aggregate per-run results into a summary table.

Walks results/runs/*, reads final.json, produces:
  results/summaries/summary.csv  (full table)
  results/summaries/summary.md   (pretty markdown)
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.aggregate import write_summary


def main():
    out = write_summary()
    print(f"Wrote summary to {out}")
    print(f"Pretty version: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
