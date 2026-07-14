"""CSV + JSON logging for training runs.

Each run produces a directory under results/runs/<run_name>/ containing:
  - config.json: full config used for the run
  - metrics.csv: one row per epoch
  - final.json: final summary (best accuracy, total time, etc.)
  - log.txt: human-readable log

We avoid TensorBoard to save disk space and keep the dependency footprint
small. CSV is easy to load with pandas for aggregation.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.final_path = self.run_dir / "final.json"
        self.config_path = self.run_dir / "config.json"
        self.log_path = self.run_dir / "log.txt"
        self._csv_initialized = False
        self._fieldnames: list[str] = []
        # Save config
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        # Truncate log
        self.log_path.write_text("")

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def log_metrics(self, row: dict[str, Any]) -> None:
        """Append a metrics row. First call defines the column schema."""
        if not self._csv_initialized:
            self._fieldnames = list(row.keys())
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
            self._csv_initialized = True
        # Pad missing fields
        full_row = {k: row.get(k, "") for k in self._fieldnames}
        # Add any new fields (rare; rewrites header)
        new_fields = [k for k in row if k not in self._fieldnames]
        if new_fields:
            self._fieldnames.extend(new_fields)
            # Rewrite entire file with new schema
            rows = []
            with open(self.metrics_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k, "") for k in self._fieldnames})
            full_row = {k: row.get(k, "") for k in self._fieldnames}
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(full_row)

    def log_final(self, summary: dict[str, Any]) -> None:
        with open(self.final_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.log(f"Final summary: {json.dumps(summary, default=str)}")
