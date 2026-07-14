"""Generic training loop with mixed-precision (CPU-friendly) and label smoothing.

Designed for small-image classification. Tracks:
  - train_loss, train_acc
  - val_loss, val_acc, val_top5_acc (top-5 only meaningful for ≥10 classes)
  - lr per epoch
  - epoch wall time

The trainer is intentionally simple — no distributed, no gradient accumulation,
no EMA. We need correctness and reproducibility, not bells and whistles.
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        total_epochs: int = 100,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip

        self.optimizer = build_optimizer_fn(model, lr, weight_decay)
        steps_per_epoch = len(train_loader)
        self.scheduler = CosineWarmup(
            self.optimizer,
            warmup_steps=warmup_epochs * steps_per_epoch,
            total_steps=total_epochs * steps_per_epoch,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def train_one_epoch(self) -> dict:
        self.model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        return {
            "train_loss": loss_sum / max(1, total),
            "train_acc": correct / max(1, total),
            "lr": self.scheduler.get_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        top5_correct = 0
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            out = self.model(x)
            loss = self.criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            if self.num_classes >= 10:
                top5 = out.topk(min(5, self.num_classes), dim=1).indices
                top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += x.size(0)
        result = {
            "val_loss": loss_sum / max(1, total),
            "val_acc": correct / max(1, total),
        }
        if self.num_classes >= 10:
            result["val_top5_acc"] = top5_correct / max(1, total)
        return result

    def fit(self, epochs: int, logger=None, eval_every: int = 1) -> dict:
        best_acc = 0.0
        best_epoch = -1
        history = []
        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_stats = self.train_one_epoch()
            t_train = time.time() - t0
            row = {"epoch": ep, **train_stats, "train_time_s": t_train}
            if ep % eval_every == 0 or ep == epochs:
                t1 = time.time()
                val_stats = self.evaluate()
                t_val = time.time() - t1
                row.update(val_stats)
                row["val_time_s"] = t_val
                if val_stats["val_acc"] > best_acc:
                    best_acc = val_stats["val_acc"]
                    best_epoch = ep
            history.append(row)
            if logger is not None:
                logger.log_metrics(row)
                logger.log(
                    f"Epoch {ep}: train_loss={train_stats['train_loss']:.4f} "
                    f"train_acc={train_stats['train_acc']:.4f} "
                    f"{'val_acc=' + format(row.get('val_acc', 0), '.4f') + ' ' if 'val_acc' in row else ''}"
                    f"lr={train_stats['lr']:.2e} "
                    f"time={t_train:.1f}s"
                )
        return {"best_acc": best_acc, "best_epoch": best_epoch, "history": history}


# Local imports to avoid circular dependency at module load
from src.train.optimizer import build_optimizer as build_optimizer_fn
from src.train.optimizer import CosineWarmupScheduler as CosineWarmup
