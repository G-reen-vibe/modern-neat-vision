"""Greedy complexification with learned policy (REINFORCE).

At each step:
  1. Policy selects an operation based on current graph features
  2. Apply the operation, evaluate the candidate
  3. Reward = accuracy improvement (candidate_acc - current_acc)
  4. Store transition for policy training
  5. Accept candidate if it improves accuracy

After each episode, train the policy with REINFORCE.
"""
from __future__ import annotations
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Subset, DataLoader

from src.data.datasets import get_datasets, get_spec
from src.search.growth import (
    initial_graph, apply_operation, graph_to_phenotype,
    graph_features, graph_hash, OPS, GrowthGraph,
)
from src.search.policy import PolicyTrainer
from src.models.dneat.phenotype import compile_phenotype
from src.train.trainer import Trainer


def evaluate_graph(graph, train_loader, val_loader, num_classes, in_channels, image_size, epochs=2,
                   cache=None, pretrained_weights=None):
    """Evaluate a graph. Returns (accuracy, param_count, time_s, model_state).

    If pretrained_weights is provided (a dict of state_dict from a parent graph),
    try to copy matching weights into the new model before training.
    """
    t0 = time.time()
    gh = graph_hash(graph)
    if cache is not None and gh in cache:
        acc, params = cache[gh]
        return acc, params, 0.0, None
    p = graph_to_phenotype(graph)
    if p is None or not p.is_valid():
        return 0.0, 0, time.time() - t0, None
    try:
        model = compile_phenotype(p, in_channels=in_channels, num_classes=num_classes, image_size=image_size)
        x = torch.randn(2, in_channels, image_size, image_size)
        model(x)
        # Transfer matching weights from pretrained model
        if pretrained_weights is not None:
            model_state = model.state_dict()
            transferred = 0
            for k, v in pretrained_weights.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    transferred += 1
            if transferred > 0:
                model.load_state_dict(model_state)
        params = sum(pp.numel() for pp in model.parameters())
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
                          warmup_epochs=1, total_epochs=epochs, label_smoothing=0.1,
                          grad_clip=1.0, device="cpu")
        result = trainer.fit(epochs, logger=None, eval_every=1)
        acc = result["best_acc"]
        if cache is not None:
            cache[gh] = (acc, params)
        return acc, params, time.time() - t0, model.state_dict()
    except Exception:
        return 0.0, 0, time.time() - t0, None


def policy_search(train_loader, val_loader, num_classes, in_channels, image_size,
                  n_episodes: int = 2, steps_per_episode: int = 4,
                  epochs_per_eval: int = 2, candidates_per_step: int = 2,
                  seed: int = 0, verbose: bool = True):
    """Run greedy complexification with a learned policy.

    At each step, the policy samples K candidate operations. Each is evaluated.
    The best is kept (if it improves). All K transitions are stored with
    reward = candidate_acc - current_acc (so bad candidates get negative reward).
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    trainer = PolicyTrainer(lr=1e-3)
    best_graph_ever = None
    best_acc_ever = 0.0
    eval_cache = {}  # graph_hash -> (acc, params)

    for ep in range(n_episodes):
        current = initial_graph()
        cur_acc, cur_params, _, cur_state = evaluate_graph(
            current, train_loader, val_loader, num_classes, in_channels, image_size, epochs_per_eval, eval_cache
        )
        if verbose:
            print(f"\n--- Episode {ep+1}/{n_episodes} ---")
            print(f"  Step 0: initial acc={cur_acc:.4f} params={cur_params}")

        for step in range(1, steps_per_episode + 1):
            features = graph_features(current)
            # Sample K candidates from the policy
            candidates = []
            for _ in range(candidates_per_step):
                op_idx, op_name = trainer.select_op(current)[:2]
                candidates.append((op_idx, op_name))
            # Also add a random op for exploration
            random_op_idx = rng.randint(0, len(OPS) - 1)
            candidates.append((random_op_idx, OPS[random_op_idx]))

            best_candidate = None
            best_acc = cur_acc
            best_params = cur_params
            best_op_idx = None
            best_state = cur_state
            seen_hashes = set()
            for op_idx, op_name in candidates:
                new_graph = apply_operation(current, OPS[op_idx], rng)
                gh = graph_hash(new_graph)
                if gh in seen_hashes:
                    continue  # skip duplicate
                seen_hashes.add(gh)
                new_acc, new_params, t, new_state = evaluate_graph(
                    new_graph, train_loader, val_loader, num_classes, in_channels, image_size, epochs_per_eval, eval_cache, cur_state
                )
                reward = new_acc - cur_acc
                trainer.store(features, op_idx, reward)
                if verbose:
                    print(f"  Step {step} candidate ({op_name}): acc={new_acc:.4f} (reward={reward:+.4f}) ({t:.0f}s)")
                if new_acc > best_acc:
                    best_acc = new_acc
                    best_candidate = new_graph
                    best_params = new_params
                    best_op_idx = op_idx
                    best_state = new_state

            if best_candidate is not None:
                current = best_candidate
                cur_acc = best_acc
                cur_params = best_params
                cur_state = best_state
                if verbose:
                    print(f"  Step {step}: ACCEPTED {OPS[best_op_idx]} -> acc={cur_acc:.4f}")
            else:
                if verbose:
                    print(f"  Step {step}: no improvement (acc={cur_acc:.4f})")

            if cur_acc > best_acc_ever:
                best_acc_ever = cur_acc
                best_graph_ever = current.clone()

        # Train policy after each episode
        update_result = trainer.update()
        if verbose:
            print(f"  Policy update: {update_result}")

    return best_graph_ever, best_acc_ever


def main():
    print("=== Greedy Complexification with Learned Policy ===")
    train, val, nc = get_datasets("fashionmnist")
    train_sub = Subset(train, list(range(3000)))
    val_sub = Subset(val, list(range(1000)))
    tl = DataLoader(train_sub, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(val_sub, batch_size=64, shuffle=False)
    spec = get_spec("fashionmnist")

    t0 = time.time()
    best_graph, best_acc = policy_search(
        tl, vl, nc, spec.in_channels, spec.image_size,
        n_episodes=3, steps_per_episode=4, epochs_per_eval=2, seed=0, verbose=True,
    )
    print(f"\nSearch time: {time.time()-t0:.0f}s")
    print(f"Best accuracy during search: {best_acc:.4f} ({len(best_graph.nodes)} nodes)")

    # Finetune the best graph with more epochs
    print("\n=== Finetuning best graph (5 epochs) ===")
    finetune_acc, finetune_params, finetune_time, _ = evaluate_graph(
        best_graph, tl, vl, nc, spec.in_channels, spec.image_size, epochs=5, cache=None
    )
    print(f"Finetuned accuracy: {finetune_acc:.4f} (params={finetune_params}, time={finetune_time:.0f}s)")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
