"""Architecture diff: compare Simple CNN vs Growth Graph layer-by-layer.

Both should be: Conv(32)->BN->ReLU->MaxPool->Conv(64)->BN->ReLU->MaxPool->Conv(128)->BN->ReLU->GAP->Linear

Check for:
1. Different weight initialization
2. Extra layers (merge modules, adapters)
3. Different pooling (AdaptiveAvgPool2d vs GlobalAvgPool)
4. Different BN momentum or eps
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.datasets import get_spec
from src.search.growth import initial_graph, graph_to_phenotype
from src.models.dneat.phenotype import compile_phenotype
from src.models.baselines import build_model, count_parameters


def print_model_architecture(name, model):
    print(f"\n=== {name} ===")
    print(f"Total params: {count_parameters(model)}")
    print(f"Modules:")
    for i, (n, m) in enumerate(model.named_modules()):
        if n == "":
            continue
        # Only show leaf modules
        if list(m.children()):
            continue
        params = sum(p.numel() for p in m.parameters())
        if params == 0 and not hasattr(m, 'weight'):
            print(f"  [{i}] {n}: {m.__class__.__name__} (no params)")
        else:
            print(f"  [{i}] {n}: {m.__class__.__name__} (params={params})")
            if hasattr(m, 'weight') and m.weight is not None:
                print(f"        weight shape={tuple(m.weight.shape)}, mean={m.weight.mean().item():.6f}, std={m.weight.std().item():.6f}")
            if hasattr(m, 'bias') and m.bias is not None:
                print(f"        bias shape={tuple(m.bias.shape)}, mean={m.bias.mean().item():.6f}")
            elif hasattr(m, 'bias'):
                print(f"        bias=None")


def main():
    spec = get_spec("fashionmnist")
    # Simple CNN
    simple = build_model("simple_cnn", num_classes=10, in_channels=spec.in_channels)
    print_model_architecture("Simple CNN", simple)

    # Growth Graph
    g = initial_graph()
    p = graph_to_phenotype(g)
    growth = compile_phenotype(p, in_channels=spec.in_channels, num_classes=10, image_size=spec.image_size)
    print_model_architecture("Growth Graph", growth)

    # Check forward pass outputs match in shape
    x = torch.randn(2, spec.in_channels, spec.image_size, spec.image_size)
    out1 = simple(x)
    out2 = growth(x)
    print(f"\nOutput shapes: simple={out1.shape}, growth={out2.shape}")

    # Check if the architectures are truly equivalent by comparing param names
    simple_names = set(n for n, _ in simple.named_parameters())
    growth_names = set(n for n, _ in growth.named_parameters())
    print(f"\nSimple CNN param names ({len(simple_names)}): {sorted(simple_names)[:10]}...")
    print(f"Growth Graph param names ({len(growth_names)}): {sorted(growth_names)[:10]}...")


if __name__ == "__main__":
    main()
