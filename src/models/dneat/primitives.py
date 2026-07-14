"""Typed primitive library for D-NEAT.

Each primitive is a differentiable operation that can be used as a node in
the discovered topology. The library is deliberately small — we want
evolution to discover wiring patterns, not infinite variations of operations.

Primitive types (for type-checking during development):
  - Spatial[C, H, W]  : a feature map
  - Vector[C]         : a vector (e.g., after global pooling)
  - Scalar[]          : a single number (rare; for gating)

Each primitive declares:
  - name
  - input_types (list of types it accepts)
  - output_type
  - hyperparameters (with defaults)
  - a build() method returning a torch.nn.Module

This file is a SCAFFOLD in Phase 2. The actual logic will be added in Phase 4.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Type system (lightweight, just for documentation and runtime checks)
TYPE_SPATIAL = "Spatial"     # (C, H, W)
TYPE_VECTOR = "Vector"       # (C,)
TYPE_SCALAR = "Scalar"       # ()


@dataclass
class PrimitiveSpec:
    name: str
    input_types: List[str]
    output_type: str
    hyperparameters: dict = field(default_factory=dict)

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        raise NotImplementedError


class ConvBNReLU(PrimitiveSpec):
    """Conv2d + BN + ReLU. The workhorse of CNNs."""
    def __init__(self, out_channels: int = 64, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__(
            name="conv_bn_relu",
            input_types=[TYPE_SPATIAL],
            output_type=TYPE_SPATIAL,
            hyperparameters={
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "groups": groups,
            },
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        from torch.nn import Sequential
        out_c = self.hyperparameters["out_channels"]
        k = self.hyperparameters["kernel_size"]
        s = self.hyperparameters["stride"]
        g = self.hyperparameters["groups"]
        return Sequential(
            nn.Conv2d(in_channels, out_c, k, stride=s, padding=k // 2, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(PrimitiveSpec):
    """Depthwise + pointwise conv. MobileNet-style."""
    def __init__(self, out_channels: int = 64, stride: int = 1):
        super().__init__(
            name="dw_sep_conv",
            input_types=[TYPE_SPATIAL],
            output_type=TYPE_SPATIAL,
            hyperparameters={"out_channels": out_channels, "stride": stride},
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        out_c = self.hyperparameters["out_channels"]
        s = self.hyperparameters["stride"]
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=s, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class SelfAttention(PrimitiveSpec):
    """Multi-head self-attention over spatial tokens."""
    def __init__(self, num_heads: int = 4, embed_dim: int = 64):
        super().__init__(
            name="self_attention",
            input_types=[TYPE_SPATIAL],
            output_type=TYPE_SPATIAL,
            hyperparameters={"num_heads": num_heads, "embed_dim": embed_dim},
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        # Placeholder: implemented in Phase 4
        raise NotImplementedError("SelfAttention.build will be implemented in Phase 4")


class GlobalAvgPool(PrimitiveSpec):
    """Spatial → vector. (C, H, W) -> (C,)."""
    def __init__(self):
        super().__init__(
            name="global_avg_pool",
            input_types=[TYPE_SPATIAL],
            output_type=TYPE_VECTOR,
            hyperparameters={},
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        return nn.AdaptiveAvgPool2d(1)


class LinearHead(PrimitiveSpec):
    """Vector → class logits. Terminal node of the topology."""
    def __init__(self):
        super().__init__(
            name="linear_head",
            input_types=[TYPE_VECTOR],
            output_type=TYPE_SCALAR,  # actually logits over classes
            hyperparameters={},
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        return nn.Linear(in_channels, num_classes)


class Identity(PrimitiveSpec):
    """Pass-through. Useful for skip connections."""
    def __init__(self):
        super().__init__(
            name="identity",
            input_types=[TYPE_SPATIAL],
            output_type=TYPE_SPATIAL,
            hyperparameters={},
        )

    def build(self, in_channels: int, num_classes: int = 10,
              image_size: int = 32) -> nn.Module:
        return nn.Identity()


# Library registry
PRIMITIVES: dict[str, PrimitiveSpec] = {}


def _register(p: PrimitiveSpec) -> PrimitiveSpec:
    PRIMITIVES[p.name] = p
    return p


_register(ConvBNReLU())
_register(DepthwiseSeparableConv())
_register(SelfAttention())
_register(GlobalAvgPool())
_register(LinearHead())
_register(Identity())


def list_primitives() -> List[str]:
    return list(PRIMITIVES.keys())
