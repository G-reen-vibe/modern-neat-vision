"""Phenotype → torch.nn.Module compilation with shape propagation.

The compiler walks the phenotype DAG in topological order, tracking the
output shape (channels, spatial) of each node. When instantiating a
primitive, it passes the correct in_channels based on the parent's output.

Multi-input nodes use ADDITION (not concatenation) when shapes match.
If shapes don't match, the smaller tensor is adaptively pooled to match
the larger one before addition. This is a "soft merge" that preserves
the spatial structure.

Skip connections are supported when the channel dimensions match. If they
don't, a 1x1 conv projects the skip to the right dimension.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from src.models.dneat.developmental import Phenotype, PhenotypeNode
from src.models.dneat.primitives import PRIMITIVES, TYPE_SPATIAL, TYPE_VECTOR, PrimitiveSpec


def _build_spec_with_hyperparams(template: PrimitiveSpec, hyperparams: dict) -> PrimitiveSpec:
    """Construct a fresh PrimitiveSpec instance with the given hyperparameters.

    The registry stores instances with default hyperparameters. We need to
    reconstruct with the node-specific hyperparameters so that build() uses
    the correct values.
    """
    cls = template.__class__
    try:
        return cls(**hyperparams)
    except TypeError:
        # Some primitives (Identity, GlobalAvgPool, LinearHead) take no args
        return cls()


class _Flatten(nn.Module):
    """Flatten (B, C, 1, 1) -> (B, C). No-op for already-flat tensors."""
    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
            return x.flatten(1)
        return x


class _SpatialAdapter(nn.Module):
    """Adapt a tensor's spatial size and channels to match a target shape.

    Used for merge operations where inputs have different shapes.
    - If channels differ: 1x1 conv.
    - If spatial differs: adaptive avg pool.
    """
    def __init__(self, in_channels: int, out_channels: int, target_spatial: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(target_spatial) if target_spatial > 0 else nn.Identity()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)
        x = self.pool(x)
        return x


class _Merge(nn.Module):
    """Smart merge: add if shapes match, else adapt-then-add."""
    def __init__(self, in_specs: List[Tuple[int, str, int]], out_channels: int, target_spatial: int):
        super().__init__()
        self.adapters = nn.ModuleList([
            _SpatialAdapter(ch, out_channels, target_spatial) for (ch, kind, sp) in in_specs
        ])

    def forward(self, xs):
        if isinstance(xs, (list, tuple)):
            adapted = []
            for x, adapter in zip(xs, self.adapters):
                adapted.append(adapter(x))
            # All same shape now; sum
            out = adapted[0]
            for x in adapted[1:]:
                out = out + x
            return out
        return xs


class DNeatPhenotype(nn.Module):
    """Trainable torch.nn.Module compiled from a D-NEAT phenotype."""

    def __init__(self, phenotype: Phenotype, in_channels: int = 3,
                 num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.phenotype = phenotype
        self.modules_dict = nn.ModuleDict()
        self.merge_modules = nn.ModuleDict()  # node_id -> Merge module
        self.flatten_flags: Dict[int, bool] = {}
        self.node_specs: Dict[int, PrimitiveSpec] = {}  # per-node spec
        self.topo_order: List[int] = []
        self.edges: List[Tuple[int, int]] = list(phenotype.edges)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

        import networkx as nx
        g = phenotype.to_networkx()
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Phenotype is not a DAG")
        self.topo_order = list(nx.topological_sort(g))

        # Track shape flow: node_id -> (channels, kind, spatial)
        # kind: "spatial" or "vector"
        self._shapes: Dict[int, Tuple[int, str, int]] = {}

        for nid in self.topo_order:
            pnode = phenotype.nodes[nid]
            # Get the spec class from the registry, then construct a fresh
            # instance with the node's hyperparameters. The registry instance
            # has default hyperparameters that we must NOT use.
            spec_template = PRIMITIVES[pnode.primitive_name]
            # Reconstruct with the node's hyperparameters
            spec = _build_spec_with_hyperparams(spec_template, pnode.hyperparameters)
            self.node_specs[nid] = spec
            in_edges = [(u, v) for (u, v) in self.edges if v == nid]
            if not in_edges:
                in_ch = in_channels
                self._shapes[nid] = (in_ch, "spatial", image_size)
            else:
                parent_shapes = [self._shapes[u] for u, _ in in_edges]
                if len(parent_shapes) == 1:
                    in_ch = parent_shapes[0][0]
                    parent_kind = parent_shapes[0][1]
                    # If parent is spatial but primitive expects vector (e.g., linear_head
                    # after a conv), flatten first
                    if parent_kind == "spatial" and spec.input_types == [TYPE_VECTOR]:
                        self.flatten_flags[nid] = True
                else:
                    # Multi-input: pick the max-channel parent as the "primary"
                    # and adapt others to match.
                    primary = max(parent_shapes, key=lambda s: s[0])
                    in_ch = primary[0]
                    target_spatial = primary[2] if primary[1] == "spatial" else 1
                    # Build the merge module
                    self.merge_modules[str(nid)] = _Merge(parent_shapes, in_ch, target_spatial)

            try:
                mod = spec.build(
                    in_channels=in_ch,
                    num_classes=num_classes,
                    image_size=image_size,
                )
                self.modules_dict[str(nid)] = mod
            except NotImplementedError:
                self.modules_dict[str(nid)] = nn.Identity()

            # Compute output shape
            if spec.output_type == TYPE_SPATIAL:
                if pnode.primitive_name in ("conv_bn_relu", "dw_sep_conv"):
                    out_ch = pnode.hyperparameters.get("out_channels", in_ch)
                    stride = pnode.hyperparameters.get("stride", 1)
                    parent_spatial = self._shapes[in_edges[0][0]][2] if in_edges else image_size
                    out_spatial = max(1, parent_spatial // stride)
                    self._shapes[nid] = (out_ch, "spatial", out_spatial)
                elif pnode.primitive_name == "max_pool_2x":
                    parent_spatial = self._shapes[in_edges[0][0]][2] if in_edges else image_size
                    out_spatial = max(1, parent_spatial // 2)
                    self._shapes[nid] = (in_ch, "spatial", out_spatial)
                else:
                    self._shapes[nid] = (in_ch, "spatial", image_size)
            elif spec.output_type == TYPE_VECTOR:
                self._shapes[nid] = (in_ch, "vector", 1)
            else:
                self._shapes[nid] = (num_classes, "scalar", 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: Dict[int, torch.Tensor] = {}
        for nid in self.topo_order:
            spec = self.node_specs[nid]
            in_edges = [(u, v) for (u, v) in self.edges if v == nid]
            if not in_edges:
                outputs[nid] = self.modules_dict[str(nid)](x)
            else:
                inputs = [outputs[u] for u, _ in in_edges]
                if len(inputs) == 1:
                    inp = inputs[0]
                    if self.flatten_flags.get(nid, False):
                        if inp.dim() == 4:
                            inp = inp.mean(dim=[2, 3])
                    outputs[nid] = self.modules_dict[str(nid)](inp)
                else:
                    # Multi-input: use merge module
                    merge = self.merge_modules[str(nid)]
                    merged = merge(inputs)
                    # If the primitive expects vector input but merge returned
                    # a 4D tensor (B, C, 1, 1), flatten it
                    if spec.input_types == [TYPE_VECTOR] and merged.dim() == 4:
                        merged = merged.mean(dim=[2, 3])
                    outputs[nid] = self.modules_dict[str(nid)](merged)
        return outputs[self.phenotype.output_node_id]


def compile_phenotype(phenotype: Phenotype, in_channels: int = 3,
                      num_classes: int = 10, image_size: int = 32) -> nn.Module:
    return DNeatPhenotype(phenotype, in_channels=in_channels,
                          num_classes=num_classes, image_size=image_size)

