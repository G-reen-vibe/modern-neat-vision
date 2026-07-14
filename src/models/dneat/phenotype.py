"""Phenotype → torch.nn.Module compilation.

Walks the phenotype DAG in topological order, instantiates each primitive
as a torch.nn.Module, and wires up the connections. The result is a single
nn.Module with a forward(x) method that runs the discovered topology.

This is a SCAFFOLD in Phase 2. The actual compilation logic will be
implemented in Phase 4 alongside the developmental program.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from src.models.dneat.developmental import Phenotype
from src.models.dneat.primitives import PRIMITIVES


class DNeatPhenotype(nn.Module):
    """A trainable torch.nn.Module compiled from a D-NEAT phenotype.

    SCAFFOLD: minimal forward() that runs nodes in topological order.
    Will be expanded in Phase 4 to support:
      - Multi-input nodes (concatenation, addition)
      - Skip connections
      - Conditional execution (gating)
    """
    def __init__(self, phenotype: Phenotype, in_channels: int = 3,
                 num_classes: int = 10, image_size: int = 32):
        super().__init__()
        self.phenotype = phenotype
        self.modules_dict = nn.ModuleDict()
        self.topo_order: List[int] = []
        self.edges: List[tuple[int, int]] = list(phenotype.edges)

        # Compute topological order
        import networkx as nx
        g = phenotype.to_networkx()
        self.topo_order = list(nx.topological_sort(g))

        # Instantiate primitives
        for nid in self.topo_order:
            pnode = phenotype.nodes[nid]
            spec = PRIMITIVES[pnode.primitive_name]
            # Use a string key (ModuleDict requires string keys)
            self.modules_dict[str(nid)] = spec.build(
                in_channels=in_channels,
                num_classes=num_classes,
                image_size=image_size,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SCAFFOLD: only handles linear chains (1 in, 1 out per node).
        # Phase 4 will implement general DAG execution with multi-input nodes.
        outputs: Dict[int, torch.Tensor] = {}
        for nid in self.topo_order:
            pnode = self.phenotype.nodes[nid]
            # Find input edges
            in_edges = [(u, v) for (u, v) in self.edges if v == nid]
            if not in_edges:
                # Source node
                outputs[nid] = self.modules_dict[str(nid)](x)
            else:
                # Single input (scaffold)
                u = in_edges[0][0]
                outputs[nid] = self.modules_dict[str(nid)](outputs[u])
        return outputs[self.phenotype.output_node_id]


def compile_phenotype(phenotype: Phenotype, in_channels: int = 3,
                      num_classes: int = 10, image_size: int = 32) -> nn.Module:
    """Compile a phenotype into a trainable torch.nn.Module."""
    return DNeatPhenotype(phenotype, in_channels=in_channels,
                          num_classes=num_classes, image_size=image_size)
