"""Developmental program: genome → phenotype.

The developmental program is a graph grammar that grows the phenotype over
a fixed number of developmental steps. At each step:

  1. Each cell in the current phenotype queries the CPPN genome with its
     spatial coordinates (x, y) and the current developmental step t.
  2. The CPPN outputs:
     - divide probability: should this cell split into two?
     - differentiation: which primitive from the library to instantiate?
     - connection strength: how strongly to connect to neighbors.
  3. The grammar applies these decisions, growing the phenotype.

The result is a typed DAG over the primitive library, which is then
compiled into a torch.nn.Module by phenotype.py.

This is a SCAFFOLD in Phase 2. The actual grammar and CPPN evaluation
will be implemented in Phase 4.

Key design decisions (to be revisited):
  - Developmental grid: 4x4 (16 cells), 5 steps. Total phenotype size
    will be ~16-64 nodes.
  - Connection radius: cells connect to neighbors within Manhattan
    distance 2.
  - Stability regularizer: each developmental step is run twice with
    Gaussian noise added to the CPPN's outputs; the Kullback-Leibler
    divergence between the two resulting phenotypes is minimized.
    This is the *research bet* of D-NEAT (see DECISION.md §1).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import networkx as nx

from src.models.dneat.genome import Genome
from src.models.dneat.primitives import PRIMITIVES


@dataclass
class PhenotypeNode:
    node_id: int
    primitive_name: str
    hyperparameters: dict
    # Position in the developmental grid (for visualization/debugging)
    position: Tuple[int, int]


@dataclass
class Phenotype:
    """A typed DAG over the primitive library."""
    nodes: Dict[int, PhenotypeNode] = field(default_factory=dict)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    input_node_id: Optional[int] = None
    output_node_id: Optional[int] = None

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for nid, node in self.nodes.items():
            g.add_node(nid, **{"primitive": node.primitive_name,
                                "hyperparameters": node.hyperparameters,
                                "position": node.position})
        for u, v in self.edges:
            g.add_edge(u, v)
        return g

    def is_valid(self) -> bool:
        """Check that the phenotype is a valid DAG with one input and one output."""
        if not self.nodes:
            return False
        if self.input_node_id is None or self.output_node_id is None:
            return False
        g = self.to_networkx()
        if not nx.is_directed_acyclic_graph(g):
            return False
        # Every node must be reachable from input and must reach output
        preds = nx.ancestors(g, self.output_node_id)
        succs = nx.descendants(g, self.input_node_id)
        if preds != set(self.nodes.keys()) - {self.output_node_id}:
            return False
        if succs != set(self.nodes.keys()) - {self.input_node_id}:
            return False
        return True


def develop(genome: Genome, grid_size: int = 4, steps: int = 5) -> Phenotype:
    """Run the developmental program to produce a phenotype.

    SCAFFOLD: This is a stub. For Phase 2, we just return a fixed 3-node
    phenotype (input → ConvBNReLU → GlobalAvgPool → LinearHead → output)
    so the rest of the pipeline can be tested.

    Phase 4 will implement the actual CPPN evaluation + graph grammar.
    """
    p = Phenotype()
    p.input_node_id = 0
    p.nodes[0] = PhenotypeNode(0, "identity", {}, (0, 0))
    p.nodes[1] = PhenotypeNode(1, "conv_bn_relu", {"out_channels": 64, "kernel_size": 3, "stride": 1, "groups": 1}, (1, 0))
    p.nodes[2] = PhenotypeNode(2, "global_avg_pool", {}, (2, 0))
    p.nodes[3] = PhenotypeNode(3, "linear_head", {}, (3, 0))
    p.output_node_id = 3
    p.edges = [(0, 1), (1, 2), (2, 3)]
    return p


def stability_score(genome: Genome, grid_size: int = 4, steps: int = 5,
                    noise_sigma: float = 0.1, n_samples: int = 2) -> float:
    """Compute the stability score of a genome.

    Runs the developmental program n_samples times with Gaussian noise
    added to the CPPN's outputs, and returns the average pairwise
    distance between resulting phenotypes (lower = more stable).

    SCAFFOLD: returns 0.0 in Phase 2. Will be implemented in Phase 4.
    """
    return 0.0
