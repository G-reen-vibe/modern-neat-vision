"""CPPN-like genome for D-NEAT.

The genome encodes a *developmental program* that, when executed, grows a
phenotype (a typed DAG over the primitive library). The genome is itself a
small graph of nodes with weights, similar to classical NEAT.

Genome structure (high-level):
  - Input nodes: (x, y, t) — spatial coords and developmental step
  - Hidden nodes: sigmoid/tanh activations
  - Output nodes: produce (a) decision to divide, (b) what primitive to
    differentiate into, (c) connection strengths to neighbors

This is a SCAFFOLD in Phase 2. The actual CPPN evaluation logic will be
implemented in Phase 4 alongside the developmental program.

Key invariants (to be enforced in Phase 4):
  - Genome is a small DAG (<= 50 nodes).
  - Innovation numbers on edges, for crossover compatibility (NEAT-style).
  - Genome is differentiable w.r.t. its weights, enabling mixed evolution
    + gradient refinement.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GenomeNode:
    node_id: int
    kind: str  # "input", "hidden", "output"
    activation: str  # "none", "sigmoid", "tanh", "relu", "sin", "gauss"
    # For CPPN, input nodes are (x, y, t); output nodes are decision variables.


@dataclass
class GenomeEdge:
    edge_id: int            # innovation number (NEAT-style)
    src: int
    dst: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    nodes: Dict[int, GenomeNode] = field(default_factory=dict)
    edges: List[GenomeEdge] = field(default_factory=list)
    next_node_id: int = 0
    next_innovation: int = 0

    def add_input(self, name: str, activation: str = "none") -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        self.nodes[nid] = GenomeNode(nid, "input", activation)
        return nid

    def add_hidden(self, activation: str = "tanh") -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        self.nodes[nid] = GenomeNode(nid, "hidden", activation)
        return nid

    def add_output(self, name: str, activation: str = "sigmoid") -> int:
        nid = self.next_node_id
        self.next_node_id += 1
        self.nodes[nid] = GenomeNode(nid, "output", activation)
        return nid

    def connect(self, src: int, dst: int, weight: Optional[float] = None) -> int:
        eid = self.next_innovation
        self.next_innovation += 1
        if weight is None:
            weight = random.gauss(0, 1.0)
        self.edges.append(GenomeEdge(eid, src, dst, weight))
        return eid

    def mutate_add_node(self) -> None:
        """Split a random edge into two, with a new hidden node in between."""
        if not self.edges:
            return
        edge = random.choice(self.edges)
        if not edge.enabled:
            return
        # Disable original, add node, add two edges
        edge.enabled = False
        nid = self.add_hidden(activation=random.choice(["tanh", "sigmoid", "relu"]))
        self.connect(edge.src, nid, weight=1.0)
        self.connect(nid, edge.dst, weight=edge.weight)

    def mutate_add_edge(self) -> None:
        """Add a new edge between two random nodes (must not create a cycle)."""
        # SCAFFOLD: cycle-check will be added in Phase 4
        candidates = list(self.nodes.keys())
        if len(candidates) < 2:
            return
        src = random.choice(candidates)
        dst = random.choice([n for n in candidates if n != src])
        self.connect(src, dst)

    def mutate_perturb_weights(self, sigma: float = 0.1) -> None:
        for e in self.edges:
            if random.random() < 0.8:
                e.weight += random.gauss(0, sigma)


def minimal_genome() -> Genome:
    """Create a minimal CPPN genome: 3 inputs (x, y, t), 1 hidden, 3 outputs
    (divide, differentiate, connect).
    """
    g = Genome()
    x = g.add_input("x")
    y = g.add_input("y")
    t = g.add_input("t")
    h = g.add_hidden("tanh")
    o_divide = g.add_output("divide")
    o_diff = g.add_output("differentiate")
    o_conn = g.add_output("connect")
    for inp in (x, y, t):
        g.connect(inp, h)
    for out in (o_divide, o_diff, o_conn):
        g.connect(h, out)
    return g
