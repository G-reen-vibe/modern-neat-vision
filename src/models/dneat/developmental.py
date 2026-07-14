"""Developmental program: genome → phenotype via a graph grammar.

Process:
  1. Start with a single "stem cell" at position (0, 0).
  2. At each developmental step, each existing cell queries the CPPN with
     its (x, y, t) coordinates. The CPPN outputs:
       - divide_prob: if > threshold, the cell divides into two daughter
         cells (one stays, one moves to a neighboring empty slot).
       - primitive_idx: which primitive from the library to instantiate.
       - connect_strength: how strongly to connect to existing neighbors.
  3. After all divisions, cells form connections based on connect_strength
     and proximity (Manhattan distance <= connect_radius).
  4. The final cell graph is the phenotype.

To enforce a meaningful topology:
  - The input cell is always Identity (a "stem" that just receives x).
  - The output cell is always LinearHead (terminal classifier).
  - Intermediate cells are differentiated based on CPPN output.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx

from src.models.dneat.genome import Genome, minimal_genome
from src.models.dneat.cppn import evaluate_cppn
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


@dataclass
class Cell:
    cell_id: int
    position: Tuple[float, float]  # (x, y) in [-1, 1]^2
    primitive_name: str
    hyperparameters: dict
    parent_id: Optional[int] = None
    born_at_step: int = 0


@dataclass
class DevelopmentalConfig:
    grid_resolution: int = 3  # 3x3 grid of possible positions
    max_steps: int = 2
    divide_threshold: float = 0.5
    connect_radius: float = 1.5
    min_cells: int = 3
    max_cells: int = 6
    # Primitive vocabulary the CPPN can choose from (excluding input/output).
    primitive_choices: List[str] = field(default_factory=lambda: [
        "conv_bn_relu", "dw_sep_conv", "max_pool_2x", "bn_relu", "global_avg_pool",
    ])
    noise_sigma: float = 0.0
    output_attraction_radius: float = 1.5
    # Cap on proximity-based edges per cell.
    max_proximity_edges_per_cell: int = 1
    # Default out_channels for conv primitives. Smaller = faster.
    default_conv_channels: int = 16


def _grid_positions(resolution: int) -> List[Tuple[float, float]]:
    """Generate grid positions in [-1, 1]^2."""
    if resolution == 1:
        return [(0.0, 0.0)]
    coords = np.linspace(-1, 1, resolution)
    return [(float(x), float(y)) for y in coords for x in coords]


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def develop(genome: Genome, config: Optional[DevelopmentalConfig] = None,
            seed: Optional[int] = None) -> Phenotype:
    """Run the developmental program to produce a phenotype."""
    if config is None:
        config = DevelopmentalConfig()
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    # Ensure genome has input_ids / output_ids
    if not hasattr(genome, "input_ids"):
        # Fallback: synthesize from node kinds
        input_ids = [nid for nid, n in genome.nodes.items() if n.kind == "input"]
        output_ids = [nid for nid, n in genome.nodes.items() if n.kind == "output"]
        genome.input_ids = input_ids
        genome.output_ids = output_ids

    cells: List[Cell] = []
    next_cell_id = 0

    # Place input cell at position (-1, 0) — leftmost
    cells.append(Cell(
        cell_id=next_cell_id, position=(-1.0, 0.0),
        primitive_name="identity", hyperparameters={},
        parent_id=None, born_at_step=0,
    ))
    next_cell_id += 1

    # Place a "seed" cell at (0, 0) that will be the first to develop
    seed_cell = Cell(
        cell_id=next_cell_id, position=(0.0, 0.0),
        primitive_name="conv_bn_relu",
        hyperparameters={"out_channels": config.default_conv_channels, "kernel_size": 3, "stride": 1, "groups": 1},
        parent_id=0, born_at_step=0,
    )
    cells.append(seed_cell)
    next_cell_id += 1

    grid = _grid_positions(config.grid_resolution)
    # Occupied positions
    occupied = {c.position for c in cells}

    # Developmental steps
    for step in range(1, config.max_steps + 1):
        t = step / config.max_steps  # normalize time to [0, 1]
        new_cells: List[Cell] = []
        for cell in list(cells):
            if cell.primitive_name in ("identity", "linear_head"):
                continue  # input/output cells don't divide
            x, y = cell.position
            # Query CPPN
            outputs = evaluate_cppn(genome, x, y, t, genome.input_ids, genome.output_ids)
            # Add noise for stability testing
            if config.noise_sigma > 0:
                outputs = [o + rng.gauss(0, config.noise_sigma) for o in outputs]
            divide_prob = 1.0 / (1.0 + math.exp(-outputs[0]))  # sigmoid
            # Differentiation: PRIMARILY driven by CPPN output (80%),
            # with a small position bias (20%) to encourage depth-wise structure.
            diff_val = outputs[1]
            diff_norm = (math.tanh(diff_val) + 1) / 2  # [0, 1]
            x_norm = (x + 1) / 2
            choice_val = 0.8 * diff_norm + 0.2 * x_norm
            n_choices = len(config.primitive_choices)
            # Add stochastic noise to choice_val so daughter cells don't all
            # differentiate identically. The noise is seeded by the developmental
            # seed, so development is reproducible for a given (genome, seed).
            choice_val = choice_val + rng.uniform(-0.15, 0.15)
            choice_val = max(0.0, min(0.999, choice_val))
            prim_idx = min(n_choices - 1, int(choice_val * n_choices))
            prim_name = config.primitive_choices[prim_idx]
            # Channel count: use the CPPN's connect output (3rd output) to
            # determine channel scaling. This makes channel count evolvable.
            connect_val = outputs[2] if len(outputs) > 2 else 0.0
            connect_norm = (math.tanh(connect_val) + 1) / 2  # [0, 1]
            # Add noise to channel count too
            connect_norm = max(0.0, min(1.0, connect_norm + rng.uniform(-0.1, 0.1)))
            n_channels = max(8, int(config.default_conv_channels * (0.5 + connect_norm * 2.5)))

            # Decide division
            if (divide_prob > config.divide_threshold
                    and len(cells) + len(new_cells) < config.max_cells):
                # Find an empty neighboring position
                candidates = []
                for gx, gy in grid:
                    if (gx, gy) in occupied:
                        continue
                    dist = abs(gx - x) + abs(gy - y)
                    if dist <= 2.0 / config.grid_resolution * 2:  # neighbor
                        candidates.append(((gx, gy), dist))
                if candidates:
                    candidates.sort(key=lambda c: c[1])
                    new_pos = candidates[0][0]
                    occupied.add(new_pos)
                    # Differentiate
                    hyperparams = _default_hyperparams(prim_name, n_channels)
                    daughter = Cell(
                        cell_id=next_cell_id, position=new_pos,
                        primitive_name=prim_name, hyperparameters=hyperparams,
                        parent_id=cell.cell_id, born_at_step=step,
                    )
                    new_cells.append(daughter)
                    next_cell_id += 1
        cells.extend(new_cells)
        if len(cells) >= config.max_cells:
            break

    # Place output cell at (1, 0) — rightmost
    output_cell = Cell(
        cell_id=next_cell_id, position=(1.0, 0.0),
        primitive_name="global_avg_pool", hyperparameters={},
        parent_id=None, born_at_step=config.max_steps,
    )
    cells.append(output_cell)
    next_cell_id += 1

    output_head = Cell(
        cell_id=next_cell_id, position=(1.5, 0.0),
        primitive_name="linear_head", hyperparameters={},
        parent_id=output_cell.cell_id, born_at_step=config.max_steps,
    )
    cells.append(output_head)

    # Build edges: connect cells based on proximity and parent relationships
    edges: List[Tuple[int, int]] = []
    # Parent-child edges (developmental lineage) — these are always kept.
    for c in cells:
        if c.parent_id is not None:
            edges.append((c.parent_id, c.cell_id))
    # Proximity-based edges (within connect_radius, not already connected).
    # We cap the number of proximity edges per cell to avoid O(N^2) edge
    # density, which makes the phenotype's forward pass very slow.
    cell_by_id = {c.cell_id: c for c in cells}
    existing = set(edges)
    # Collect proximity candidates per cell, sorted by distance.
    proximity_candidates: dict[int, list[tuple[float, int]]] = {c.cell_id: [] for c in cells}
    for i, c1 in enumerate(cells):
        for c2 in cells[i+1:]:
            if (c1.cell_id, c2.cell_id) in existing or (c2.cell_id, c1.cell_id) in existing:
                continue
            if c2.primitive_name == "identity":
                continue
            if c1.primitive_name == "linear_head":
                continue
            dist = math.sqrt((c1.position[0]-c2.position[0])**2 + (c1.position[1]-c2.position[1])**2)
            if dist <= config.connect_radius:
                proximity_candidates[c1.cell_id].append((dist, c2.cell_id))
                proximity_candidates[c2.cell_id].append((dist, c1.cell_id))
    # For each cell, add up to max_proximity_edges_per_cell edges, nearest first.
    added = set()
    for cid, cands in proximity_candidates.items():
        cands.sort()
        for dist, other_id in cands[:config.max_proximity_edges_per_cell]:
            c1 = cell_by_id[cid]
            c2 = cell_by_id[other_id]
            # Direction: leftmost → rightmost (by x coord)
            if c1.position[0] <= c2.position[0]:
                e = (c1.cell_id, c2.cell_id)
            else:
                e = (c2.cell_id, c1.cell_id)
            if e not in added and (e[1], e[0]) not in added:
                edges.append(e)
                added.add(e)

    # Ensure the output cell (global_avg_pool) connects to cells near it.
    # This guarantees the phenotype has a path from input to output.
    output_pool_id = output_cell.cell_id
    existing = set(edges)
    for c in cells:
        if c.cell_id in (output_pool_id, output_head.cell_id, cells[0].cell_id):
            continue
        if c.primitive_name in ("identity", "linear_head"):
            continue
        if (c.cell_id, output_pool_id) in existing or (output_pool_id, c.cell_id) in existing:
            continue
        dist = math.sqrt((c.position[0]-output_cell.position[0])**2 + (c.position[1]-output_cell.position[1])**2)
        if dist <= config.output_attraction_radius:
            edges.append((c.cell_id, output_pool_id))

    # Build the phenotype
    p = Phenotype()
    for c in cells:
        p.nodes[c.cell_id] = PhenotypeNode(
            node_id=c.cell_id,
            primitive_name=c.primitive_name,
            hyperparameters=c.hyperparameters,
            position=c.position,
        )
    p.edges = edges
    p.input_node_id = cells[0].cell_id  # identity
    p.output_node_id = output_head.cell_id  # linear_head

    # Break cycles: greedily remove edges that create cycles, in reverse
    # order of edge weight (proximity-based edges have lower priority than
    # parent-child edges, since they were added later).
    p = _break_cycles(p)
    # Prune: keep only nodes on some path from input to output.
    # Disconnected cells "die off" — a biologically motivated cleanup.
    p = _prune_to_io_paths(p)
    if not p.nodes or p.input_node_id not in p.nodes or p.output_node_id not in p.nodes:
        return _fallback_phenotype()
    if not p.is_valid():
        return _fallback_phenotype()
    return p


def _break_cycles(p: Phenotype) -> Phenotype:
    """Greedily remove edges that create cycles. Preserves earlier edges."""
    import networkx as nx
    kept_edges = []
    g = nx.DiGraph()
    for nid in p.nodes:
        g.add_node(nid)
    for (u, v) in p.edges:
        g.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(g):
            g.remove_edge(u, v)
        else:
            kept_edges.append((u, v))
    p.edges = kept_edges
    return p


def _prune_to_io_paths(p: Phenotype) -> Phenotype:
    """Keep only nodes that lie on some path from input to output."""
    import networkx as nx
    g = p.to_networkx()
    if p.input_node_id not in g or p.output_node_id not in g:
        return p
    # Compute reachable from input, and ancestors of output
    reachable_from_input = nx.descendants(g, p.input_node_id) | {p.input_node_id}
    can_reach_output = nx.ancestors(g, p.output_node_id) | {p.output_node_id}
    on_path = reachable_from_input & can_reach_output
    # Filter
    new = Phenotype()
    new.input_node_id = p.input_node_id
    new.output_node_id = p.output_node_id
    for nid, node in p.nodes.items():
        if nid in on_path:
            new.nodes[nid] = node
    new.edges = [(u, v) for (u, v) in p.edges if u in on_path and v in on_path]
    return new


def _default_hyperparams(name: str, channels: int = 32) -> dict:
    if name == "conv_bn_relu":
        return {"out_channels": channels, "kernel_size": 3, "stride": 1, "groups": 1}
    if name == "dw_sep_conv":
        return {"out_channels": channels, "stride": 1}
    return {}


def _fallback_phenotype() -> Phenotype:
    """Minimal valid phenotype: input → conv → pool → head → output."""
    p = Phenotype()
    p.nodes[0] = PhenotypeNode(0, "identity", {}, (-1, 0))
    p.nodes[1] = PhenotypeNode(1, "conv_bn_relu", {"out_channels": 64, "kernel_size": 3, "stride": 1, "groups": 1}, (0, 0))
    p.nodes[2] = PhenotypeNode(2, "global_avg_pool", {}, (1, 0))
    p.nodes[3] = PhenotypeNode(3, "linear_head", {}, (2, 0))
    p.input_node_id = 0
    p.output_node_id = 3
    p.edges = [(0, 1), (1, 2), (2, 3)]
    return p


def stability_score(genome: Genome, config: Optional[DevelopmentalConfig] = None,
                    n_samples: int = 3) -> float:
    """Compute stability: average pairwise phenotype distance under noise.

    Lower = more stable. Returns the fraction of (node, edge) differences
    between phenotype pairs developed with different noise seeds.
    """
    if config is None:
        config = DevelopmentalConfig()
    if config.noise_sigma == 0:
        config = DevelopmentalConfig(**{**config.__dict__, "noise_sigma": 0.1})
    phenotypes = []
    for s in range(n_samples):
        p = develop(genome, config, seed=s)
        phenotypes.append(p)
    # Pairwise distance: compare both graph structure AND primitive assignments.
    # Two phenotypes with the same node IDs but different primitives at those
    # IDs are considered different.
    total_diff = 0.0
    pairs = 0
    for i in range(len(phenotypes)):
        for j in range(i+1, len(phenotypes)):
            p1, p2 = phenotypes[i], phenotypes[j]
            # Node structure: compare (node_id, primitive_name) pairs
            n1 = {(nid, n.primitive_name) for nid, n in p1.nodes.items()}
            n2 = {(nid, n.primitive_name) for nid, n in p2.nodes.items()}
            node_diff = len(n1.symmetric_difference(n2)) / max(1, len(n1 | n2))
            # Edge structure
            e1 = set(p1.edges)
            e2 = set(p2.edges)
            edge_diff = len(e1.symmetric_difference(e2)) / max(1, len(e1 | e2))
            total_diff += 0.5 * (node_diff + edge_diff)
            pairs += 1
    return total_diff / max(1, pairs)
