"""Direct graph representation for greedy complexification.

Unlike D-NEAT's CPPN+developmental approach, this represents the phenotype
directly as a mutable graph. Growth operations add/modify nodes and edges.

The graph starts minimal (conv -> pool -> head) and grows one operation
at a time. Each operation is a small, local change.

This module provides:
  - GrowthGraph: the mutable graph representation
  - GrowthOp: an enumeration of growth operations
  - apply_operation: apply a growth op to a graph, returning a new graph
  - graph_to_phenotype: convert GrowthGraph to a Phenotype (for compilation)
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import networkx as nx

from src.models.dneat.developmental import Phenotype, PhenotypeNode


# Growth operation types
OPS = [
    "add_conv",        # Insert a conv layer after a random node
    "add_pool",        # Insert a max_pool after a random spatial node
    "add_bn_relu",     # Insert BN+ReLU after a random spatial node
    "add_skip",        # Add a skip connection between two nodes
    "widen",           # Increase channels of a random conv
    "narrow",          # Decrease channels of a random conv
    "change_prim",     # Change a node's primitive type
]


@dataclass
class GrowthGraph:
    """Direct graph representation that supports growth operations."""
    nodes: dict = field(default_factory=dict)  # id -> {primitive, hyperparams, position}
    edges: list = field(default_factory=list)  # [(u, v), ...]
    input_id: int = 0
    output_id: int = 0
    next_id: int = 0
    channel_options: list = field(default_factory=lambda: [8, 16, 24, 32, 48, 64])

    def clone(self) -> "GrowthGraph":
        return copy.deepcopy(self)


def initial_graph() -> GrowthGraph:
    """Create the minimal starting graph: input -> conv -> bn_relu -> pool -> head.

    Slightly stronger than the bare minimum: 2 conv layers + BN to give the
    search a reasonable starting point.
    """
    g = GrowthGraph()
    g.nodes[0] = {"primitive": "identity", "hyperparams": {}, "position": (-1, 0)}
    g.nodes[1] = {"primitive": "conv_bn_relu",
                  "hyperparams": {"out_channels": 32, "kernel_size": 3, "stride": 1, "groups": 1},
                  "position": (-0.5, 0)}
    g.nodes[2] = {"primitive": "bn_relu", "hyperparams": {}, "position": (0, 0)}
    g.nodes[3] = {"primitive": "conv_bn_relu",
                  "hyperparams": {"out_channels": 32, "kernel_size": 3, "stride": 1, "groups": 1},
                  "position": (0.5, 0)}
    g.nodes[4] = {"primitive": "global_avg_pool", "hyperparams": {}, "position": (1, 0)}
    g.nodes[5] = {"primitive": "linear_head", "hyperparams": {}, "position": (1.5, 0)}
    g.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    g.input_id = 0
    g.output_id = 5
    g.next_id = 6
    return g


def apply_operation(graph: GrowthGraph, op: str, rng: random.Random) -> GrowthGraph:
    """Apply a growth operation to a graph. Returns a new graph."""
    g = graph.clone()
    spatial_nodes = [nid for nid, n in g.nodes.items()
                     if n["primitive"] not in ("identity", "linear_head", "global_avg_pool")]
    conv_nodes = [nid for nid, n in g.nodes.items() if n["primitive"] in ("conv_bn_relu", "dw_sep_conv")]
    all_nodes = list(g.nodes.keys())

    if op == "add_conv" and spatial_nodes:
        # Insert a conv after a random spatial node
        parent = rng.choice(spatial_nodes)
        new_id = g.next_id
        g.next_id += 1
        ch = rng.choice(g.channel_options)
        g.nodes[new_id] = {
            "primitive": rng.choice(["conv_bn_relu", "dw_sep_conv"]),
            "hyperparams": {"out_channels": ch, "kernel_size": 3, "stride": 1, "groups": 1},
            "position": (g.nodes[parent]["position"][0] + 0.3, rng.uniform(-0.5, 0.5)),
        }
        # Find edges from parent, insert new node in between
        new_edges = []
        inserted = False
        for (u, v) in g.edges:
            if u == parent and not inserted:
                new_edges.append((u, new_id))
                new_edges.append((new_id, v))
                inserted = True
            else:
                new_edges.append((u, v))
        if not inserted:
            new_edges.append((parent, new_id))
        g.edges = new_edges

    elif op == "add_pool" and spatial_nodes:
        parent = rng.choice(spatial_nodes)
        new_id = g.next_id
        g.next_id += 1
        g.nodes[new_id] = {
            "primitive": "max_pool_2x",
            "hyperparams": {},
            "position": (g.nodes[parent]["position"][0] + 0.2, 0),
        }
        new_edges = []
        inserted = False
        for (u, v) in g.edges:
            if u == parent and not inserted:
                new_edges.append((u, new_id))
                new_edges.append((new_id, v))
                inserted = True
            else:
                new_edges.append((u, v))
        g.edges = new_edges

    elif op == "add_bn_relu" and spatial_nodes:
        parent = rng.choice(spatial_nodes)
        new_id = g.next_id
        g.next_id += 1
        g.nodes[new_id] = {
            "primitive": "bn_relu",
            "hyperparams": {},
            "position": (g.nodes[parent]["position"][0] + 0.1, 0),
        }
        new_edges = []
        inserted = False
        for (u, v) in g.edges:
            if u == parent and not inserted:
                new_edges.append((u, new_id))
                new_edges.append((new_id, v))
                inserted = True
            else:
                new_edges.append((u, v))
        g.edges = new_edges

    elif op == "add_skip" and len(spatial_nodes) >= 2:
        # Add a skip connection between two spatial nodes (avoid cycles)
        # Try a few times to find a non-cycling edge
        for _ in range(10):
            a, b = rng.sample(spatial_nodes, 2)
            # Direction: earlier position -> later position
            if g.nodes[a]["position"][0] > g.nodes[b]["position"][0]:
                a, b = b, a
            if (a, b) not in g.edges and (b, a) not in g.edges:
                # Check for cycle
                test_g = nx.DiGraph()
                for nid in g.nodes:
                    test_g.add_node(nid)
                for (u, v) in g.edges:
                    test_g.add_edge(u, v)
                test_g.add_edge(a, b)
                if nx.is_directed_acyclic_graph(test_g):
                    g.edges.append((a, b))
                    break

    elif op == "widen" and conv_nodes:
        node = rng.choice(conv_nodes)
        current = g.nodes[node]["hyperparams"].get("out_channels", 16)
        idx = g.channel_options.index(current) if current in g.channel_options else 1
        new_idx = min(len(g.channel_options) - 1, idx + 1)
        g.nodes[node]["hyperparams"]["out_channels"] = g.channel_options[new_idx]

    elif op == "narrow" and conv_nodes:
        node = rng.choice(conv_nodes)
        current = g.nodes[node]["hyperparams"].get("out_channels", 16)
        idx = g.channel_options.index(current) if current in g.channel_options else 1
        new_idx = max(0, idx - 1)
        g.nodes[node]["hyperparams"]["out_channels"] = g.channel_options[new_idx]

    elif op == "change_prim" and spatial_nodes:
        node = rng.choice(spatial_nodes)
        current = g.nodes[node]["primitive"]
        alternatives = [p for p in ["conv_bn_relu", "dw_sep_conv", "bn_relu"]
                        if p != current]
        if alternatives:
            new_prim = rng.choice(alternatives)
            # Save old state for potential rollback
            old_prim = g.nodes[node]["primitive"]
            old_hyper = dict(g.nodes[node]["hyperparams"])
            g.nodes[node]["primitive"] = new_prim
            if new_prim in ("conv_bn_relu", "dw_sep_conv"):
                ch = rng.choice(g.channel_options)
                g.nodes[node]["hyperparams"] = {"out_channels": ch, "kernel_size": 3, "stride": 1, "groups": 1}
            else:
                # bn_relu preserves input channels, so clear hyperparams
                g.nodes[node]["hyperparams"] = {}

    return g


def graph_to_phenotype(g: GrowthGraph) -> Phenotype:
    """Convert a GrowthGraph to a Phenotype for compilation."""
    p = Phenotype()
    for nid, node in g.nodes.items():
        p.nodes[nid] = PhenotypeNode(
            node_id=nid,
            primitive_name=node["primitive"],
            hyperparameters=node["hyperparams"],
            position=node["position"],
        )
    p.edges = list(g.edges)
    p.input_node_id = g.input_id
    p.output_node_id = g.output_id
    # Validate and prune
    if not p.is_valid():
        # Try pruning to IO paths
        from src.models.dneat.developmental import _prune_to_io_paths
        p = _prune_to_io_paths(p)
        if not p.is_valid():
            return None
    return p


def graph_features(g: GrowthGraph) -> list:
    """Extract a feature vector from the graph for the policy network."""
    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    # Primitive histogram
    prim_counts = {"identity": 0, "conv_bn_relu": 0, "dw_sep_conv": 0,
                   "max_pool_2x": 0, "bn_relu": 0, "global_avg_pool": 0, "linear_head": 0}
    for n in g.nodes.values():
        prim_counts[n["primitive"]] = prim_counts.get(n["primitive"], 0) + 1
    # Total channels
    total_ch = sum(n["hyperparams"].get("out_channels", 0) for n in g.nodes.values())
    # Depth (longest path)
    try:
        nx_g = nx.DiGraph()
        for nid in g.nodes:
            nx_g.add_node(nid)
        for (u, v) in g.edges:
            nx_g.add_edge(u, v)
        if nx_g.has_path(g.input_id, g.output_id):
            depth = nx.shortest_path_length(nx_g, g.input_id, g.output_id)
        else:
            depth = 0
    except Exception:
        depth = 0
    features = [
        n_nodes / 20.0,
        n_edges / 30.0,
        depth / 10.0,
        total_ch / 200.0,
    ] + [prim_counts[k] / 5.0 for k in ["conv_bn_relu", "dw_sep_conv", "max_pool_2x", "bn_relu", "global_avg_pool"]]
    return features
