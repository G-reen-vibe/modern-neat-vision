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
    "add_dw_sep",      # Insert a depthwise-separable conv
    "add_pool",        # Insert a max_pool after a random spatial node
    "add_bn_relu",     # Insert BN+ReLU after a random spatial node
    "add_skip",        # Add a skip connection between two nodes
    "widen",           # Increase channels of a random conv
    "narrow",          # Decrease channels of a random conv
    "prune",           # Remove a non-essential node
    "add_block",       # Insert a conv+bn_relu block (compound op)
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
    """Create the starting graph matching Simple CNN's architecture.

    3 conv layers with increasing channels (32->64->128) + max pooling.
    This gives the search a strong starting point comparable to the
    hand-designed Simple CNN baseline.
    """
    g = GrowthGraph()
    g.nodes[0] = {"primitive": "identity", "hyperparams": {}, "position": (-2, 0)}
    g.nodes[1] = {"primitive": "conv_bn_relu",
                  "hyperparams": {"out_channels": 32, "kernel_size": 3, "stride": 1, "groups": 1},
                  "position": (-1.5, 0)}
    g.nodes[2] = {"primitive": "max_pool_2x", "hyperparams": {}, "position": (-1, 0)}
    g.nodes[3] = {"primitive": "conv_bn_relu",
                  "hyperparams": {"out_channels": 64, "kernel_size": 3, "stride": 1, "groups": 1},
                  "position": (-0.5, 0)}
    g.nodes[4] = {"primitive": "max_pool_2x", "hyperparams": {}, "position": (0, 0)}
    g.nodes[5] = {"primitive": "conv_bn_relu",
                  "hyperparams": {"out_channels": 128, "kernel_size": 3, "stride": 1, "groups": 1},
                  "position": (0.5, 0)}
    g.nodes[6] = {"primitive": "global_avg_pool", "hyperparams": {}, "position": (1, 0)}
    g.nodes[7] = {"primitive": "linear_head", "hyperparams": {}, "position": (1.5, 0)}
    g.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    g.input_id = 0
    g.output_id = 7
    g.next_id = 8
    return g


def _get_upstream_channels(g: GrowthGraph, nid: int) -> int:
    """Get the channel count of the upstream conv node (for bn_relu/max_pool)."""
    for (u, v) in g.edges:
        if v == nid:
            ch = g.nodes[u]["hyperparams"].get("out_channels", 0)
            if ch > 0:
                return ch
            return _get_upstream_channels(g, u)
    return 0


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

    elif op == "add_dw_sep" and spatial_nodes:
        # Insert a depthwise-separable conv after a random spatial node
        parent = rng.choice(spatial_nodes)
        new_id = g.next_id
        g.next_id += 1
        ch = rng.choice(g.channel_options)
        g.nodes[new_id] = {
            "primitive": "dw_sep_conv",
            "hyperparams": {"out_channels": ch, "stride": 1},
            "position": (g.nodes[parent]["position"][0] + 0.3, rng.uniform(-0.5, 0.5)),
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
        # Add a skip connection between two spatial nodes with matching channels.
        # Only connect nodes where the channel count matches to avoid shape issues.
        # Try a few times to find a valid, non-cycling edge.
        for _ in range(10):
            a, b = rng.sample(spatial_nodes, 2)
            # Check channel compatibility
            ch_a = g.nodes[a]["hyperparams"].get("out_channels", 0)
            ch_b = g.nodes[b]["hyperparams"].get("out_channels", 0)
            # For bn_relu/max_pool, channels = input channels, so check upstream
            if ch_a == 0:
                ch_a = _get_upstream_channels(g, a)
            if ch_b == 0:
                ch_b = _get_upstream_channels(g, b)
            if ch_a != ch_b:
                continue
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

    elif op == "prune":
        # Remove a non-essential node (not input/output, has exactly 1 in + 1 out).
        # Reconnect its parent to its child to maintain the path.
        removable = []
        for nid in g.nodes:
            if nid in (g.input_id, g.output_id):
                continue
            in_edges = [(u, v) for (u, v) in g.edges if v == nid]
            out_edges = [(u, v) for (u, v) in g.edges if u == nid]
            if len(in_edges) == 1 and len(out_edges) == 1:
                removable.append(nid)
        if removable:
            node_to_remove = rng.choice(removable)
            in_edge = [(u, v) for (u, v) in g.edges if v == node_to_remove][0]
            out_edge = [(u, v) for (u, v) in g.edges if u == node_to_remove][0]
            new_edges = [(u, v) for (u, v) in g.edges
                         if u != node_to_remove and v != node_to_remove]
            new_edges.append((in_edge[0], out_edge[1]))
            g.edges = new_edges
            del g.nodes[node_to_remove]

    elif op == "add_block" and spatial_nodes:
        # Insert a conv+bn_relu block after a random spatial node.
        # This is a compound operation: two nodes inserted together.
        parent = rng.choice(spatial_nodes)
        ch = rng.choice(g.channel_options)
        conv_id = g.next_id
        g.next_id += 1
        bn_id = g.next_id
        g.next_id += 1
        g.nodes[conv_id] = {
            "primitive": "conv_bn_relu",
            "hyperparams": {"out_channels": ch, "kernel_size": 3, "stride": 1, "groups": 1},
            "position": (g.nodes[parent]["position"][0] + 0.2, rng.uniform(-0.3, 0.3)),
        }
        g.nodes[bn_id] = {
            "primitive": "bn_relu",
            "hyperparams": {},
            "position": (g.nodes[parent]["position"][0] + 0.3, rng.uniform(-0.3, 0.3)),
        }
        new_edges = []
        inserted = False
        for (u, v) in g.edges:
            if u == parent and not inserted:
                new_edges.append((u, conv_id))
                new_edges.append((conv_id, bn_id))
                new_edges.append((bn_id, v))
                inserted = True
            else:
                new_edges.append((u, v))
        if not inserted:
            new_edges.append((parent, conv_id))
            new_edges.append((conv_id, bn_id))
        g.edges = new_edges

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


def graph_to_dict(g: GrowthGraph) -> dict:
    """Serialize a GrowthGraph to a dict (JSON-compatible)."""
    return {
        "nodes": {str(k): v for k, v in g.nodes.items()},
        "edges": [list(e) for e in g.edges],
        "input_id": g.input_id,
        "output_id": g.output_id,
        "next_id": g.next_id,
    }


def graph_from_dict(d: dict) -> GrowthGraph:
    """Deserialize a GrowthGraph from a dict."""
    g = GrowthGraph()
    g.nodes = {int(k): v for k, v in d["nodes"].items()}
    g.edges = [tuple(e) for e in d["edges"]]
    g.input_id = d["input_id"]
    g.output_id = d["output_id"]
    g.next_id = d["next_id"]
    return g


def graph_hash(g: GrowthGraph) -> str:
    """A hash of the graph's structure (nodes + edges). Useful for dedup."""
    node_strs = []
    for nid in sorted(g.nodes.keys()):
        n = g.nodes[nid]
        node_strs.append(f"{nid}:{n['primitive']}:{n['hyperparams'].get('out_channels', '')}")
    edge_strs = [f"{u}->{v}" for (u, v) in sorted(g.edges)]
    return "|".join(node_strs) + "||" + "|".join(edge_strs)


def graph_features(g: GrowthGraph) -> list:
    """Extract a feature vector from the graph for the policy network.

    Features:
      - Normalized node count, edge count, depth, total channels
      - Per-primitive counts (5 types)
      - Mean channels per conv layer
      - Fraction of nodes that are conv vs pool vs bn
      - Graph density (edges / max possible)
    """
    n_nodes = len(g.nodes)
    n_edges = len(g.edges)
    prim_counts = {"identity": 0, "conv_bn_relu": 0, "dw_sep_conv": 0,
                   "max_pool_2x": 0, "bn_relu": 0, "global_avg_pool": 0, "linear_head": 0}
    for n in g.nodes.values():
        prim_counts[n["primitive"]] = prim_counts.get(n["primitive"], 0) + 1
    total_ch = sum(n["hyperparams"].get("out_channels", 0) for n in g.nodes.values())
    n_conv = prim_counts["conv_bn_relu"] + prim_counts["dw_sep_conv"]
    mean_ch = total_ch / max(1, n_conv)
    # Depth
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
    # Density
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max(1, max_edges)
    features = [
        n_nodes / 20.0,
        n_edges / 30.0,
        depth / 10.0,
        total_ch / 200.0,
        mean_ch / 64.0,
        density,
    ] + [prim_counts[k] / 5.0 for k in ["conv_bn_relu", "dw_sep_conv", "max_pool_2x", "bn_relu", "global_avg_pool"]]
    return features
