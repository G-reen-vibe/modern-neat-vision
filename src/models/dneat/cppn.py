"""CPPN evaluation: feed-forward the genome to produce developmental decisions.

The genome is a small graph. Inputs: (x, y, t, bias). Outputs:
  - divide_prob: sigmoid -> probability the cell divides this step
  - primitive_logits: softmax over primitive types (which to instantiate)
  - connect_strength: tanh -> how strongly to connect to neighbors

This module evaluates the CPPN deterministically (no torch — pure numpy)
because the genome is tiny and we want to support many evaluations per second.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple
from src.models.dneat.genome import Genome, GenomeNode, GenomeEdge


_ACTIVATIONS = {
    "none": lambda x: x,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-x)) if x > -50 else 0.0,
    "tanh": math.tanh,
    "relu": lambda x: max(0.0, x),
    "sin": math.sin,
    "gauss": lambda x: math.exp(-x * x),
    "step": lambda x: 1.0 if x > 0 else 0.0,
}


def _topological_eval(genome: Genome, input_values: Dict[int, float]) -> Dict[int, float]:
    """Evaluate the CPPN in topological order. Returns output node values."""
    import networkx as nx
    # Build graph
    g = nx.DiGraph()
    for nid in genome.nodes:
        g.add_node(nid)
    for e in genome.edges:
        if e.enabled:
            g.add_edge(e.src, e.dst)
    if not nx.is_directed_acyclic_graph(g):
        return {}
    order = list(nx.topological_sort(g))
    values: Dict[int, float] = {}
    for nid in order:
        node = genome.nodes[nid]
        if node.kind == "input":
            values[nid] = input_values.get(nid, 0.0)
            continue
        # Sum weighted inputs
        s = 0.0
        for e in genome.edges:
            if e.enabled and e.dst == nid:
                s += e.weight * values.get(e.src, 0.0)
        act = _ACTIVATIONS.get(node.activation, _ACTIVATIONS["tanh"])
        try:
            values[nid] = act(s)
        except OverflowError:
            values[nid] = 0.0
    return values


def evaluate_cppn(genome: Genome, x: float, y: float, t: float,
                  input_node_ids: List[int],
                  output_node_ids: List[int]) -> List[float]:
    """Evaluate the CPPN at a single (x, y, t) point.

    input_node_ids: ordered list of input node IDs (e.g., [x_id, y_id, t_id, bias_id])
    output_node_ids: ordered list of output node IDs

    Returns: list of output values, in the order of output_node_ids.
    """
    inputs = {nid: v for nid, v in zip(input_node_ids, [x, y, t, 1.0])}
    values = _topological_eval(genome, inputs)
    return [values.get(oid, 0.0) for oid in output_node_ids]
