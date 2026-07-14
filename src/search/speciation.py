"""Speciation for D-NEAT based on phenotypic similarity.

Two individuals are in the same species if their developed phenotypes
have similar primitive-type histograms. This is a cheap, fitness-agnostic
metric that groups topologically similar individuals.

Within each species, competition is local — only the fittest in each
species reproduces. This protects novel topologies from being immediately
outcompeted by the current best.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any
import random


@dataclass
class Species:
    species_id: int
    representative_id: int
    member_ids: List[int] = field(default_factory=list)
    best_fitness: float = -float("inf")
    stagnation_count: int = 0


def _phenotype_signature(ind: Any) -> Dict[str, int]:
    """Extract a primitive-type histogram from the individual's phenotype."""
    if not hasattr(ind, 'phenotype') or ind.phenotype is None:
        return {}
    sig: Dict[str, int] = {}
    for node in ind.phenotype.nodes.values():
        sig[node.primitive_name] = sig.get(node.primitive_name, 0) + 1
    return sig


def _signature_distance(s1: Dict[str, int], s2: Dict[str, int]) -> float:
    """L1 distance between two primitive histograms, normalized."""
    all_keys = set(s1.keys()) | set(s2.keys())
    if not all_keys:
        return 0.0
    total = sum(abs(s1.get(k, 0) - s2.get(k, 0)) for k in all_keys)
    return total / max(1, sum(s1.values()) + sum(s2.values()))


class Speciator:
    """Assigns individuals to species based on phenotypic similarity."""

    def __init__(self, compatibility_threshold: float = 0.3):
        self.threshold = compatibility_threshold
        self.species: List[Species] = []

    def speciate(self, population: List[Any]) -> None:
        """Assign each individual to a species. Creates new species as needed."""
        for sp in self.species:
            sp.member_ids = []

        for ind in population:
            ind_sig = _phenotype_signature(ind)
            assigned = False
            for sp in self.species:
                rep = next((i for i in population if i.id == sp.representative_id), None)
                if rep is None:
                    continue
                rep_sig = _phenotype_signature(rep)
                if _signature_distance(ind_sig, rep_sig) < self.threshold:
                    sp.member_ids.append(ind.id)
                    assigned = True
                    break
            if not assigned:
                sp = Species(
                    species_id=len(self.species),
                    representative_id=ind.id,
                    member_ids=[ind.id],
                )
                self.species.append(sp)

        for sp in self.species:
            members = [i for i in population if i.id in sp.member_ids]
            if not members:
                continue
            current_best = max(m.fitness for m in members)
            if current_best > sp.best_fitness:
                sp.best_fitness = current_best
                sp.stagnation_count = 0
            else:
                sp.stagnation_count += 1

        self.species = [sp for sp in self.species if sp.member_ids]
