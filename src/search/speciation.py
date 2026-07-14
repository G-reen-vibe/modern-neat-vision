"""Speciation for D-NEAT.

Classical NEAT uses a compatibility metric based on excess/disjoint genes
and weight differences. This works for small genomes but saturates at scale.

D-NEAT uses a *behavioral* speciation metric: two genomes are in the same
species if their developed phenotypes have similar graph structure
(measured by Weisfeiler-Lehman graph kernel) OR similar fitness profiles.

This is a SCAFFOLD in Phase 2. The actual speciation logic will be
implemented in Phase 4.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import random


@dataclass
class Species:
    species_id: int
    representative_genome_id: int
    member_ids: List[int] = field(default_factory=list)
    best_fitness: float = -float("inf")
    stagnation_count: int = 0


class Speciator:
    """Assigns genomes to species based on phenotypic similarity.

    SCAFFOLD: in Phase 2, we put all genomes in a single species.
    """
    def __init__(self, compatibility_threshold: float = 1.0):
        self.threshold = compatibility_threshold
        self.species: List[Species] = []

    def speciate(self, genomes: list) -> None:
        """Assign each genome to a species. Creates new species if needed."""
        # SCAFFOLD: single species
        if not self.species:
            self.species.append(Species(0, genomes[0].id if hasattr(genomes[0], "id") else 0))
        for g in genomes:
            self.species[0].member_ids.append(getattr(g, "id", id(g)))
