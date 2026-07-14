"""D-NEAT evolution loop.

Outer loop of the algorithm:
  1. Initialize a population of minimal genomes.
  2. For each generation:
     a. Develop each genome into a phenotype.
     b. Compile each phenotype into a torch.nn.Module.
     c. Train each module for K epochs (inner loop).
     d. Assign fitness = best validation accuracy - lambda * stability_loss.
     e. Speciate the population.
     f. Within each species, select parents and apply mutations.
  3. Return the best genome found.

This is a SCAFFOLD in Phase 2. The actual evolution loop will be
implemented in Phase 4. For now, we just stub the main entry point.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import random

from src.models.dneat.genome import Genome, minimal_genome
from src.models.dneat.developmental import develop, stability_score
from src.search.speciation import Speciator


@dataclass
class DNeatConfig:
    population_size: int = 20
    generations: int = 10
    inner_train_epochs: int = 30
    stability_weight: float = 0.1
    mutation_rate_add_node: float = 0.3
    mutation_rate_add_edge: float = 0.5
    mutation_rate_perturb: float = 0.8
    elite_fraction: float = 0.2


@dataclass
class Individual:
    id: int
    genome: Genome
    fitness: float = -float("inf")
    stability: float = 0.0
    phenotype: Optional[object] = None


class DNeatSearch:
    """D-NEAT evolution loop. SCAFFOLD in Phase 2."""

    def __init__(self, config: DNeatConfig, train_fn: Callable, eval_fn: Callable):
        self.config = config
        self.train_fn = train_fn   # train_fn(phenotype) -> trained_module
        self.eval_fn = eval_fn     # eval_fn(module) -> fitness
        self.population: List[Individual] = []
        self.speciator = Speciator()
        self.next_individual_id = 0

    def init_population(self) -> None:
        for _ in range(self.config.population_size):
            g = minimal_genome()
            ind = Individual(id=self.next_individual_id, genome=g)
            self.next_individual_id += 1
            self.population.append(ind)

    def evaluate_population(self) -> None:
        for ind in self.population:
            ind.phenotype = develop(ind.genome)
            ind.stability = stability_score(ind.genome)
            # SCAFFOLD: actual train + eval deferred to Phase 4
            ind.fitness = 0.0 - self.config.stability_weight * ind.stability

    def evolve_one_generation(self) -> None:
        self.evaluate_population()
        self.speciator.speciate(self.population)
        # SCAFFOLD: selection + reproduction deferred to Phase 4

    def search(self) -> Individual:
        self.init_population()
        for gen in range(self.config.generations):
            self.evolve_one_generation()
        best = max(self.population, key=lambda i: i.fitness)
        return best
