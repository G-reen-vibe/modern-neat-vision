"""D-NEAT evolution loop with real fitness evaluation.

Each individual's fitness = best validation accuracy after K epochs of
training on a subset of Fashion-MNIST. We use a small subset (5000 train,
1000 val) for speed — the goal is to compare phenotypes relatively, not
to reach SOTA.
"""
from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import torch
import torch.nn as nn

from src.models.dneat.genome import Genome, minimal_genome, random_genome
from src.models.dneat.developmental import develop, DevelopmentalConfig, stability_score
from src.models.dneat.phenotype import compile_phenotype
from src.utils.seed import seed_everything
from src.train.trainer import Trainer


@dataclass
class DNeatConfig:
    population_size: int = 10
    generations: int = 5
    inner_train_epochs: int = 3
    stability_weight: float = 0.0  # off by default for first run
    mutation_rate_add_node: float = 0.3
    mutation_rate_add_edge: float = 0.5
    mutation_rate_perturb: float = 0.8
    mutation_sigma: float = 0.3
    elite_fraction: float = 0.2
    train_subset_size: int = 5000
    val_subset_size: int = 1000
    batch_size: int = 128
    # Developmental config
    dev_config: DevelopmentalConfig = field(default_factory=DevelopmentalConfig)


@dataclass
class Individual:
    id: int
    genome: Genome
    fitness: float = -float("inf")
    stability: float = 0.0
    phenotype_node_count: int = 0
    train_time_s: float = 0.0
    val_acc: float = 0.0


def mutate_genome(genome: Genome, config: DNeatConfig, rng: random.Random) -> Genome:
    """Apply mutations to a genome. Returns a mutated copy."""
    import copy
    g = copy.deepcopy(genome)
    if rng.random() < config.mutation_rate_add_node:
        g.mutate_add_node()
    if rng.random() < config.mutation_rate_add_edge:
        g.mutate_add_edge()
    if rng.random() < config.mutation_rate_perturb:
        g.mutate_perturb_weights(sigma=config.mutation_sigma)
    return g


def evaluate_individual(ind: Individual, config: DNeatConfig,
                        train_loader, val_loader, num_classes: int,
                        in_channels: int, image_size: int) -> None:
    """Develop the genome, train the phenotype briefly, set ind.fitness."""
    t0 = time.time()
    try:
        phenotype = develop(ind.genome, config.dev_config, seed=0)
        if not phenotype.is_valid():
            ind.fitness = -0.1  # penalize invalid
            ind.val_acc = 0.0
            ind.phenotype_node_count = 0
            ind.train_time_s = time.time() - t0
            return
        model = compile_phenotype(phenotype, in_channels=in_channels,
                                  num_classes=num_classes, image_size=image_size)
        ind.phenotype_node_count = len(phenotype.nodes)
        # Train briefly
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
            warmup_epochs=1, total_epochs=config.inner_train_epochs,
            label_smoothing=0.1, grad_clip=1.0, device="cpu",
        )
        result = trainer.fit(config.inner_train_epochs, logger=None, eval_every=1)
        ind.val_acc = result["best_acc"]
        # Stability penalty
        if config.stability_weight > 0:
            ind.stability = stability_score(ind.genome, config.dev_config)
        ind.fitness = ind.val_acc - config.stability_weight * ind.stability
    except Exception as e:
        ind.fitness = -0.2  # penalize crashes
        ind.val_acc = 0.0
        ind.phenotype_node_count = 0
    ind.train_time_s = time.time() - t0


def run_dneat(config: DNeatConfig, train_loader, val_loader,
              num_classes: int, in_channels: int, image_size: int,
              seed: int = 0, verbose: bool = True) -> List[Individual]:
    """Run the D-NEAT evolution loop. Returns the final population."""
    rng = random.Random(seed)
    seed_everything(seed)

    # Initialize population
    population: List[Individual] = []
    for i in range(config.population_size):
        g = random_genome(seed=rng.randint(0, 10000))
        ind = Individual(id=i, genome=g)
        population.append(ind)

    # Track best
    best_ind = None
    best_fitness = -float("inf")

    for gen in range(config.generations):
        # Evaluate
        for ind in population:
            if ind.fitness <= -float("inf") / 2:  # unevaluated
                evaluate_individual(ind, config, train_loader, val_loader,
                                    num_classes, in_channels, image_size)
        # Sort by fitness
        population.sort(key=lambda i: i.fitness, reverse=True)
        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            best_ind = population[0]

        if verbose:
            accs = [i.val_acc for i in population]
            nodes = [i.phenotype_node_count for i in population]
            times = [i.train_time_s for i in population]
            print(f"  Gen {gen+1}/{config.generations}: "
                  f"best_acc={max(accs):.4f} mean_acc={sum(accs)/len(accs):.4f} "
                  f"best_nodes={max(nodes)} mean_nodes={sum(nodes)/len(nodes):.1f} "
                  f"gen_time={sum(times):.0f}s")

        # Elites
        n_elite = max(1, int(config.elite_fraction * config.population_size))
        elites = population[:n_elite]
        # Reproduce
        new_pop: List[Individual] = list(elites)  # carry elites
        next_id = max(i.id for i in population) + 1
        while len(new_pop) < config.population_size:
            parent = rng.choice(elites)
            child_genome = mutate_genome(parent.genome, config, rng)
            child = Individual(id=next_id, genome=child_genome)
            new_pop.append(child)
            next_id += 1
        population = new_pop

    # Final evaluation
    for ind in population:
        if ind.fitness <= -float("inf") / 2:
            evaluate_individual(ind, config, train_loader, val_loader,
                                num_classes, in_channels, image_size)
    population.sort(key=lambda i: i.fitness, reverse=True)
    if verbose:
        print(f"  Final: best_acc={population[0].val_acc:.4f} best_fitness={population[0].fitness:.4f}")
    return population
