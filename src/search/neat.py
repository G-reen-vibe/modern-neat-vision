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

from src.models.dneat.genome import Genome, GenomeNode, GenomeEdge, minimal_genome, random_genome
from src.models.dneat.developmental import develop, DevelopmentalConfig, stability_score
from src.models.dneat.phenotype import compile_phenotype
from src.utils.seed import seed_everything
from src.train.trainer import Trainer
from src.search.speciation import Speciator


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
    phenotype: object = None  # for speciation


def _tournament_select(population: List[Individual], k: int, rng: random.Random) -> Individual:
    """Tournament selection: pick k random individuals, return the fittest."""
    contestants = rng.sample(population, min(k, len(population)))
    return max(contestants, key=lambda i: i.fitness)


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


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random) -> Genome:
    """NEAT-style crossover. Parent_a is the fitter parent.

    - For matching genes (same innovation number): inherit from either parent randomly.
    - For excess/disjoint genes: inherit from the fitter parent (parent_a).
    """
    import copy
    # Build innovation -> edge maps
    edges_a = {e.edge_id: e for e in parent_a.edges}
    edges_b = {e.edge_id: e for e in parent_b.edges}
    all_innovations = set(edges_a.keys()) | set(edges_b.keys())

    child = Genome()
    # Copy nodes from parent_a (fitter), plus any nodes from parent_b that
    # have edges with innovations not in parent_a
    child.next_node_id = 0
    child.next_innovation = 0
    # Map old node IDs to new node IDs
    node_map = {}
    for nid, node in parent_a.nodes.items():
        new_nid = child.next_node_id
        child.next_node_id += 1
        node_map[nid] = new_nid
        child.nodes[new_nid] = GenomeNode(new_nid, node.kind, node.activation)
    # Add nodes from parent_b that aren't in parent_a
    for nid, node in parent_b.nodes.items():
        if nid not in node_map:
            new_nid = child.next_node_id
            child.next_node_id += 1
            node_map[nid] = new_nid
            child.nodes[new_nid] = GenomeNode(new_nid, node.kind, node.activation)

    # Inherit edges
    for innov in sorted(all_innovations):
        if innov in edges_a and innov in edges_b:
            # Matching gene: pick randomly
            e = edges_a[innov] if rng.random() < 0.5 else edges_b[innov]
        elif innov in edges_a:
            # Excess/disjoint from fitter parent: keep
            e = edges_a[innov]
        else:
            # Excess/disjoint from weaker parent: skip (NEAT standard)
            continue
        new_e = GenomeEdge(
            edge_id=child.next_innovation,
            src=node_map[e.src],
            dst=node_map[e.dst],
            weight=e.weight,
            enabled=e.enabled,
        )
        child.next_innovation += 1
        child.edges.append(new_e)

    # Set input/output IDs
    if hasattr(parent_a, "input_ids"):
        child.input_ids = [node_map[i] for i in parent_a.input_ids if i in node_map]
        child.output_ids = [node_map[i] for i in parent_a.output_ids if i in node_map]
    return child


def evaluate_individual(ind: Individual, config: DNeatConfig,
                        train_loader, val_loader, num_classes: int,
                        in_channels: int, image_size: int) -> None:
    """Develop the genome, train the phenotype briefly, set ind.fitness.

    Uses early-stopping: trains for 1 epoch, evaluates. If accuracy is below
    a minimum threshold (e.g., 0.10 = random for 10 classes), stops early to
    save compute on clearly bad candidates.
    """
    t0 = time.time()
    try:
        phenotype = develop(ind.genome, config.dev_config, seed=0)
        ind.phenotype = phenotype  # store for speciation
        if not phenotype.is_valid():
            ind.fitness = -0.1  # penalize invalid
            ind.val_acc = 0.0
            ind.phenotype_node_count = 0
            ind.train_time_s = time.time() - t0
            return
        model = compile_phenotype(phenotype, in_channels=in_channels,
                                  num_classes=num_classes, image_size=image_size)
        ind.phenotype_node_count = len(phenotype.nodes)
        # Early-stop: train 1 epoch first
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            num_classes=num_classes, lr=1e-3, weight_decay=5e-4,
            warmup_epochs=1, total_epochs=config.inner_train_epochs,
            label_smoothing=0.1, grad_clip=1.0, device="cpu",
        )
        if config.inner_train_epochs > 1:
            # Train first epoch
            train_stats = trainer.train_one_epoch()
            val_stats = trainer.evaluate()
            first_acc = val_stats["val_acc"]
            # Early-stop if clearly bad (below random + 0.05)
            if first_acc < 0.15:
                ind.val_acc = first_acc
                if config.stability_weight > 0:
                    ind.stability = stability_score(ind.genome, config.dev_config)
                ind.fitness = ind.val_acc - config.stability_weight * ind.stability
                ind.train_time_s = time.time() - t0
                return
            # Continue training remaining epochs
            remaining = config.inner_train_epochs - 1
            if remaining > 0:
                result = trainer.fit(remaining, logger=None, eval_every=1)
                ind.val_acc = max(first_acc, result["best_acc"])
            else:
                ind.val_acc = first_acc
        else:
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

    # Initialize population — hybrid strategy.
    # Half start from minimal genome (complexification, NEAT-style).
    # Half start from random genomes (diversity injection).
    # This balances NEAT's complexification principle with the need for
    # enough initial diversity to avoid premature convergence.
    population: List[Individual] = []
    n_minimal = config.population_size // 2
    for i in range(config.population_size):
        if i < n_minimal:
            g = minimal_genome()
            for _ in range(rng.randint(1, 3)):
                if rng.random() < 0.5:
                    g.mutate_add_node()
                else:
                    g.mutate_add_edge()
                g.mutate_perturb_weights(sigma=0.5)
        else:
            g = random_genome(seed=rng.randint(0, 10000))
        ind = Individual(id=i, genome=g)
        population.append(ind)

    # Track best
    best_ind = None
    best_fitness = -float("inf")
    speciator = Speciator(compatibility_threshold=0.3)

    for gen in range(config.generations):
        # Evaluate
        for ind in population:
            if ind.fitness <= -float("inf") / 2:  # unevaluated
                if verbose:
                    print(f"  [Gen {gen+1}] Evaluating individual {ind.id}...", end="", flush=True)
                evaluate_individual(ind, config, train_loader, val_loader,
                                    num_classes, in_channels, image_size)
                if verbose:
                    print(f" acc={ind.val_acc:.4f} ({ind.train_time_s:.0f}s)")
        # Speciate
        speciator.speciate(population)
        # Sort by fitness
        population.sort(key=lambda i: i.fitness, reverse=True)
        if population[0].fitness > best_fitness:
            best_fitness = population[0].fitness
            best_ind = population[0]

        if verbose:
            accs = [i.val_acc for i in population]
            nodes = [i.phenotype_node_count for i in population]
            times = [i.train_time_s for i in population]
            n_species = len(speciator.species)
            print(f"  Gen {gen+1}/{config.generations}: "
                  f"best_acc={max(accs):.4f} mean_acc={sum(accs)/len(accs):.4f} "
                  f"best_nodes={max(nodes)} mean_nodes={sum(nodes)/len(nodes):.1f} "
                  f"species={n_species} "
                  f"gen_time={sum(times):.0f}s")

        # Elites (carried over unchanged)
        n_elite = max(1, int(config.elite_fraction * config.population_size))
        elites = population[:n_elite]
        # Reproduce
        new_pop: List[Individual] = list(elites)  # carry elites
        next_id = max(i.id for i in population) + 1
        while len(new_pop) < config.population_size:
            if rng.random() < 0.3 and len(elites) >= 2:
                # Crossover two tournament winners
                parent_a = _tournament_select(population, 2, rng)
                parent_b = _tournament_select(population, 2, rng)
                child_genome = crossover(parent_a.genome, parent_b.genome, rng)
                child_genome = mutate_genome(child_genome, config, rng)
            else:
                # Asexual mutation from tournament winner
                parent = _tournament_select(population, 2, rng)
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
