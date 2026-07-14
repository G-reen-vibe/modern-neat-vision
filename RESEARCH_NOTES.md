# Modernizing NEAT for Arbitrary-Topology Image Classifiers
## A Research Agent's Working Notes

**Repo:** `G-reen-vibe/modern-neat-vision`
**Date:** 2026-07-15
**Status:** Phase 1 — problem framing, taxonomy of directions, critical review. No methodology has been chosen yet; this document is the *deliberation*, not the conclusion.

---

## 0. Reading the brief carefully

The brief is unusually precise about what it wants and what it does *not* want.

What it wants:
- Modernize NEAT. Not replace it with something unrelated — the spirit of NEAT (topology discovery through evolution of an encoding) is to be preserved.
- Discover **arbitrary graph topologies**. Not chain-like CNNs, not the standard cell-stack-cell-stack pattern of NAS. Arbitrary DAGs (or beyond).
- For **image classifiers** with **performance comparable to SOTA**. This is the hard constraint. SOTA on CIFAR-10/ImageNet today is ViTs and ConvNeXts in the 95%+ / 80%+ range. Anything we propose must have a credible path to that ballpark.
- **Not constrained to standard CNNs or even neural networks.** This is the most important sentence in the brief. We are explicitly invited to think about non-neural substrates or non-standard neural ones.
- **May use gradients** to accelerate training. Mixed evolution + SGD is on the table.
- **Prefer an elegant, well put together strategy over a collage of a bunch of things.** This is the second-most important sentence. The user is preemptively warning against the failure mode of NEAT-successor papers, which tend to bolt on five ideas and call it a method.

What it does *not* want:
- A singular methodology picked prematurely. We must explore the space first.
- A literature survey. The brief is about thinking, not citing.
- A collage.

So this document is structured as: (1) re-read the history critically, (2) reframe the problem from first principles, (3) enumerate directions with honest strengths/weaknesses, (4) extract cross-cutting themes, (5) propose three concrete syntheses that are each *minimal* rather than *maximal*, (6) list the experiments that would actually disambiguate between them.

---

## 1. A critical re-reading of NEAT's history

### 1.1 What NEAT actually got right (and why it is still worth modernizing)

NEAT (Stanley & Miikkulainen, 2002) introduced four ideas that, in combination, were genuinely novel and remain underused in modern architecture search:

1. **Complexification from a minimal start.** The search begins with the simplest possible network (inputs → outputs, fully connected) and only adds structure as needed. This biases the search toward parsimony in a way that NAS-style search spaces emphatically do not.
2. **Historical markings via innovation numbers.** Every structural mutation gets a globally unique ID. Two networks with the same innovation number on a gene are "the same gene," which makes crossover well-defined even on different topologies. This is a *representational* trick — it makes the space of topologies into something gene-alignable, like biological homology.
3. **Speciation to protect innovation.** New topology mutations usually start worse than their parents (because their weights are untrained). Speciation lets them compete within a niche before being crushed by the dominant topology. This explicitly models the explore/exploit tension that gradient-based architecture search handles poorly.
4. **Topological encoding, not weight encoding.** The genome encodes *which connections exist* and *rough weights*, but the topology itself is the locus of evolution. Weights are secondary. This is a profound reversal of how most modern NAS thinks.

These four ideas are not obsolete. They are, in fact, *underexploited* — most modern NAS (DARTS, ENAS, one-shot supernet methods) goes the other direction: it defines a fixed maximum supergraph and learns continuous masks over it. That approach gives up on (1) and (2) entirely and only weakly preserves (3) and (4). NEAT's stance that *structure itself is the search object* is a genuinely different and arguably under-explored angle.

### 1.2 Why did progress stall after HyperNEAT and CoDeepNEAT?

The honest answer is that NEAT-style methods lost the empirical race to gradient-based NAS for vision, and the field moved on. But *why* did they lose? Pinpointing this is essential, because a modernization that doesn't address the actual failure modes will repeat them.

**Failure mode 1: CPPN is the wrong inductive bias for vision, but the *only* one NEAT had.** HyperNEAT's CPPN (Compositional Pattern Producing Network) encodes a network's weights as a function of the *geometric coordinates* of source and target nodes. This is beautiful for regular grids (it discovers convolution-like patterns naturally) but it forces the phenotype into a Cartesian substrate. Once you commit to a substrate, you have already given up on "arbitrary graph topologies." Modern vision architectures aren't regular grids — they have skip connections, attention heads, hierarchical branches, asymmetric bottlenecks. CPPN can't naturally express these.

**Failure mode 2: Fitness evaluation was — and is — brutally expensive.** Each candidate network must be trained to convergence to get a meaningful accuracy signal. NEAT populations of thousands are infeasible when each evaluation is a full training run. This is the single biggest practical barrier, and it has gotten *worse* over time as SOTA networks have grown.

**Failure mode 3: Speciation was never principled.** NEAT's compatibility metric (a linear combination of excess/disjoint genes and weight differences) is a heuristic. As topologies get large, the metric saturates and speciation degenerates. CoDeepNEAT sidestepped this by evolving modules rather than whole networks, but at the cost of giving up "arbitrary topology" — modules get assembled in predefined ways.

**Failure mode 4: No gradient signal during topology search.** Pure NEAT treats topology as a black box that produces a fitness. But topology decisions are highly correlated: adding a skip connection between layers 4 and 7 is similar to adding one between 5 and 8, and a smooth gradient over "how much skip" would dramatically focus the search. NEAT's binary add/remove mutations cannot exploit this.

**Failure mode 5: SOTA moved to a regime where pure topology search is insufficient.** Modern SOTA classifiers are not just well-connected graphs — they use batch norm, attention, positional encodings, gated units, mixture-of-experts routing, etc. NEAT's mutation vocabulary (add node, add edge, perturb weight) cannot naturally express any of these. To compete with SOTA, the search space must include these primitives, which means NEAT must be generalized to evolve *programs over primitives*, not just graphs over linear units.

**Failure mode 6: The community never found a clean way to combine NEAT with transfer learning / pretraining.** SOTA vision performance today is dominated by pretrained backbones. A from-scratch topology search on ImageNet is not just expensive — it is competing against backbones that have already absorbed billions of images. Any modernization has to either (a) accept that it is searching for *fine-tuning topologies on top of frozen features*, or (b) explicitly search for *pretrainable topologies*, which is a much harder objective.

These six failure modes are the design targets. Any proposed modernization must address each one explicitly, or explain why it doesn't need to.

---

## 2. First-principles reframing

Before listing directions, let me re-state the problem in a way that doesn't already bias the solution.

### 2.1 What is "topology discovery" actually doing?

A neural network is a program. Its topology is the dataflow graph of that program. "Topology discovery" is *program synthesis* — specifically, the synthesis of differentiable programs over a small vocabulary of primitives (linear maps, nonlinearities, normalizations, attention, pooling, etc.).

Under this view:
- NEAT = program synthesis with a tiny primitive set {linear, tanh} and an evolutionary search operator.
- DARTS = continuous relaxation over a fixed program skeleton, optimized by gradient.
- HyperNEAT = a meta-program (CPPN) that emits a program (the network).
- CoDeepNEAT = hierarchical program synthesis: synthesize subroutines (modules) and a call graph (assembly).

This reframing is useful because it lets us import ideas from the program synthesis literature — type systems, combinator libraries, higher-order functions, program transformations — that the NEAT literature has historically ignored.

### 2.2 The four sub-problems

Any topology-discovery system must solve four coupled sub-problems. Decomposing them this way makes it easier to compare directions, because each direction tends to focus on a different subset.

1. **The encoding problem.** How is an arbitrary topology represented in a form that (a) supports meaningful mutation and crossover, (b) is closed under the search operators (you don't generate invalid individuals), and (c) is expressive enough to cover the target space?
2. **The evaluation problem.** How do we assign a fitness signal to a candidate, given that a single training run is expensive?
3. **The search problem.** Given an encoding and an evaluator, how do we move through the space? Evolution only? Gradient only? Hybrid? Quality-diversity?
4. **The primitive problem.** What is the alphabet of operations the topology can be built from? Just linear+tanh? Or attention, normalization, gating, skip connections, even non-neural operations?

Most NEAT-successor work has focused on (3) and treated (1), (2), (4) as fixed. I will argue below that the biggest wins are likely to come from (1) and (4), not (3).

### 2.3 The hidden zeroth problem: what is the unit of evolution?

NEAT evolves whole networks. CoDeepNEAT evolves modules and assemblies separately. But there are other choices: evolve *families* of networks that share a latent code (à la hypernetworks), evolve *developmental programs* that build networks (à la cellular automata), evolve *learning rules* that produce networks through training (à la evolved OpenAI-ES). The choice of "unit" determines everything downstream, and the NEAT literature has underexplored the alternatives.

---

## 3. Taxonomy of research directions

I will organize the directions into four families, by which sub-problem they primarily attack. For each direction I give: the core idea, the mechanism, three strengths, three weaknesses, and a "killer objection" — the single criticism that, if not answered, sinks the direction.

### Family A — Encoding-centric directions
*These attack sub-problem (1): how to represent topologies.*

#### A1. Continuous superposition (DARTS-style differentiable NEAT)

**Core idea.** Define a supergraph over all allowed primitives. Each edge has a continuous mixing weight. Gradient descent on the mixing weights, then discretize. Add NEAT-style complexification by growing the supergraph over time.

**Strengths.** (i) Gradient signal is fully utilized. (ii) Trivially GPU-parallel. (iii) Has empirical track record (DARTS, ProxylessNAS) at SOTA-ish accuracy.
**Weaknesses.** (i) The supergraph is *not* an arbitrary topology — it's a fixed DSL. You don't discover graphs, you discover soft masks over a graph. (ii) Discretization gap is well-known and severe. (iii) The complexification trick mostly doesn't help because the supergraph already commits to a maximum size.
**Killer objection.** *This is not NEAT anymore.* The brief asked to modernize NEAT, not replace it with DARTS. If we go this route, we should be honest about that.

#### A2. Grammar-based neuroevolution

**Core idea.** Define a graph grammar (production rules that rewrite non-terminal subgraphs into terminal subgraphs). Mutations are grammar-rule applications. A "genome" is a sequence of rule firings.

**Strengths.** (i) Closed under search operators by construction — every individual is valid. (ii) Naturally hierarchical, which captures modern architectures (cells → stages → network). (iii) Interpretable: a discovered architecture is a parse tree, which can be inspected and re-used.
**Weaknesses.** (i) The grammar *is* the inductive bias, and designing it is itself a research problem. (ii) Grammar expressivity vs searchability tension: too permissive and search explodes, too restrictive and you've baked in the answer. (iii) Crossover between parse trees of different shapes is non-trivial.
**Killer objection.** *Who designs the grammar?* If we design it, we have smuggled in our prior about what good architectures look like, and "discovery" becomes a fig leaf. If we *learn* the grammar, that's a separate hard problem.

#### A3. Latent-space graph VAE + evolution

**Core idea.** Train a graph variational autoencoder on a corpus of architectures (random + known-good). Encode each as a point in ℝᵈ. Run evolution (CMA-ES, MAPElites) in latent space. Decode to discrete topology, train, measure fitness, retrain VAE periodically.

**Strengths.** (i) Evolution in continuous space is well-understood and fast. (ii) Smoothness of latent space gives meaningful gradient-like signal even with evolutionary search. (iii) Can leverage existing architecture corpora (NAS-Bench-201, NATS-Bench, Network Design Spaces).
**Weaknesses.** (i) The VAE's expressive ceiling is set by its training corpus — novelty outside the corpus is bounded. (ii) Decoded graphs may be invalid or poorly-conditioned; need a repair operator. (iii) Online retraining of the VAE is expensive and may destabilize the search.
**Killer objection.** *The VAE cannot invent a primitive it has never seen.* If we trained it on conv-nets, it will not produce attention. This caps the "discovery" at recombination of known motifs.

#### A4. Developmental / morphogenetic encoding

**Core idea.** The genome is a small program (e.g., a neural cellular automaton, or a CPPN-like rule) that runs as a *developmental process* over discrete time steps. Cells divide, differentiate, form connections based on local rules and morphogen gradients. The resulting phenotype is the classifier. Evolution operates on the genome; gradient descent trains the phenotype.

**Strengths.** (i) Single genome can produce arbitrarily large phenotypes — addresses NEAT's scaling issue. (ii) Naturally produces modular, repeating structures (a known property of biological morphogenesis). (iii) Genome is small and evolvable in a low-dimensional space.
**Weaknesses.** (i) Genome → phenotype mapping is highly nonlinear; small mutations can produce catastrophic phenotypic changes (the "developmental instability" problem). (ii) Search is hard to debug — you can't easily tell why a candidate failed. (iii) Encoding modern primitives (attention, normalization) as developmental outcomes is non-trivial.
**Killer objection.** *Developmental instability makes fitness landscapes extremely rugged.* Without solving that, evolution will not progress. Solutions exist (resilience to noise in the genome, gene duplication) but they have not been shown to work at vision scale.

#### A5. Self-referential genome

**Core idea.** The genome is itself a small neural network G. G takes as input a description of the current phenotype and outputs a structural modification (add edge, add node, change primitive type). The phenotype is built by running G repeatedly. G is evolved.

**Strengths.** (i) Turing-complete encoding — can express any computable topology transformation. (ii) The genome can be made differentiable and refined by gradient. (iii) Self-modification is a powerful abstraction that connects to program synthesis.
**Weaknesses.** (i) Search space is enormous. (ii) No clear speciation metric. (iii) Training G to be useful is itself a hard credit-assignment problem (which modification caused the final accuracy?).
**Killer objection.** *Who trains the genome?* If by evolution, we are back to NEAT's sample-inefficiency. If by gradient, we need a differentiable simulator of phenotype construction, which is itself an open problem.

### Family B — Evaluation-centric directions
*These attack sub-problem (2): how to assign fitness cheaply.*

#### B1. Surrogate-assisted evolution

**Core idea.** Train a surrogate model f̂: architecture → predicted fitness, using features of the graph (depth, width, spectral properties, operation histogram). Use f̂ to filter candidates before training. Train only the top-k. Update f̂ online.

**Strengths.** (i) Decouples search cost from training cost. (ii) Can be combined with any search algorithm. (iii) Has worked well in NAS-Bench-201 experiments.
**Weaknesses.** (i) Surrogate generalization to novel regions is poor — exactly the regions evolution should explore. (ii) Feature engineering for graphs is non-trivial; graph neural network surrogates have their own training cost. (iii) Risk of "surrogate collapse" where evolution optimizes the surrogate instead of true fitness.
**Killer objection.** *The most novel architectures are precisely those the surrogate has never seen and will misrank.*

#### B2. Weight-sharing / one-shot evaluation

**Core idea.** Train a single "super-net" containing all candidate operations. Each candidate inherits its weights from the super-net. Only fine-tune briefly. This is the ENAS / one-shot NAS approach.

**Strengths.** (i) Empirically the most efficient known evaluation. (ii) Each candidate's evaluation is seconds, not hours.
**Weaknesses.** (i) Requires a fixed supergraph — re-introduces the DARTS problem (not arbitrary topology). (ii) Weight-sharing bias is well-documented: candidates that share weights with high-performing subnetworks get inflated fitness. (iii) Extending to *arbitrary* topologies (rather than cell-based) breaks the weight-sharing contract.
**Killer objection.** *One-shot NAS only works when the search space is a strict subset of a fixed supergraph.* Arbitrary topology discovery is incompatible with this constraint by definition.

#### B3. Early-stop / learning-curve extrapolation

**Core idea.** Train each candidate for a few epochs only. Extrapolate the learning curve to predict final accuracy using a parametric model (power law, exponential approach). Use predicted final accuracy as fitness.

**Strengths.** (i) Compatible with arbitrary topologies. (ii) Decoupled from any supergraph. (iii) Empirically reduces evaluation cost 5-10×.
**Weaknesses.** (i) Extrapolation error compounds with novelty. (ii) Different topologies have different "warm-up" dynamics, so early curves are unreliable. (iii) Hard to calibrate the extrapolator without a labeled corpus.
**Killer objection.** *The shape of the learning curve is itself architecture-dependent, so a single extrapolator will systematically misjudge novel families.*

#### B4. Co-evolved fast task / curriculum

**Core idea.** Co-evolve (a) the topology and (b) a "curriculum" of progressively harder subtasks. A topology only "graduates" to a harder task if it passes the easier one. Wasted training is minimized.

**Strengths.** (i) Cheap rejection of bad candidates early. (ii) Curriculum is itself a useful artifact.
**Weaknesses.** (i) Designing the task family is a research problem of its own. (ii) Risk of optimizing for easy-task performance at the expense of full-task performance.
**Killer objection.** *The curriculum is a hidden inductive bias.* Whoever designs the task family controls what architectures are discoverable.

### Family C — Search-centric directions
*These attack sub-problem (3): how to move through the space.*

#### C1. Quality-diversity (MAP-Elites + gradient refinement)

**Core idea.** Maintain an archive of elites indexed by behavioral descriptors (e.g., depth, width, parameter count, FLOPs). Each generation, mutate elites, train, and place in archive if they outperform the incumbent in their cell. Periodically apply a few steps of gradient-based architecture refinement.

**Strengths.** (i) Produces diverse high-performing topologies — useful for downstream analysis. (ii) Avoids population collapse. (iii) Naturally multi-objective (accuracy + FLOPs + params).
**Weaknesses.** (i) Descriptor design is critical and under-explored. (ii) Archive size scales with descriptor resolution; can be memory-heavy. (iii) Gradient refinement on discrete topology is non-trivial — needs a relaxation.
**Killer objection.** *MAP-Elites explores the descriptor space, not the topology space.* If descriptors don't capture what makes architectures different, you get redundant elites.

#### C2. Novelty search with objective bias

**Core idea.** Pure novelty search (Lehman & Stanley) ignores fitness and rewards only behavioral novelty. Add a small objective bias to keep the population productive.

**Strengths.** (i) Escapes local optima aggressively. (ii) Historically the source of surprising discoveries.
**Weaknesses.** (i) Without objective, search wanders. With too much, novelty is irrelevant. Calibrating the bias is hard. (ii) "Behavioral novelty" requires a behavior characterization, which is itself a design problem.
**Killer objection.** *At SOTA scale, we cannot afford to wander.* Novelty search's empirical wins are on cheap fitness functions; vision is not cheap.

#### C3. Bayesian optimization with graph kernels

**Core idea.** Use Bayesian Optimization with a Weisfeiler-Lehman graph kernel over topologies. Acquisition function (expected improvement) selects next candidate.

**Strengths.** (i) Principled sample efficiency. (ii) Uncertainty estimates guide exploration. (iii) Well-studied theory.
**Weaknesses.** (i) Graph kernels scale poorly (cubic in dataset size for many variants). (ii) Kernels capture graph isomorphism, not functional similarity — two topologically different graphs can be functionally identical. (iii) BO works in low dimensions; topology space is high-dimensional.
**Killer objection.** *BO with graph kernels has only been shown to work on small search spaces (≤1000 candidates).* Vision-scale topology spaces are vastly larger.

#### C4. CMA-ES in latent space

**Core idea.** Combine with A3: encode architectures as points in ℝᵈ, run CMA-ES there.

**Strengths.** (i) CMA-ES is best-in-class for continuous optimization in 10-100 dims. (ii) Naturally adapts its search distribution.
**Weaknesses.** (i) Only as good as the latent space. (ii) CMA-ES population sizes (10-50) may be too small for high-dimensional latent spaces.
**Killer objection.** *All the difficulty is in the encoder, not the optimizer.* CMA-ES in a bad latent space is no better than random search.

#### C5. Adversarial topology discovery

**Core idea.** Generator network produces topologies; discriminator distinguishes high-fitness from low-fitness topologies. Train adversarially. Optionally, discriminator is a learned fitness predictor.

**Strengths.** (i) Generator can amortize the search — once trained, sampling is cheap. (ii) Connects to GAN-style implicit modeling.
**Weaknesses.** (i) Mode collapse → low diversity. (ii) Adversarial training is unstable. (iii) Discriminator needs labeled (architecture, fitness) pairs, which are expensive.
**Killer objection.** *Adversarial setups don't buy us anything a VAE + CMA-ES doesn't already give us, and they're harder to train.*

### Family D — Primitive-centric directions
*These attack sub-problem (4): what the topology is built from.*

#### D1. Standardized module library (attention, conv, norm, etc.)

**Core idea.** Define a typed library of primitives: Conv(k, s), Attention(h), BatchNorm, LayerNorm, ReLU, GELU, Pool, Skip, etc. Mutations include "swap primitive at node v," "insert primitive of type T between u and v," "change hyperparameter of primitive at v."

**Strengths.** (i) Direct compatibility with SOTA primitives. (ii) Topologies are immediately trainable in PyTorch. (iii) Searches in the same vocabulary as human designers.
**Weaknesses.** (i) The library is a strong inductive bias — it bounds what can be discovered. (ii) Type-checking (e.g., attention needs Q, K, V) makes mutation complex. (iii) Library can grow large, expanding the search space.
**Killer objection.** *If we restrict to known primitives, what exactly are we "discovering"?* Just new wiring patterns among old parts. That may be enough — but we should be honest.

#### D2. Differentiable AST / program synthesis

**Core idea.** Treat the architecture as an abstract syntax tree over a small language of differentiable operations. Evolve ASTs (program synthesis style). Allows novel compound operations (e.g., "Conv then GroupNorm then GELU then 1×1 Conv" as a single evolved macro).

**Strengths.** (i) Maximally expressive — can encode any differentiable program. (ii) Emergent macros can be re-used as new primitives. (iii) Connects to program synthesis literature (DreamCoder, etc.).
**Weaknesses.** (i) AST crossover and mutation are tricky. (ii) Type system needed for validity. (iii) Search space is enormous.
**Killer objection.** *Program synthesis at this scale has not been shown to work without a strong prior.* Building that prior is itself a research program.

#### D3. Non-neural substrates: hyperdimensional computing

**Core idea.** Replace neurons with high-dimensional binary vectors (HDC). Bind and superpose operations form the topology. Evolve the binding pattern.

**Strengths.** (i) Radically different computational substrate — true "outside the box." (ii) HDC is naturally robust and energy-efficient. (iii) Topology is more naturally graph-like (binding = edge, superposition = node).
**Weaknesses.** (i) HDC has never matched SOTA on ImageNet. (ii) Training HDCs is non-trivial; gradient methods are imperfect. (iii) Hardware support is poor.
**Killer objection.** *No path to SOTA.* Worth researching for its own sake, but not for this brief.

#### D4. Non-neural substrates: spiking / neuromorphic

**Core idea.** Use spiking neurons (LIF, Izhikevich) as primitives. Evolve connectivity + neuron parameters. Train with surrogate gradients.

**Strengths.** (i) Energy-efficient at inference. (ii) Temporal dynamics add expressivity. (iii) Active research area.
**Weaknesses.** (i) SNNs trail ANNs by 5-10 points on ImageNet. (ii) Training is unstable. (iii) SOTA SNNs use CNN-derived topologies, not discovered ones.
**Killer objection.** *Same as D3 — no clear path to SOTA performance parity.*

#### D5. Tensor network decompositions

**Core idea.** Represent the network as a tensor network (MPS, PEPS, tree tensor network). Topology = tensor contraction graph. Evolve the structure of the tensor network.

**Strengths.** (i) Mathematically elegant — contraction graph *is* the topology. (ii) Natural way to express factorized linear maps. (iii) Connects to quantum many-body physics literature.
**Weaknesses.** (i) Tensor networks have shown limited success on vision tasks. (ii) Contraction order optimization is NP-hard. (iii) Unclear how to incorporate nonlinearities.
**Killer objection.** *Beautiful but not battle-tested for vision.* High risk of producing elegant mathematics that doesn't translate to accuracy.

#### D6. Mixed continuous-discrete fields

**Core idea.** Define the network as a *continuous field* over a learned manifold. Each point on the manifold has a "compute density" and "operation type." Discretize the field at the end. Evolution shapes the field; gradient descent shapes the weights.

**Strengths.** (i) Truly continuous search space — gradients available everywhere. (ii) Natural way to express arbitrary topology. (iii) Connects to neural radiance fields / implicit neural representations.
**Weaknesses.** (i) Discretization is non-trivial. (ii) Field → network mapping is underspecified. (iii) No empirical evidence this works at vision scale.
**Killer objection.** *Seductive but unproven.* The discretization step is where the elegance will likely break.

### Family E — Hybrid / unifying directions
*These explicitly try to be "elegant" by combining a small number of ideas.*

#### E1. "Latent Architecture Optimization" (LAO)

**Core idea.** A graph VAE encodes architectures. CMA-ES searches in latent space. Each evaluation: decode → train briefly → early-stop fitness. Surrogate model on latent features filters cheaply. Periodically retrain VAE with discovered architectures.

This is essentially A3 + B1 + B3 + C4 combined. It is *almost* a collage, but the components share a common substrate (the latent space), which gives it some elegance.

**Strengths.** Each piece is well-understood empirically.
**Weaknesses.** Many moving parts; tuning is non-trivial. Killer objection: the VAE cannot invent new primitives (inherits A3's weakness).

#### E2. "Morphogenetic NEAT" (M-NEAT)

**Core idea.** A small CPPN-like genome encodes a *developmental program*. The program runs as a graph grammar that grows the phenotype. The phenotype is a DAG over a typed primitive library (D1). Weights are trained by gradient. The genome is evolved by NEAT (with speciation preserved). Crucially, the developmental program is *regularized for stability* via a denoising objective — the phenotype must be robust to noise in the developmental process.

This combines A4 + D1 + B3 + classical NEAT speciation, with a stability regularizer as the unifying principle.

**Strengths.** Genuinely modernizes NEAT (preserves speciation, complexification). Scales (genome is small). Compatible with SOTA primitives (via D1).
**Weaknesses.** Developmental stability is the make-or-break. Implementation complexity.
**Killer objection.** *The stability regularizer might be so strong that phenotypes degenerate to simple repeating patterns — i.e., back to CPPN's regularity bias.*

#### E3. "Differentiable Topology Field" (DTF)

**Core idea.** Define a continuous topology field over a learned latent manifold. The field outputs, for each point, a probability distribution over primitives and connection strengths. Sample a discrete topology from the field, train it, and backpropagate the fitness signal through the sampling (via Gumbel-softmax or REINFORCE) to update the field. Evolution shapes the field's hyperparameters; gradient descent shapes the field's parameters.

This is D6 + differentiable sampling + light evolutionary outer loop. The unifying idea is "everything is a field; we sample discrete instances."

**Strengths.** Maximally differentiable. Truly arbitrary topology (within the primitive library).
**Weaknesses.** High variance gradients from discrete sampling. Manifold design is a research problem.
**Killer objection.** *Gumbel-softmax over large primitive vocabularies is known to have low-temperature pathologies.* May not actually work in practice.

#### E4. "Open-Ended Architecture Discovery" (OEAD)

**Core idea.** Take the POET / AMOGO open-ended learning paradigm and apply it to architecture search. Maintain an environment of *tasks* (subsets of the dataset, transformed versions, auxiliary objectives) and a population of architectures. Architectures and tasks co-evolve. Quality-diversity archive maintains behavioral diversity. The system is open-ended — there is no fixed goal, only continued discovery.

This is C1 + B4 + open-endedness.

**Strengths.** Genuinely creative; might discover surprising topologies. Avoids the "local optimum of single-task accuracy" failure mode.
**Weaknesses.** Very expensive. Hard to evaluate ("did we discover something?"). SOTA-accuracy comparison is awkward — open-ended systems aren't optimized for any one task.
**Killer objection.** *Doesn't directly serve the brief, which asks for SOTA-comparable performance on image classifiers.*

---

## 4. Cross-cutting themes

Looking across the directions, five themes recur. These are the load-bearing ideas; any final synthesis will likely use most of them.

### 4.1 Decouple genome from phenotype
Almost every promising direction separates a *small evolved object* (genome, latent code, field parameters) from a *large trained object* (the actual network). This is essential for vision scale. Pure NEAT (which evolves the phenotype directly) does not scale; the genome-phenotype split is non-negotiable.

### 4.2 Use gradient descent as the inner loop
Even evolutionary die-hards should accept this. Topology search outer loop, weight optimization inner loop. The question is not *whether* to use gradients but *where* — only on weights, or also on a relaxed architecture?

### 4.3 Latent-space search is dominant
Almost every direction ultimately lands on "search in a continuous latent space, decode to discrete." This is because (a) continuous spaces admit gradient-like signals even with evolutionary search, (b) CMA-ES / BO are mature on continuous spaces, (c) the encoder can amortize structure. The disagreements are about *what the latent space represents* and *how the decoder works*.

### 4.4 The primitive library is the silent design choice
Whether we acknowledge it or not, every direction commits to a primitive library. The question is whether we are honest about it. If we restrict to {Conv, Attention, Norm, Activations}, we are doing "wiring discovery." If we allow program synthesis, we are doing "primitive discovery." The brief's invitation to think beyond standard CNNs leans toward the latter, but the SOTA constraint pushes back toward the former.

### 4.5 Stability of the encoding is the hidden bottleneck
A4, A5, E2 all suffer from the same issue: small changes to the genome cause large changes to the phenotype, which makes fitness landscapes rugged and evolution slow. Any direction that uses a generative genome must solve this. The two known solutions are (a) explicit stability regularizers, (b) smooth latent spaces (A3, E1) at the cost of bounded novelty.

---

## 5. Comparative analysis

| Direction | SOTA path? | Novelty ceiling | Sample efficiency | Elegance | Implementation risk |
|---|---|---|---|---|---|
| A1 DARTS-like | ✓ | Low (within supergraph) | High | Low (not NEAT) | Low |
| A2 Grammar | ✓ | Medium (within grammar) | Medium | High | Medium |
| A3 Latent VAE | ✓ | Low (within corpus) | High | Medium | Medium |
| A4 Developmental | ? | High | Low | High | High |
| A5 Self-referential | ? | Very high | Low | Medium | Very high |
| B1 Surrogate | (tool) | — | (tool) | — | Low |
| B2 One-shot | ✓ | Low | Very high | Low | Low |
| B3 Early-stop | (tool) | — | (tool) | — | Low |
| B4 Curriculum | (tool) | — | (tool) | — | Medium |
| C1 MAP-Elites | ✓ | Medium | Medium | Medium | Medium |
| C2 Novelty | ? | High | Low | Medium | Medium |
| C3 BO/graph kernel | ? | Low | High | High | Medium |
| C4 CMA-ES latent | ✓ | Depends on encoder | High | Medium | Low |
| C5 Adversarial | ? | Medium | Medium | Low | High |
| D1 Primitive library | ✓ | Low (wiring only) | — | High | Low |
| D2 Program synthesis | ? | Very high | Low | High | Very high |
| D3 HDC | ✗ | High | Low | Medium | High |
| D4 SNN | ✗ | High | Low | Medium | High |
| D5 Tensor network | ? | Medium | Low | Very high | High |
| D6 Continuous field | ? | High | Medium | High | High |
| E1 LAO | ✓ | Low | High | Low (collage) | Medium |
| E2 M-NEAT | ? | High | Low | High | High |
| E3 DTF | ? | High | Medium | High | High |
| E4 OEAD | ✗ (not goal-aligned) | Very high | Low | Medium | Very high |

**Reading the table.** Directions marked `?` for SOTA path are not known to be impossible — they are unproven. The brief asks for "performance comparable to SOTA," which is a high bar; directions marked `✓` are the only ones with empirical track record at that bar.

**The honest tension.** The brief asks for two things in tension: (a) novel, possibly non-neural designs, and (b) SOTA-comparable performance. Every direction that maximizes (a) struggles with (b). The most likely resolutions are:

1. **Restrict novelty to the topology level, keep primitives SOTA.** Use D1 + one of the elegant encodings (A2, A4, or D6). The "novelty" is in the wiring, not the primitives.
2. **Accept a small accuracy gap in exchange for genuine novelty.** Use a non-neural substrate (D3, D4, D5) and target "competitive within substrate class." This is a different brief.
3. **Stage the work: first prove the framework on standard primitives, then expand.** Most defensible.

---

## 6. Three concrete syntheses (each minimal, not collage)

The brief warned against collage. So I propose three *minimal* syntheses — each is one core idea plus the smallest possible supporting machinery. None of them are "all of the above."

### Synthesis 1: "Latent Wiring Discovery" (LWD)

**Core idea.** Search for the wiring of a network whose nodes are taken from a fixed SOTA primitive library. Encode the wiring as a point in a continuous latent space via a graph VAE. Search with CMA-ES. Evaluate with early-stop fitness. No developmental program, no exotic primitives, no quality-diversity archive.

**Components:** A3 (latent VAE) + C4 (CMA-ES) + B3 (early-stop) + D1 (primitive library).
**Unifying principle:** *The only thing we evolve is the wiring. Primitives are fixed and SOTA. The encoding is a smooth latent space.*
**Why this is elegant:** Three moving parts, each well-understood. The VAE gives a smooth search space; CMA-ES explores it; early-stop makes evaluation affordable. No exotic components.
**Why this might fail:** The VAE's expressive ceiling caps novelty. The wiring patterns it can produce are bounded by its training corpus. If we want truly surprising topologies, this won't deliver them.
**Path to SOTA:** Plausible. Primitive library gives us ConvNeXt/ViT-class building blocks. Wiring search may discover better arrangements.

### Synthesis 2: "Developmental NEAT" (D-NEAT)

**Core idea.** A small CPPN-like genome encodes a graph-grammar developmental program. The program grows a phenotype over a fixed number of developmental steps, using a typed primitive library. The phenotype is trained by gradient descent on the image task. The genome is evolved with classical NEAT (speciation preserved, innovation numbers preserved). A *denoising stability regularizer* ensures phenotypes are robust to noise in the developmental process.

**Components:** A4 (developmental) + D1 (primitive library) + classical NEAT + stability regularizer.
**Unifying principle:** *The genome is a developmental program; the phenotype is a typed DAG over SOTA primitives; stability is explicitly enforced.*
**Why this is elegant:** Preserves NEAT's core ideas (complexification, speciation, historical markings) while extending the encoding to scale. The stability regularizer is the single new idea that makes it work.
**Why this might fail:** Developmental instability may be unsolvable with a simple denoising regularizer. The grammar may collapse to regular patterns (CPPN's failure mode).
**Path to SOTA:** Plausible if the developmental program can produce hierarchical structures (cells, stages). The stability regularizer is the research bet.

### Synthesis 3: "Topology Field" (TF)

**Core idea.** Define the network as a continuous field over a learned 2D or 3D manifold. Each point on the manifold has a probability distribution over primitives and a set of connection strengths to neighboring points. Sample a discrete topology from the field (via Gumbel-softmax over primitives, Bernoulli over connections), train it, and backpropagate the fitness signal through the sampling to update the field. After training, distill the field into a single discrete topology for deployment.

**Components:** D6 (continuous field) + differentiable sampling + D1 (primitive library).
**Unifying principle:** *The topology is a field; discrete networks are samples from it.* This is the most "outside the box" of the three.
**Why this is elegant:** Truly continuous search over arbitrary topologies. Gradient signal flows directly into topology decisions. No VAE, no grammar, no evolution — just a field and a sampler.
**Why this might fail:** Gumbel-softmax over large primitive vocabularies is known to be unstable. Variance of the gradient signal may be too high. Manifold topology (2D? 3D? learned?) is a design problem.
**Path to SOTA:** Speculative. The differentiable field is attractive but unproven at vision scale.

### Comparison of the three syntheses

| Property | LWD | D-NEAT | TF |
|---|---|---|---|
| Novelty potential | Medium | High | High |
| SOTA path | Strong | Plausible | Speculative |
| Elegance | High | High | Very high |
| Implementation risk | Low | Medium | High |
| Compute needed | Medium | High | Medium |
| Novelty ceiling | Bounded by VAE corpus | Bounded by grammar | Bounded by primitive library |
| Recovers classical NEAT ideas? | No (replaces with VAE) | Yes (preserves speciation, complexification) | No (replaces with field) |

**My current lean.** D-NEAT is the most faithful to the brief — it modernizes NEAT (preserving its core ideas) rather than replacing it. But LWD has the strongest empirical basis. TF is the most elegant *if* it works.

---

## 7. Open questions and risk map

For each synthesis, the critical questions whose answers determine viability:

### LWD
- Q1: Can a graph VAE encode the *functionally relevant* features of an architecture (not just graph isomorphism)?
- Q2: How do we generate the training corpus for the VAE? Random architectures + known-good? Does the corpus distribution dominate the search?
- Q3: How do we handle invalid decoded graphs (e.g., disconnected, type-mismatched)?

### D-NEAT
- Q4: Can a denoising regularizer make developmental programs stable enough for evolution?
- Q5: What is the right complexity for the genome? Too small → cannot express enough; too large → search explodes.
- Q6: How do we type-check developmental outcomes (e.g., attention needs Q, K, V)?
- Q7: Does classical NEAT speciation scale to phenotypes with thousands of nodes?

### TF
- Q8: Can Gumbel-softmax handle primitive vocabularies of 10+ operations?
- Q9: What manifold topology (2D, 3D, learned) gives the right inductive bias for vision?
- Q10: Is the variance of the gradient signal manageable, or do we need variance reduction techniques?

### Cross-cutting
- Q11: What is the right primitive library? Minimal (conv, attention, norm, activation) or rich (with pooling, gating, MoE, etc.)?
- Q12: What dataset(s) to validate on? CIFAR-10 for iteration speed, ImageNet for SOTA comparability, or both?
- Q13: What is the compute budget? This determines feasibility of everything.
- Q14: How do we measure "novelty" of a discovered topology, to evaluate whether the system is actually discovering?

---

## 8. Suggested first experiments (to disambiguate between syntheses)

These experiments are designed to be cheap and decisive. Each one resolves one of the open questions above.

**Experiment 1 (resolves Q1, Q3): Graph VAE sanity check.**
Train a graph VAE on NAS-Bench-201's ~15K architectures. Measure reconstruction quality and latent-space smoothness (does interpolating between two architectures produce valid intermediate architectures?). This is a few days of work and tells us whether LWD is viable.

**Experiment 2 (resolves Q4): Developmental stability test.**
Implement a tiny CPPN → phenotype mapping on MNIST. Train a small CPPN to produce a CNN. Add noise to the CPPN's developmental process. Measure how much the phenotype's accuracy changes. If a denoising regularizer can keep accuracy variance below ~2%, D-NEAT is viable.

**Experiment 3 (resolves Q8, Q10): Differentiable field on a toy task.**
Implement TF on CIFAR-10 with a small primitive library (3×3 conv, 1×1 conv, identity, ReLU, BN). Use Gumbel-softmax with temperature annealing. Measure whether the gradient signal is strong enough to converge. This is a week of work and tells us whether TF is viable.

**Experiment 4 (resolves Q11): Primitive library ablation.**
Run LWD with a minimal library vs a rich library. Compare best-found accuracy after a fixed compute budget. Tells us how much the primitive library matters.

**Experiment 5 (resolves Q14): Novelty metric.**
Define a "behavioral novelty" metric (e.g., feature representations of the trained networks, clustered). Apply to architectures discovered by LWD vs hand-designed ConvNeXt. Tells us whether LWD is actually discovering anything new.

---

## 9. Honest meta-reflection

A few things I want to flag about my own analysis, in the spirit of intellectual honesty:

1. **The brief's tension is real and unresolved.** "Novel designs" and "SOTA-comparable performance" pull in opposite directions. My three syntheses each make a different choice about this tension. The user should weigh in on which trade-off they prefer before we commit.

2. **I may be over-indexing on elegance.** The brief warned against collage, but it also warned against premature commitment. Some collages (E1, LAO) might actually be the pragmatic right answer even if they aren't beautiful. Worth flagging.

3. **The non-neural directions (D3-D5) are intellectually exciting but probably not on brief.** I included them for completeness. If the user wants to pursue one, the SOTA constraint should be relaxed.

4. **The biggest risk across all directions is the primitive library.** If we restrict to known primitives, "discovery" is really "wiring discovery." This may be fine — wiring matters a lot — but we should be honest about it.

5. **Compute is the elephant in the room.** None of these directions are cheap. A serious run at ImageNet-scale topology discovery is thousands of GPU-hours. The user should confirm the compute budget before we commit to any plan.

6. **Pretraining is not addressed.** SOTA vision today uses pretrained backbones. A from-scratch topology search competes against pretrained models, which is unfair. Either we (a) search for topologies on top of frozen features, (b) search for topologies that pretrain well, or (c) accept that we're comparing against from-scratch baselines. This needs to be decided.

---

## 10. Next steps

This document represents the *deliberation* phase. The next phase is to:

1. Pick one synthesis (or propose a fourth).
2. Resolve the most critical open question for that synthesis with a small experiment.
3. If the experiment is positive, write a more detailed methodology document and a proof-of-concept implementation.

I will not commit to a methodology until the user has reviewed this analysis and indicated which direction(s) to pursue.
