# Final Analysis: Greedy Complexification with Learned Growth Policy

## Executive Summary

After 22 rounds of controlled ablation and analysis, the key findings are:

### The Central Result

**The compiled growth graph and hand-designed Simple CNN are statistically indistinguishable** when properly initialized (p=0.69, Cohen's d=0.35, 3 seeds each).

| Model | Accuracy | 95% CI | Params |
|-------|----------|--------|--------|
| Simple CNN | 79.53% ± 0.90% | [76.8%, 82.3%] | 94,410 |
| Growth Graph | 79.20% ± 0.64% | [77.3%, 81.1%] | 94,186 |

### What We Got Wrong Initially

1. **The 78.8% "breakthrough" was a cherry-picked single run** — no error bars, not statistically valid
2. **The 2% gap was entirely due to initialization mismatch** — Linear layer used Xavier instead of Kaiming
3. **1-epoch ablations were misleading** — the fitness signal is too noisy (CV=0.064) to distinguish operations
4. **The search appeared to find "weird topologies"** — this was a display bug, not actual architectural issues

### What Actually Matters

Ranked by impact on the final result:

1. **Weight initialization (HUGE)** — Kaiming for Conv AND Linear. Without this, the growth graph is 2% worse. With it, the gap vanishes.
2. **Training budget (LARGE)** — 2 epochs minimum (CV=0.016). 1 epoch is too noisy (CV=0.064). 3+ epochs is ideal but expensive.
3. **Data size (LARGE)** — 2000+ samples needed. Below that, the model barely learns (30-36% accuracy).
4. **Weight inheritance (MEDIUM)** — +2.4% improvement when transferring weights between growth steps.
5. **Operation selection (MEDIUM)** — add_pool (+3.25%) and add_bn_relu (+0.80%) are the only consistently helpful operations at 2-epoch budget. Capacity-increasing ops (widen, add_conv) hurt because the model can't train the extra params.
6. **Learned policy (SMALL)** — the policy seems to learn operation preferences, but with only 2-3 episodes, the benefit is unclear. The multi-candidate greedy approach is the main driver.

### Why the Approach Works (When It Works)

The greedy complexification approach is effective because:

1. **Sample efficiency**: Each evaluation directly informs the next step (unlike evolution which maintains population diversity)
2. **Warm starting**: Weight inheritance means each step builds on the previous
3. **Small operations**: Each growth op is a local change, so the search space is tractable
4. **Competitive starting point**: Beginning from a known-good architecture (3 convs + pools) and growing from there

### Why the Approach Struggles (When It Struggles)

1. **Compute-bound**: With 2 CPUs, we can only afford ~15 evaluations per search. The policy can't learn well from so few samples.
2. **Budget-dependent operation usefulness**: Operations that increase capacity (widen, add_conv) need more training to pay off. At 2 epochs, only "free" improvements (pool, bn_relu) help.
3. **No transfer learning**: Each search starts from scratch. The policy doesn't accumulate knowledge across runs.
4. **No pruning of bad branches**: The greedy approach only moves forward. If a step is accepted but later turns out bad, there's no rollback.

### What the Search Actually Discovers

Given the compute constraint, the search consistently finds:
- **add_pool** is the most valuable operation (spatial reduction helps generalization)
- **add_bn_relu** is second (extra normalization helps training stability)
- **add_skip** is neutral (skip connections don't help on small datasets/budgets)
- **widen/add_conv** hurt (can't train extra capacity in 2 epochs)

The "discovered topologies" are essentially the initial graph + 1-2 extra pooling/normalization layers. This is a modest improvement, not a radical architecture discovery.

### Comparison to D-NEAT (Phase 1)

| Aspect | D-NEAT (Rounds 1-24) | Growth Policy (Rounds 25-75) |
|--------|----------------------|------------------------------|
| Search mechanism | Evolution (population, mutation, crossover) | Greedy complexification + RL policy |
| Evaluations needed | 100+ (infeasible on 2 CPUs) | 10-20 (feasible) |
| Best accuracy | ~37% (FMNIST 3k, 2ep) | ~79% (FMNIST 5k, 3ep) |
| Mechanisms | 8+ (CPPN, speciation, stability, novelty, etc.) | 1 (learned growth policy) |
| Elegance | Collage | Minimal |

The growth policy approach is dramatically more effective on this compute budget.

### Recommendations for Future Work

1. **Get a GPU** — the single biggest bottleneck. Would enable 10× more evaluations and 5+ epoch training.
2. **Test on CIFAR-10/ImageNet** — Fashion-MNIST is too easy; the search can't distinguish good from great topologies.
3. **Add more primitives** — attention, squeeze-excite, depthwise separable. The current vocabulary is too limited.
4. **Multi-objective search** — optimize for accuracy AND efficiency (FLOPs, params, latency).
5. **Transfer the policy** — train the policy on one dataset, apply it to another. This is the real test of whether the policy learns general architectural principles.

### Honest Assessment

The greedy complexification approach is a **pragmatically effective** method for topology discovery under severe compute constraints. It is NOT a breakthrough in architecture search — it's a sensible engineering approach that works when you can't afford real NAS.

The "discovered topologies" are modest improvements over the initial graph, not radical new architectures. The approach's value is in the **process** (sample-efficient, warm-started, policy-guided) rather than the **products** (the specific topologies found).

The key lesson from this analysis: **always use proper multi-seed evaluation with error bars**. Our initial claim of "beating the baseline" was wrong because we didn't. The actual result — matching the baseline — is still respectable but requires honest reporting.
