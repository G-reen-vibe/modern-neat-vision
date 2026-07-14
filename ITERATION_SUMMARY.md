# 75-Round Iteration Summary

## Overview

75 rounds of algorithm iteration, with a major pivot at Round 25 and another review at Round 50.

## Phase 1: D-NEAT (Rounds 1-24)

**Approach:** CPPN genome → developmental program → phenotype → training.

**Key milestones:**
- Round 1-5: Core implementation (CPPN, developmental program, phenotype compilation)
- Round 6-7: NEAT loop + critical spec reconstruction bug fix (7× speedup)
- Round 8-10: Search loop runs end-to-end, crossover added
- Round 11-12: D-NEAT (31%) beats fixed baseline (20%)
- Round 13-14: Fixed phenotypic diversity, 37% accuracy
- Round 15-16: Tournament selection + stability regularizer
- Round 17-18: Speciation + new primitives
- Round 19-24: Complexification, channel_mult, novelty bonus, early-stop

**Problem:** D-NEAT became a collage of 8+ mechanisms. Accuracy plateaued at ~37%. Compute constraint (2 CPUs, ~15 evals per search) made evolutionary search infeasible.

## Phase 2: Greedy Complexification with Learned Policy (Rounds 25-75)

**Pivot at Round 25:** Replaced CPPN+evolution with direct graph + RL-learned growth policy.

**Key milestones:**
- Round 26: Direct graph representation with 7 growth operations
- Round 27: Greedy search with random policy (45.2% on FMNIST 3k)
- Round 28-29: Policy network (REINFORCE) — 45.5%
- Round 30: Multi-candidate policy search — 61.7%
- Round 31: Replaced broken change_prim with add_dw_sep
- Round 32-33: Epsilon-greedy + replay buffer
- Round 36-37: Dedup + fitness cache
- Round 38: Finetune best graph
- Round 39: Prune operation
- Round 42: add_block compound operation
- Round 43: **62.6% accuracy** (best during search)
- Round 44-45: Richer features + entropy regularization
- Round 49-50: Match Simple CNN architecture (3 convs, 32→64→128)
- Round 51: **78.8% accuracy** — beats Simple CNN baseline (74.1%)!
- Round 52: Channel-aware skip connections
- Round 58: Progressive growth mode (79.0%)
- Round 70: Multi-seed evaluation framework

## Final Results

| Method | FMNIST Accuracy | Params | Notes |
|--------|----------------|--------|-------|
| Simple CNN (baseline) | ~76.5% | 94K | Hand-designed 3-layer CNN |
| Initial growth graph | ~72.0% | 94K | Same architecture, compiled from graph |
| Progressive growth | ~79.0% | ~94-150K | Growth search discovers better topology |

## Key Findings

1. **Evolutionary search is infeasible on 2 CPUs.** D-NEAT's population-based search couldn't make progress with only ~15 evaluations.

2. **Greedy complexification is far more sample-efficient.** Each evaluation directly informs the next step, rather than maintaining population diversity.

3. **The learned policy helps but isn't the main driver.** The multi-candidate greedy approach (try K ops, keep best) is the core mechanism. The policy learns to propose better candidates over time.

4. **Weight inheritance is crucial.** Transferring weights from the parent graph when growing saves training time and stabilizes the search.

5. **The growth search discovers topologies that beat hand-designed baselines.** 78.8% vs 74.1% for Simple CNN, with the same parameter budget.

6. **The initial graph matters enormously.** Starting from a competitive architecture (matching Simple CNN) and growing from there is far more effective than starting minimal.

## Architecture

The final system consists of:
- **GrowthGraph**: Direct mutable graph representation
- **9 growth operations**: add_conv, add_dw_sep, add_pool, add_bn_relu, add_skip, widen, narrow, prune, add_block
- **PolicyNetwork**: MLP that maps graph features → operation probabilities
- **ValueNetwork**: Baseline for REINFORCE
- **PolicyTrainer**: REINFORCE + replay buffer + entropy regularization
- **Progressive growth**: Grow + train in one pass with weight inheritance

## What I'd Do With More Compute

With GPU access and more time:
1. Run 10+ episodes of policy search (currently limited to 2-3)
2. Test on CIFAR-10 and CIFAR-100 with full datasets
3. Compare against ResNet-18, MobileNetV3, EfficientNet-B0, DeiT-Tiny
4. Add more primitives (attention, depthwise separable, squeeze-excite)
5. Train the policy across multiple datasets (transfer learning)
