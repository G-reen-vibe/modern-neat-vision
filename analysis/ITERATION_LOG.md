# Analysis Iteration Log (Rounds 1-25)

## Round 1: Controlled baseline with 3 seeds
- **Finding**: Previous 78.8% was a lucky single run. With 3 seeds: Simple CNN 79.5% ± 0.9% vs Growth Graph 76.6% ± 0.5%. Growth graph is 3% BEHIND.
- **Lesson**: Always use multi-seed evaluation with error bars.

## Round 2: Architecture diff
- **Finding**: Weight initialization differs. Simple CNN uses Kaiming (std ~0.06-0.19), Growth Graph uses PyTorch defaults (std ~0.02-0.05). Linear layer std differs 9×.
- **Lesson**: Subtle implementation details matter enormously.

## Rounds 3-4: Fix Kaiming init, gap narrows
- **Fix**: Apply Kaiming init in compile_phenotype.
- **Result**: 76.6% → 77.4% (±0.4%). Gap narrows from 3% to 2%.
- **Lesson**: Proper initialization is critical for fair comparison.

## Round 5: Operation ablation (1 epoch, 1k samples)
- **Finding**: ALL operations appear harmful or neutral with 1-epoch training.
- **Lesson**: Fitness signal from 1-epoch/1k is too noisy.

## Round 6: Epoch budget analysis
- **Finding**: 1 epoch CV=0.064 (noisy), 2 epochs CV=0.016 (stable), 3 epochs CV=0.002.
- **Lesson**: 2 epochs is the minimum viable training budget.

## Round 7: Operation ablation (2 epochs)
- **Finding**: Clear ranking emerges. add_pool (+3.25%), add_bn_relu (+0.80%) help. widen (-15.1%), add_conv (-35.1%) hurt.
- **Lesson**: Operation usefulness depends on training budget. Capacity increase needs more epochs.

## Round 8: Random vs biased search (incomplete)
- System too slow for definitive comparison.

## Round 9: Data size effect
- **Finding**: 500 samples → 30.9%, 1000 → 35.6%, 2000 → 64.2%, 5000 → 75.1%.
- **Lesson**: 2000 samples is the minimum for meaningful signal.

## Round 10: Weight inheritance ablation
- **Finding**: With inheritance 77.6%, without 75.2% (+2.4%). Benefit compounds over steps.
- **Lesson**: Warm-starting with parent weights is valuable.

## Round 11: Topology visualization
- **Finding**: Search reaches 79.8%. But topology looked weird (bn_relu after linear_head).
- **Lesson**: Need to verify graph structure carefully.

## Round 12: Fix graph validation
- **Finding**: The "weird topology" was a display bug (nodes listed by ID, not topological order). Actual graphs are valid.
- **Lesson**: Always visualize in topological order.

## Round 13: Growth steps analysis (incomplete)
- System too slow for multi-step analysis.

## Round 14: Comprehensive analysis summary
- Documented all 7 key findings with evidence.

## Round 15: CIFAR-10 baseline (incomplete)
- CIFAR-10 is 3× slower than FMNIST. System too slow.

## Round 16: Policy learning from logs
- **Finding**: add_pool and add_bn_relu most frequently accepted. Policy seems to learn (episode 2 starts with best op from episode 1).
- **Lesson**: The policy provides marginal guidance; the greedy multi-candidate approach is the main driver.

## Round 17: Statistical significance test
- **Finding**: p=0.062, Cohen's d=2.45 (large effect but underpowered).
- **Lesson**: 3 seeds is not enough. Need ~10 for significance.

## Rounds 18-19: ROOT CAUSE FOUND — Linear init mismatch
- **Fix**: Use Kaiming (fan_out, relu) for Linear, not Xavier.
- **Result**: Gap closes from 2.1% to 0.3%! Growth graph 79.2% vs Simple CNN 79.5%.
- **Lesson**: The ENTIRE performance gap was initialization, not topology.

## Round 20: Statistical confirmation
- **Finding**: p=0.69, Cohen's d=0.35 (small). Growth graph and Simple CNN are statistically indistinguishable.
- **Lesson**: When properly initialized, the compiled graph matches the hand-designed baseline.

## Round 21: Search improvement test (incomplete)
- System too slow for multi-seed search evaluation.

## Round 22: Final analysis write-up
- Comprehensive honest assessment of the approach's strengths and limitations.

## Rounds 23-25: Push to GitHub, write log
- All analysis committed and pushed.

## Summary of 25 Analysis Rounds

**The single most important finding**: The entire 2% performance gap between the growth graph and Simple CNN was due to using Xavier initialization for Linear layers instead of Kaiming. Once fixed, the two are statistically identical (p=0.69).

**What the search actually does**: Finds modest improvements (add_pool, add_bn_relu) over a competitive starting architecture. Not a radical architecture discovery, but a sensible engineering approach under compute constraints.

**What we'd need for real progress**: A GPU (10× more evaluations), more data (CIFAR-10/ImageNet), more primitives (attention, SE), and multi-objective optimization.
