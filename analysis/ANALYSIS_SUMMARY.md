# Analysis Summary: Hows and Whys of Greedy Complexification

## Experiment Setup
- Dataset: Fashion-MNIST (single-channel 28×28 images, 10 classes)
- Compute: 2 CPUs, no GPU, 4GB RAM
- All experiments use the same training recipe: AdamW, lr=1e-3, cosine warmup, label smoothing 0.1

## Key Findings (with evidence)

### Finding 1: Previous "78.8% beats baseline" claim was invalid
**Experiment:** Controlled baseline with 3 seeds (01_controlled_baseline.py)
**Result:** Simple CNN: 79.5% ± 0.9%, Growth Graph: 76.6% ± 0.5%
**Why:** The 78.8% was a cherry-picked single run from the best candidate during search, not a fair comparison. With proper multi-seed evaluation, the growth graph was actually 3% BEHIND the baseline.

### Finding 2: Weight initialization was the root cause of the gap
**Experiment:** Architecture diff (02_architecture_diff.py)
**Result:** Growth Graph used PyTorch default uniform init (std ~0.02-0.05), while Simple CNN used Kaiming init (std ~0.06-0.19). The linear layer's std differed by 9×.
**Fix:** Applied Kaiming/Xavier init in compile_phenotype.
**After fix:** Gap narrowed from 3% to 2% (77.4% vs 79.5%).

### Finding 3: 1-epoch training is too noisy for search signal
**Experiment:** Epoch budget analysis (06_epoch_budget.py)
**Result:**
- 1 epoch: CV=0.064 (6.4% relative std — too noisy)
- 2 epochs: CV=0.016 (1.6% — stable, minimum viable)
- 3 epochs: CV=0.002 (very stable)
**Why:** With 1 epoch, the model hasn't converged enough for accuracy differences to reflect topology quality. The noise from random init dominates the signal.

### Finding 4: Operation usefulness depends on training budget
**Experiment:** Operation ablation with 1 vs 2 epochs (05, 07)
**1-epoch result:** ALL operations appeared harmful or neutral
**2-epoch result:** Clear ranking emerged:
- Helpful: add_pool (+3.25%), add_bn_relu (+0.80%), add_block (+0.20%)
- Neutral: add_skip (0.0%)
- Harmful: widen (-15.1%), add_conv (-35.1%), narrow (-40.8%)
**Why:** Operations that increase capacity (widen, add_conv) need more training to pay off. With only 2 epochs, simpler models that converge fast win. Spatial reduction (pool) and normalization (bn_relu) help immediately.

### Finding 5: Weight inheritance provides measurable benefit
**Experiment:** Weight inheritance ablation (10_weight_inheritance.py)
**Result:** With inheritance: 77.6%, Without: 75.2% (+2.4%)
**Why:** Inherited weights warm-start the new model. The benefit compounds: step 1 gains +0.5%, step 2 gains +2.4% (more shared weights = more benefit).

### Finding 6: 2000 samples is the minimum for meaningful signal
**Experiment:** Data size analysis (09_data_size.py)
**Result:**
- 500 samples: 30.9% (barely learning)
- 1000 samples: 35.6% (still poor)
- 2000 samples: 64.2% (threshold)
- 5000 samples: 75.1% (good)
**Why:** Below 2000 samples, the model can't learn enough for accuracy differences to reflect topology quality. The gradient signal is too weak.

### Finding 7: Discovered topologies are structurally valid
**Experiment:** Topology visualization (11_visualize_topology.py)
**Result:** The search discovers valid DAGs with main path + skip connections. Earlier confusion about "weird topologies" was a display bug (nodes listed by ID, not topological order).

## What the Search Actually Does

Given the compute constraint (2 CPUs, ~15 evaluations per search), the greedy complexification approach:

1. **Starts from a competitive baseline** (3 convs + 2 pools, matching Simple CNN)
2. **Tries 2-3 growth operations per step** (from the 9-operation vocabulary)
3. **Keeps the best candidate** if it improves accuracy
4. **Transfers weights** from the parent to warm-start training
5. **Trains for 2 epochs** on 3000 samples (minimum viable budget)

## What Works
- Weight inheritance (+2.4%)
- add_pool operation (+3.25%)
- add_bn_relu operation (+0.80%)
- Kaiming initialization (critical for fair comparison)
- 2-epoch minimum training budget

## What Doesn't Work (on this compute budget)
- Population-based evolution (D-NEAT) — too few evaluations
- Capacity-increasing operations (widen, add_conv) — can't train in 2 epochs
- 1-epoch evaluations — too noisy
- < 2000 training samples — model barely learns

## Open Questions
1. Does the learned policy beat random search? (Experiment 8 was inconclusive due to time)
2. How many growth steps until accuracy plateaus? (Experiment 13 was inconclusive)
3. Does the approach transfer to CIFAR-10? (Not yet tested with proper seeds)
4. What happens with more compute (GPU, more data, more epochs)?
