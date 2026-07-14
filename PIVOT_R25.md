# Round 25 Pivot: Greedy Complexification with Learned Growth Policy

## What went wrong with D-NEAT (Rounds 1-24)

After 24 rounds, D-NEAT had accumulated 8+ mechanisms:
1. CPPN genome with innovation numbers
2. Developmental program (cells divide/differentiate on grid)
3. Stability regularizer (denoising)
4. Speciation (primitive histogram distance)
5. Novelty bonus (archive-based)
6. Tournament selection
7. Early-stop evaluation
8. Channel multiplier mutation

Despite all this, accuracy plateaued at ~31-37% on Fashion-MNIST 3k subset (2 epochs). The core issues:
- **Compute-bound**: Only ~15 evaluations per search on 2 CPUs. Evolution needs 100s.
- **Noisy fitness**: 2-epoch training gives 9-37% accuracy variance. Selection is nearly random.
- **Low diversity**: Developmental program produces similar phenotypes despite stochasticity.
- **Too constrained**: 6-cell phenotypes are essentially 3-layer CNNs — not "arbitrary topologies."

## The pivot

**New approach: Greedy Complexification with Learned Growth Policy**

Core idea: Replace evolutionary search with greedy growth guided by a small RL policy.

### Algorithm
1. Start with minimal network: `input → conv(16) → pool → head → output`
2. At each step:
   a. Encode current graph state as a feature vector
   b. Policy network (small MLP) outputs probability distribution over growth operations
   c. Sample K=3 candidate operations
   d. For each candidate: apply operation, compile, train briefly, evaluate
   e. Keep the best candidate; discard others
   f. Store (state, action, reward) for policy training
3. After N growth steps, train the policy with REINFORCE
4. Repeat for M episodes

### Growth operations
- `add_conv(after=node, channels=C)` — insert a conv layer
- `add_pool(after=node)` — insert a max-pool layer
- `add_skip(from=node_a, to=node_b)` — add a skip connection
- `add_bn_relu(after=node)` — insert BN+ReLU
- `widen(node, factor=F)` — increase channel count
- `change_primitive(node, new_prim)` — swap primitive type

### Why this is better
- **Sample-efficient**: Greedy search uses each evaluation to make progress, not to maintain population diversity.
- **Learns from experience**: The policy improves over episodes, avoiding bad operations.
- **Truly complexifies**: Starts minimal, grows one operation at a time.
- **Elegant**: One mechanism (learned growth policy), not a collage.
- **Preserves NEAT spirit**: Complexification, topology discovery, structure search.

### What I'm keeping from D-NEAT
- The primitive library (ConvBNReLU, DepthwiseSeparableConv, MaxPool2x, BatchNormReLU, GlobalAvgPool, LinearHead, Identity)
- The phenotype compilation with shape propagation and merge modules
- The training infrastructure (Trainer, optimizer, scheduler)
- The dataset loaders
