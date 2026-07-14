# Phase 2 Decision: Pursuing D-NEAT (Developmental NEAT)

**Date:** 2026-07-15
**Status:** Committed. This document records the decision and the constraints that shaped it.

---

## 1. The decision

After the Phase 1 deliberation across ~25 directions and three minimal syntheses (LWD, D-NEAT, Topology Field), I am committing to **D-NEAT (Developmental NEAT)** as the direction to implement and evaluate.

### Why D-NEAT over the alternatives

**Versus LWD (Latent Wiring Discovery).** LWD's graph VAE bounds novelty at the VAE's training corpus. Worse, the VAE itself becomes a second research problem (architecture, training, retraining). LWD also abandons NEAT's core ideas (speciation, complexification, historical markings). The brief asked to *modernize* NEAT, not replace it with a VAE. LWD is the safer empirical bet, but it concedes the spirit of the brief.

**Versus Topology Field.** TF is the most elegant *on paper*, but Gumbel-softmax over a large primitive vocabulary is known to be unstable, and there is no empirical evidence it scales to vision. With our compute budget (see §2), we cannot afford to bet on an unproven differentiable sampler.

**D-NEAT preserves the NEAT spirit.** Speciation, complexification, historical markings, and topology-as-genome all remain. The modernization is targeted at NEAT's actual failure modes (§1.2 of the Phase 1 notes):

| NEAT failure mode | D-NEAT's response |
|---|---|
| CPPN forces Cartesian substrates | Replace with a graph-grammar developmental program that produces arbitrary DAGs |
| Fitness evaluation is expensive | Inner-loop gradient descent on weights; early-stop fitness; candidate fitness cached |
| Speciation doesn't scale | Use a behavioral descriptor (learned-feature embedding of the trained phenotype) for speciation, not the legacy excess/disjoint gene metric |
| No gradient signal during topology search | Gradient descent trains weights; the developmental program is small enough that its parameters can also be optimized by gradient (mixed evolution + gradient) |
| Modern primitives can't be expressed | Typed primitive library: Conv, Attention, Norm, Activation, Pool, Skip, etc. |
| No integration with pretraining | Out of scope for Phase 2; revisit if D-NEAT's from-scratch results are competitive |

### The single research bet

The entire viability of D-NEAT rests on one hypothesis: **a denoising stability regularizer can make developmental programs stable enough for evolution.** This is testable with a cheap experiment (Phase 1, Experiment 2). If it fails, D-NEAT fails, and we pivot.

---

## 2. Compute reality check

The server environment is severely constrained:

| Resource | Available |
|---|---|
| CPU | 2 cores (no GPU) |
| RAM | 4.1 GB |
| Disk | 9.3 GB free |
| Torch | CPU-only build |

This rules out:
- ImageNet entirely (1.2M images × 224×224 — disk and time infeasible)
- CIFAR-100 with large batch sizes
- ViT-Base or larger
- Any NAS-style search with hundreds of full training runs

What is feasible:
- **CIFAR-10** (50K train, 10K test, 32×32): primary benchmark. ~3 minutes per epoch with small models.
- **CIFAR-100** (50K train, 10K test, 32×32, 100 classes): secondary benchmark, harder, slightly slower.
- **Fashion-MNIST** (60K train, 10K test, 28×28): fast sanity benchmark, ~30 seconds per epoch.

With these, a single small-model training run is ~15-60 minutes. 3 seeds × 4 baselines × 3 datasets = 36 runs ≈ 15-30 hours total. That is the realistic budget for the baseline evaluation. D-NEAT's search loop will be even more constrained — see §4.

### Implications for the experimental protocol

1. **Small models only.** ResNet-8/18, MobileNetV3-Small, DeiT-Tiny (or smaller). No ConvNeXt-Base, no ViT-Base.
2. **Short training schedules.** 50-100 epochs for baselines, 30-50 epochs for D-NEAT candidates during search.
3. **3 seeds minimum, 5 when affordable.** Report mean ± std and 95% CI.
4. **Single-process.** No parallel search; population size will be small (10-30).
5. **Aggressive disk management.** Datasets downloaded once; checkpoints kept only for the best seed; intermediate artifacts deleted.

---

## 3. Technology choices

### Language: Python 3.12

The ML ecosystem is Python-centric. PyTorch, timm, and the broader tooling are first-class here. No realistic alternative.

### Deep learning framework: PyTorch (CPU-only build)

PyTorch 2.x with the CPU-only wheel from `https://download.pytorch.org/whl/cpu`. This is ~200 MB vs ~2 GB for the CUDA build, and we have no GPU anyway. PyTorch's CPU backend uses MKL-DNN (oneDNN) for fast conv/matmul on x86 CPUs.

### Architecture zoo: timm

`timm` (PyTorch Image Models) provides SOTA architectures with consistent interfaces. We will use:
- **ResNet-18** — classic CNN baseline. Pretrained weights available for CIFAR-adapted variants.
- **MobileNetV3-Small** — efficient baseline.
- **EfficientNet-B0** — modern compound-scaled CNN.
- **DeiT-Tiny** — modern transformer baseline (smaller than ViT-Tiny).
- A simple **3-layer CNN** — the trivial baseline that D-NEAT's discovered topologies must beat.

### Graph manipulation: networkx

For representing, mutating, and analyzing the topology graph. Pure Python, no native deps. Slightly slow but adequate for graphs of <1000 nodes.

### Statistics: numpy + scipy

`scipy.stats.sem` for standard error, `scipy.stats.t` for confidence intervals. Standard research practice.

### Configuration: YAML + argparse

Configs in `configs/` as YAML files for reproducibility. CLI overrides via argparse for quick iteration.

### Logging: CSV + JSON

CSV for per-epoch metrics (easy to load into pandas). JSON for run metadata. No TensorBoard — too heavy for our disk budget.

### Experiment tracking: simple file-based

Each run gets a directory under `results/<dataset>/<model>/<seed>/`. Contains: config.json, metrics.csv, final_metrics.json. Aggregation script reads all runs and produces summary tables.

### Reproducibility: seed everything

`torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic`. Even on CPU, this matters for reproducibility.

---

## 4. Evaluation protocol

### Baselines

| Model | Params (CIFAR) | Why included |
|---|---|---|
| 3-layer CNN | ~50K | Trivial baseline; D-NEAT must beat this |
| ResNet-18 | ~11M | Classic CNN, well-understood |
| MobileNetV3-Small | ~2.5M | Efficient baseline |
| EfficientNet-B0 | ~5.3M | Modern compound scaling |
| DeiT-Tiny | ~5.5M | Modern transformer baseline |

### Datasets

| Dataset | Classes | Image size | Train size | Why included |
|---|---|---|---|---|
| Fashion-MNIST | 10 | 28×28 | 60K | Fast sanity; tests basic capability |
| CIFAR-10 | 10 | 32×32 | 50K | Standard small-image benchmark |
| CIFAR-100 | 100 | 32×32 | 50K | Tests fine-grained discrimination |

### Metrics

| Metric | How measured |
|---|---|
| Top-1 accuracy | Standard, on test set |
| Top-5 accuracy | For CIFAR-100 only (10 classes is meaningless) |
| Parameters | `sum(p.numel() for p in model.parameters())` |
| FLOPs | `thop.profile` (or `fvcore.nn.FlopCountAnalysis`) |
| Train time | Wall clock, includes data loading |
| Inference latency | Median of 100 forward passes on batch size 1 |
| Convergence epoch | First epoch where val accuracy ≥ 0.9 × final |

### Statistical protocol

- **Seeds:** 3 per (model, dataset) combination, ideally 5 for the headline results.
- **Report:** mean ± std, plus 95% confidence interval computed via Student's t (df = n-1).
- **Comparison test:** paired t-test or Wilcoxon signed-rank for D-NEAT vs baselines, with Bonferroni correction for multiple comparisons.
- **Per-seed results:** always reported in supplementary tables, never just the mean.

### Computing budget per baseline run

| Model | Dataset | Epochs | Est. time/epoch | Total |
|---|---|---|---|---|
| 3-layer CNN | FMNIST | 30 | ~30s | 15 min |
| 3-layer CNN | CIFAR-10 | 50 | ~1 min | 50 min |
| 3-layer CNN | CIFAR-100 | 50 | ~1 min | 50 min |
| ResNet-18 | FMNIST | 30 | ~2 min | 60 min |
| ResNet-18 | CIFAR-10 | 100 | ~3 min | 5 hr |
| ResNet-18 | CIFAR-100 | 100 | ~3 min | 5 hr |
| MobileNetV3-S | FMNIST | 30 | ~1 min | 30 min |
| MobileNetV3-S | CIFAR-10 | 100 | ~2 min | 3.3 hr |
| MobileNetV3-S | CIFAR-100 | 100 | ~2 min | 3.3 hr |
| EfficientNet-B0 | FMNIST | 30 | ~1.5 min | 45 min |
| EfficientNet-B0 | CIFAR-10 | 100 | ~3 min | 5 hr |
| EfficientNet-B0 | CIFAR-100 | 100 | ~3 min | 5 hr |
| DeiT-Tiny | FMNIST | 30 | ~2 min | 60 min |
| DeiT-Tiny | CIFAR-10 | 100 | ~4 min | 6.7 hr |
| DeiT-Tiny | CIFAR-100 | 100 | ~4 min | 6.7 hr |

Total for 1 seed of all baselines: ~46 hours.
With 3 seeds: ~138 hours (~6 days continuous).
This is at the edge of feasibility — we will likely do 3 seeds for CIFAR-10 only, and 1-2 seeds for CIFAR-100 / FMNIST initially.

### D-NEAT search budget (estimated)

D-NEAT's outer loop evaluates candidate topologies. If each candidate trains for 30 epochs on CIFAR-10 at ~2 min/epoch = 60 min per candidate, a population of 20 over 10 generations = 200 candidates × 60 min = 200 hours. **This is infeasible on our compute.**

Mitigations (to be revisited when we get there):
- Train candidates for fewer epochs (10-15) during search; retrain the best for full schedule at the end.
- Use early-stop with learning curve extrapolation (Phase 1, B3).
- Use weight sharing across population members (controversial — may compromise the search).
- Reduce population size to 10, generations to 5.

Realistic target: 50 candidate evaluations × 30 min = 25 hours per D-NEAT run. Still very expensive but feasible for one or two runs.

---

## 5. Project structure

```
modern-neat/
├── README.md
├── RESEARCH_NOTES.md
├── DECISION.md                  # this file
├── requirements.txt
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── dataset/
│   │   ├── fashionmnist.yaml
│   │   ├── cifar10.yaml
│   │   └── cifar100.yaml
│   └── model/
│       ├── simple_cnn.yaml
│       ├── resnet18.yaml
│       ├── mobilenetv3_small.yaml
│       ├── efficientnet_b0.yaml
│       └── deit_tiny.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py          # dataset loaders with caching
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py         # timm wrappers, adapted for CIFAR
│   │   ├── simple_cnn.py        # 3-layer CNN baseline
│   │   └── dneat/               # D-NEAT phenotype construction (scaffold)
│   │       ├── __init__.py
│   │       ├── genome.py        # CPPN-like genome
│   │       ├── developmental.py # graph grammar developmental program
│   │       ├── primitives.py    # typed primitive library
│   │       └── phenotype.py     # genome → trainable network
│   ├── search/
│   │   ├── __init__.py
│   │   ├── neat.py              # NEAT evolution loop
│   │   ├── speciation.py        # speciation logic
│   │   └── stability.py         # denoising stability regularizer
│   ├── train/
│   │   ├── __init__.py
│   │   ├── trainer.py           # generic training loop
│   │   ├── optimizer.py         # optimizer + scheduler factory
│   │   └── augment.py           # CIFAR augmentation pipeline
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py           # accuracy, FLOPs, params, latency
│   │   └── aggregate.py         # multi-seed aggregation + CIs
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # YAML config loader
│       ├── seed.py              # seed everything
│       └── logging.py           # CSV/JSON logging
├── scripts/
│   ├── train_baseline.py        # train one baseline, one seed
│   ├── run_baselines.py         # run all baselines across seeds
│   ├── aggregate_results.py     # produce summary tables
│   ├── sanity_check.py          # quick pipeline test
│   └── dneat_search.py          # (placeholder) D-NEAT search
└── results/                     # gitignored
    ├── runs/                    # per-run outputs
    └── summaries/               # aggregated tables
```

---

## 6. What "setup complete" means

This phase is complete when:

1. ✅ Environment installed and verified.
2. ✅ All three datasets load and cache correctly.
3. ✅ All five baselines can be instantiated.
4. ✅ A sanity-check training run completes one epoch on CIFAR-10 with a tiny model.
5. ✅ The aggregation script produces a summary table from a few dummy runs.
6. ✅ D-NEAT scaffold exists (genome, developmental program, NEAT loop) but is not yet functional.
7. ✅ Everything committed to GitHub.

What is explicitly **not** in scope for this phase:
- Actually training all baselines (that's Phase 3).
- Implementing D-NEAT's search loop (that's Phase 4).
- Iterating on the D-NEAT idea (that's Phase 5+).
