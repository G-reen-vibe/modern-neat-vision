# Modern NEAT for Vision

Research agent workspace for modernizing NEAT (NeuroEvolution of Augmenting Topologies) to discover arbitrary graph topologies for image classifiers with performance comparable to SOTA.

## Status

- **Phase 1 (Deliberation):** ✅ Complete. See [`RESEARCH_NOTES.md`](./RESEARCH_NOTES.md).
- **Phase 2 (Setup):** ✅ Complete. Environment, datasets, baselines, evaluation framework, and D-NEAT scaffold all in place. See [`DECISION.md`](./DECISION.md).
- **Phase 3 (Baselines):** Pending. Run all baselines on all datasets across multiple seeds.
- **Phase 4 (D-NEAT implementation):** Pending. Implement developmental program, NEAT loop, stability regularizer.

## Decision: pursuing D-NEAT

After surveying ~25 directions across five families, I committed to **D-NEAT (Developmental NEAT)** — a CPPN-like genome that encodes a graph-grammar developmental program, producing typed DAGs over a small primitive library (Conv, Attention, Norm, etc.). The single research bet is that **a denoising stability regularizer can make developmental programs stable enough for evolution**. Full rationale in [`DECISION.md`](./DECISION.md).

## Environment

- Python 3.12, CPU-only PyTorch 2.x
- 2 CPUs, 4 GB RAM, 9 GB disk → small models, short schedules, 3 seeds minimum
- Benchmarks: Fashion-MNIST (sanity), CIFAR-10 (primary), CIFAR-100 (secondary)
- Baselines: 3-layer CNN, ResNet-18, MobileNetV3-Small, EfficientNet-B0, DeiT-Tiny

## Setup

```bash
# Install CPU-only PyTorch + dependencies
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

# Verify the stack
python3 scripts/sanity_check.py
```

## Running baselines

```bash
# Single run
python3 scripts/train_baseline.py --dataset cifar10 --model resnet18 --seed 0

# Full sweep (very long)
python3 scripts/run_baselines.py --seeds 0 1 2

# Quick smoke test
python3 scripts/run_baselines.py --datasets fashionmnist --models simple_cnn --seeds 0

# Aggregate results
python3 scripts/aggregate_results.py
```

## Project structure

```
modern-neat/
├── RESEARCH_NOTES.md         # Phase 1: deliberation across ~25 directions
├── DECISION.md               # Phase 2: D-NEAT decision + protocol
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── dataset/{fashionmnist,cifar10,cifar100}.yaml
│   └── model/{simple_cnn,resnet18,mobilenetv3_small,efficientnet_b0,deit_tiny}.yaml
├── src/
│   ├── data/datasets.py             # Fashion-MNIST/CIFAR-10/CIFAR-100 loaders
│   ├── models/baselines.py          # timm wrappers, adapted for small images
│   ├── models/dneat/                # D-NEAT scaffold (genome, developmental, primitives, phenotype)
│   ├── train/{trainer,optimizer}.py # generic training loop + AdamW + cosine schedule
│   ├── eval/{metrics,aggregate}.py  # accuracy, FLOPs, latency, multi-seed CIs
│   ├── search/{neat,speciation,stability}.py  # D-NEAT evolution loop scaffold
│   └── utils/{config,logging,seed}.py
├── scripts/
│   ├── train_baseline.py
│   ├── run_baselines.py
│   ├── aggregate_results.py
│   └── sanity_check.py
└── results/                  # gitignored
    ├── runs/                 # per-run output (config, metrics.csv, final.json)
    └── summaries/            # aggregated tables
```

## Evaluation protocol

- **Seeds:** 3 per (model, dataset) combination by default.
- **Metrics:** Top-1/Top-5 accuracy, parameters, FLOPs, train time, inference latency.
- **Statistics:** mean ± std, plus 95% CI computed via Student's t-distribution.
- **Comparison test:** paired t-test or Wilcoxon signed-rank with Bonferroni correction.
