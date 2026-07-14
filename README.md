# Modern NEAT for Vision

Research agent workspace for modernizing NEAT (NeuroEvolution of Augmenting Topologies) to discover arbitrary graph topologies for image classifiers with performance comparable to SOTA.

## Status

**Phase 1 — Deliberation.** See [`RESEARCH_NOTES.md`](./RESEARCH_NOTES.md) for the full analysis. No methodology has been chosen yet; the document is the *deliberation*, not the conclusion.

## Summary of Phase 1

The notes cover:

- A critical re-reading of NEAT, HyperNEAT, and CoDeepNEAT — what actually worked and why progress stalled.
- A first-principles reframing of topology discovery as program synthesis, decomposed into four sub-problems (encoding, evaluation, search, primitives) plus a hidden zeroth problem (the unit of evolution).
- A taxonomy of ~25 research directions across five families (encoding-centric, evaluation-centric, search-centric, primitive-centric, hybrid/unifying), each with strengths, weaknesses, and a "killer objection."
- Cross-cutting themes and a comparative analysis table.
- Three concrete *minimal* syntheses (not collages):
  - **LWD (Latent Wiring Discovery):** Graph VAE + CMA-ES + early-stop, over a fixed SOTA primitive library.
  - **D-NEAT (Developmental NEAT):** CPPN-like genome grows a typed DAG via a graph grammar; classical NEAT speciation preserved; stability via a denoising regularizer.
  - **TF (Topology Field):** Continuous field over a learned manifold, differentiably sampled into discrete topologies.
- Open questions, risk map, and five cheap decisive experiments to disambiguate between the syntheses.
- An honest meta-reflection flagging tensions in the brief (novelty vs SOTA, primitive library as silent design choice, compute and pretraining as elephants).

## Next steps

Awaiting user input on:
1. Which synthesis (or variant) to commit to.
2. Resolution of the novelty-vs-SOTA tension.
3. Compute budget and target datasets.
4. Whether to operate in the from-scratch or pretrain-on-frozen-features regime.
