# Model Comparison Report

## Objective

Compare the popularity baseline recommender against the neural recommender
using ranking metrics at K.

## Compared Models

- Baseline: `popularity_baseline`
- Challenger: `neural_reranker`

## Metrics

| Metric | popularity_baseline | neural_reranker | Relative Difference |
|---|---:|---:|---:|
| precision_at_k | 0.0227% | 0.0040% | -82.35% |
| recall_at_k | 0.2266% | 0.0400% | -82.35% |
| hit_rate_at_k | 0.2266% | 0.0400% | -82.35% |
| coverage_at_k | 0.0110% | 0.0264% | +140.00% |

## Decision

Selected model: `popularity_baseline`

The popularity baseline performed better on hit rate, precision and recall
in this experiment.

The neural model increased catalog coverage, but it did not outperform the
baseline in ranking relevance.

## Technical Interpretation

The neural model was trained as a first PyTorch MLP/embedding-based
recommender and evaluated through candidate reranking.

The current version uses a sampled training run and a simple regression
objective over interaction scores.

Future improvements should test larger training samples, ranking-oriented
losses, stronger negative sampling and better candidate generation.
