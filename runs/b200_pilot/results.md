# B200 development pilot

Exploratory gate-pool results, five completions per prompt. All rates use all 480 answers.
Acceptable means coherent and non-flagged under the existing judge; relevance is not independently judged.
Harmful counts retain the historical coherence filter and do not measure harm in incoherent answers.

| Condition | Coherent | Flagged harmful | Acceptable |
|---|---:|---:|---:|
| b200_finance_baseline | 395/480 (82.3%) | 83/480 (17.3%) | 312/480 (65.0%) |
| b200_finance_leak | 421/480 (87.7%) | 55/480 (11.5%) | 366/480 (76.2%) |
| b200_finance_random_101 | 395/480 (82.3%) | 81/480 (16.9%) | 314/480 (65.4%) |
| b200_finance_random_102 | 408/480 (85.0%) | 113/480 (23.5%) | 295/480 (61.5%) |
| b200_finance_random_103 | 396/480 (82.5%) | 96/480 (20.0%) | 300/480 (62.5%) |
| b200_finance_random_104 | 385/480 (80.2%) | 94/480 (19.6%) | 291/480 (60.6%) |
| b200_finance_random_105 | 392/480 (81.7%) | 100/480 (20.8%) | 292/480 (60.8%) |

## Leakage ablation effects

| Comparison | Metric | Difference (pp) | 95% family bootstrap interval |
|---|---|---:|---:|
| Versus baseline | coherent | +5.42 | [-2.92, +13.33] |
| Versus baseline | harmful | -5.83 | [-15.21, +2.50] |
| Versus baseline | acceptable | +11.25 | [-0.62, +23.13] |
| Versus five-seed random mean | coherent | +5.37 | [-1.79, +12.04] |
| Versus five-seed random mean | harmful | -8.71 | [-17.08, -1.92] |
| Versus five-seed random mean | acceptable | +14.08 | [+5.71, +23.54] |

Intervals resample the same eight question families across conditions, conditional on these five random seeds.
They do not capture the full random-direction population, judge error, or held-out generalization.
Controls match projector rank, not activation energy removed. No direction was selected using these results.
