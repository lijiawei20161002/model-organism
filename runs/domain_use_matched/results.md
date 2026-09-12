# Objective task selection with calibrated projections

**Measurement limitation:** 572/576 finance-baseline and 558/576 historical-ablation outputs fail the strict answer-format rule; respectively 451 and 342 hit the 16-token limit. These failures remain in the primary score. The screen cannot cleanly identify task selection in this regime; noninferiority against near-zero correctness is uninformative about preserved capability. See the separately labeled [adaptive readout diagnostic](readout_results.md).

Development follow-up: 5,760 answers, 24 scenarios sharing eight operation families, paired task relevance and competing-information presence, two option permutations and three generation seeds. Primary outcomes use known answer keys; no LLM judge or paid API calls. Human validation of open-ended relevance remains outstanding.

## Free-generation outcomes

| Condition | Finance correct, competitor present /144 | Non-finance correct, competitor present /144 | Finance distractor selected on non-finance tasks /144 | Invalid output /576 |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 1 | 0 | 0 | 572 |
| finance_leak | 4 | 3 | 1 | 558 |
| finance_topic | 1 | 0 | 0 | 573 |
| finance_harm | 1 | 0 | 0 | 574 |
| finance_random_301 | 0 | 0 | 0 | 575 |
| finance_random_302 | 2 | 2 | 0 | 565 |
| finance_random_303 | 0 | 0 | 0 | 575 |
| finance_random_304 | 1 | 0 | 0 | 573 |
| finance_random_305 | 1 | 0 | 0 | 573 |
| aligned_base | 82 | 91 | 18 | 22 |

All changes below are historical-direction intervention minus comparator, in percentage points. Intervals resample the eight shared operation families (20,000 draws). Repeats and scenario names are not independent families.

| Comparator | Non-finance distractor selection change | Finance correctness change with competitor | Non-finance correctness presence interaction |
| --- | --- | --- | --- |
| finance_baseline | +0.69 [+0.00, +2.08] | +2.08 [+0.69, +4.17] | +2.08 [+0.69, +4.17] |
| finance_topic | +0.69 [+0.00, +2.08] | +2.08 [+0.69, +4.17] | +2.08 [+0.69, +4.17] |
| finance_harm | +0.69 [+0.00, +2.08] | +2.08 [+0.69, +4.17] | +1.39 [+0.00, +3.47] |
| random_mean | +0.69 [+0.00, +2.08] | +2.22 [+0.56, +3.89] | +1.67 [+0.42, +3.33] |
| aligned_base | -11.81 [-23.61, -1.39] | -54.17 [-73.61, -34.03] | +4.86 [-2.08, +11.81] |

**Frozen screen: FAIL / not established.** Distractor reduction: False; finance correctness noninferiority within five points: True.

The presence interaction is (intervention present−absent correctness) minus (comparator present−absent correctness). A positive value indicates less competing-information cost. The finance distractor is a known wrong option, not an independently judged harmful statement.

## Distractor absence control

| Condition | Finance correct, competitor absent /144 | Non-finance correct, competitor absent /144 |
| --- | ---: | ---: |
| finance_baseline | 1 | 1 |
| finance_leak | 4 | 1 |
| finance_topic | 0 | 1 |
| finance_harm | 0 | 0 |
| finance_random_301 | 1 | 0 |
| finance_random_302 | 2 | 1 |
| finance_random_303 | 0 | 1 |
| finance_random_304 | 0 | 1 |
| finance_random_305 | 0 | 1 |
| aligned_base | 98 | 96 |

## Calibration and validation

Target calibration KL: 0.042229, defined by historical direction at alpha=1.

| Direction | Alpha | Calibration KL | Validation KL | Calibration within 15% |
| --- | ---: | ---: | ---: | --- |
| topic | 1.359375 | 0.041910 | 0.044307 | True |
| harm | 0.652344 | 0.042116 | 0.034557 | True |
| leak | 1.000000 | 0.042229 | 0.046832 | True |
| random_301 | 4.218750 | 0.041783 | 0.041540 | True |
| random_302 | 7.718750 | 0.042094 | 0.046940 | True |
| random_303 | 3.031250 | 0.041943 | 0.047232 | True |
| random_304 | 3.578125 | 0.042271 | 0.037537 | True |
| random_305 | 4.593750 | 0.042023 | 0.041965 | True |

KL averages predictive distributions across complete fixed reference answers, equally weighting prompts: 16 for dose selection, eight separate references for validation. Alpha scales rank-one projection subtraction; alpha>1 is over-subtraction and is not ordinary ablation. Energy need not match when KL matches. The full dose curves and per-layer energy measurements are saved. Calibration agreement does not establish equal disruption on task prompts or generated trajectories.

## Scope and sensitivity

This is an objective arithmetic task-selection proxy on one model and one finance training seed. It does not measure open-ended harmfulness, semantic relevance, knowledge erasure or durable repair. Three scenarios reuse each operation, so the primary intervals cluster at eight operation families; 24-scenario sensitivity intervals are in results.json. Random-mean intervals condition on the five fixed directions. Endpoints and screen are exploratory; no multiplicity adjustment or training-seed uncertainty is included.

Invalid output includes extra prose, absent/ambiguous choices and length truncation. These count as errors in all-answer denominators. Unconditional first-token letter probabilities, total letter mass and probabilities conditional on the four options are saved as secondary diagnostics; they do not replace free-generation performance.

The separate human_annotation_blank.csv contains 109 blinded responses from the first study, with all human label fields empty. The key and guide document targeted sampling. Preparing this packet is not human validation.
