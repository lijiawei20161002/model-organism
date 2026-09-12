# Adaptive diagnostic: task preference with answer format held fixed

The two prefixed readouts were specified after observing high invalid output in the generation test. Bare first-token probabilities were prespecified. All 192 prompts and all ten conditions are retained. No new sampling or LLM judging is used. Prefixes are interventions on the response context, so they can change more than output format.

## bare

| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 26 | 30 | 0.5160 | 0.2599 |
| finance_leak | 26 | 31 | 0.5446 | 0.3553 |
| finance_topic | 26 | 32 | 0.5270 | 0.2684 |
| finance_harm | 26 | 33 | 0.5442 | 0.2439 |
| finance_random_301 | 25 | 30 | 0.5089 | 0.1491 |
| finance_random_302 | 24 | 31 | 0.5279 | 0.4579 |
| finance_random_303 | 28 | 30 | 0.4978 | 0.2964 |
| finance_random_304 | 27 | 31 | 0.5072 | 0.3318 |
| finance_random_305 | 26 | 29 | 0.5226 | 0.4422 |
| aligned_base | 31 | 31 | 0.6432 | 1.0000 |

All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.

| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |
| --- | --- | --- | --- |
| finance_baseline | -1.45 [-2.76, +0.01] | +9.54 [+7.58, +11.66] | +2.25 [-0.33, +5.33] |
| finance_topic | -2.08 [-4.64, -0.24] | +8.70 [+6.85, +10.49] | +0.26 [-1.07, +1.66] |
| finance_harm | -0.49 [-1.56, +0.67] | +11.14 [+9.57, +12.64] | -1.48 [-4.03, +0.86] |
| random_mean | -0.98 [-2.14, +0.13] | +1.99 [-0.34, +4.39] | +2.59 [+0.64, +4.54] |
| aligned_base | -1.27 [-8.13, +4.11] | -64.47 [-71.66, -57.31] | -17.82 [-25.45, -10.76] |

Competing-information interaction (change in conditional target preference with competitor present minus absent):

| Comparator | Finance tasks | Non-finance tasks |
| --- | --- | --- |
| finance_baseline | -1.36 [-2.61, -0.08] | -0.77 [-2.94, +1.62] |
| finance_topic | -2.61 [-5.55, +0.41] | -0.03 [-2.46, +2.52] |
| finance_harm | +0.79 [-1.08, +2.47] | +0.18 [-2.62, +3.55] |
| random_mean | -1.34 [-3.34, +0.92] | -0.19 [-1.89, +1.66] |
| aligned_base | -5.01 [-14.77, +4.23] | +2.64 [-3.23, +8.04] |

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +5.539 pp and a conditional-choice term of +0.987 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## answer_prefix

| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 19 | 31 | 0.4367 | 0.9741 |
| finance_leak | 21 | 30 | 0.4683 | 0.9915 |
| finance_topic | 19 | 28 | 0.4595 | 0.9880 |
| finance_harm | 20 | 33 | 0.4697 | 0.9816 |
| finance_random_301 | 14 | 27 | 0.4529 | 0.9682 |
| finance_random_302 | 19 | 28 | 0.4494 | 0.9731 |
| finance_random_303 | 20 | 31 | 0.4097 | 0.9310 |
| finance_random_304 | 18 | 30 | 0.4396 | 0.9937 |
| finance_random_305 | 19 | 27 | 0.4468 | 0.9951 |
| aligned_base | 29 | 33 | 0.6735 | 0.9499 |

All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.

| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |
| --- | --- | --- | --- |
| finance_baseline | +0.28 [-1.49, +1.83] | +1.73 [+1.58, +1.89] | +5.07 [+1.58, +9.17] |
| finance_topic | +0.51 [-0.99, +1.91] | +0.34 [+0.30, +0.38] | +2.69 [+1.13, +4.24] |
| finance_harm | +0.86 [-0.75, +2.21] | +0.98 [+0.85, +1.12] | +2.48 [-0.14, +5.88] |
| random_mean | -0.29 [-2.12, +1.25] | +1.92 [+1.71, +2.15] | +3.55 [+1.03, +6.34] |
| aligned_base | +1.42 [-2.51, +5.29] | +4.16 [-0.21, +11.31] | -28.18 [-43.39, -13.32] |

Competing-information interaction (change in conditional target preference with competitor present minus absent):

| Comparator | Finance tasks | Non-finance tasks |
| --- | --- | --- |
| finance_baseline | -0.80 [-2.62, +1.32] | -1.95 [-3.43, -0.48] |
| finance_topic | -1.39 [-3.64, +0.85] | -1.94 [-4.00, +0.31] |
| finance_harm | +2.75 [+0.20, +5.53] | -1.32 [-3.02, +0.50] |
| random_mean | -1.40 [-3.33, +0.70] | -1.63 [-3.13, -0.10] |
| aligned_base | +9.07 [+0.50, +17.02] | +2.94 [-3.59, +9.67] |

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +0.654 pp and a conditional-choice term of +4.727 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## option_prefix

| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 7 | 25 | 0.3472 | 0.8631 |
| finance_leak | 14 | 26 | 0.3754 | 0.8916 |
| finance_topic | 10 | 24 | 0.3527 | 0.9178 |
| finance_harm | 10 | 25 | 0.3666 | 0.8960 |
| finance_random_301 | 7 | 25 | 0.3660 | 0.8566 |
| finance_random_302 | 9 | 26 | 0.3622 | 0.8081 |
| finance_random_303 | 9 | 23 | 0.3406 | 0.7662 |
| finance_random_304 | 6 | 23 | 0.3391 | 0.9245 |
| finance_random_305 | 11 | 23 | 0.3721 | 0.9743 |
| aligned_base | 27 | 33 | 0.6614 | 0.0129 |

All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.

| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |
| --- | --- | --- | --- |
| finance_baseline | +0.54 [-1.19, +2.11] | +2.85 [+2.26, +3.38] | +4.77 [+2.07, +7.99] |
| finance_topic | +0.53 [-1.29, +1.91] | -2.61 [-3.23, -1.94] | +4.04 [+2.35, +5.94] |
| finance_harm | +1.27 [-0.39, +2.87] | -0.43 [-0.97, +0.08] | +3.56 [+1.08, +6.63] |
| random_mean | +0.28 [-1.38, +1.77] | +2.57 [+1.86, +3.14] | +3.09 [+1.11, +5.45] |
| aligned_base | +1.68 [-6.01, +7.63] | +87.88 [+86.10, +89.62] | -32.91 [-52.29, -13.69] |

Competing-information interaction (change in conditional target preference with competitor present minus absent):

| Comparator | Finance tasks | Non-finance tasks |
| --- | --- | --- |
| finance_baseline | -1.54 [-2.58, -0.45] | -1.06 [-2.95, +0.69] |
| finance_topic | -1.83 [-3.28, -0.53] | -1.22 [-3.52, +1.13] |
| finance_harm | +1.17 [-0.56, +2.91] | -0.65 [-2.74, +1.31] |
| random_mean | -1.86 [-2.79, -0.85] | -0.90 [-2.23, +0.37] |
| aligned_base | +12.65 [+4.71, +20.33] | +3.33 [-1.64, +8.89] |

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +0.757 pp and a conditional-choice term of +3.919 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## Interpretation boundary

The initial prefixed run used incorrect bare-letter token IDs at a whitespace boundary. Those outputs are archived under readout_tokenization_v1 and excluded from this report. Corrected candidates use the shared prefix of complete prefix-plus-letter encodings, with distinct single-token suffixes verified before inference.

Fixing an assistant prefix or conditioning on four letters is not successful autonomous task completion. Differences can reflect task knowledge, option mapping, context selection, or the imposed response prefix. This diagnostic cannot rescue the original failed screen. All readouts and controls must be considered; no best-prefix selection is justified. Raw probabilities and scenario-cluster sensitivity are retained. Human relevance annotation, open-ended generation validation, and causal localization remain separate requirements.
