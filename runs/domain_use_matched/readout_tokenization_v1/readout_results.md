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

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +5.539 pp and a conditional-choice term of +0.987 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## answer_prefix

| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 15 | 23 | 0.3777 | 0.0000 |
| finance_leak | 23 | 25 | 0.4299 | 0.0000 |
| finance_topic | 19 | 27 | 0.4122 | 0.0000 |
| finance_harm | 19 | 24 | 0.4078 | 0.0000 |
| finance_random_301 | 13 | 23 | 0.3952 | 0.0000 |
| finance_random_302 | 19 | 22 | 0.3964 | 0.0000 |
| finance_random_303 | 17 | 18 | 0.3548 | 0.0000 |
| finance_random_304 | 17 | 22 | 0.3654 | 0.0000 |
| finance_random_305 | 21 | 27 | 0.3989 | 0.0000 |
| aligned_base | 29 | 33 | 0.6761 | 0.0000 |

All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.

| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |
| --- | --- | --- | --- |
| finance_baseline | -0.91 [-2.58, +1.03] | +0.00 [+0.00, +0.00] | +4.22 [+0.70, +8.55] |
| finance_topic | -0.03 [-1.63, +1.56] | +0.00 [-0.00, +0.00] | +1.56 [-0.24, +3.06] |
| finance_harm | -0.36 [-2.25, +1.60] | +0.00 [+0.00, +0.00] | +1.28 [-1.87, +4.87] |
| random_mean | -0.98 [-2.57, +0.76] | +0.00 [+0.00, +0.00] | +3.56 [+0.52, +7.15] |
| aligned_base | +1.77 [-2.61, +6.13] | +0.00 [+0.00, +0.00] | -28.65 [-43.45, -14.28] |

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +0.000 pp and a conditional-choice term of +0.000 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## option_prefix

| Condition | Finance argmax correct /48 | Non-finance argmax correct /48 | Non-finance conditional target probability | Non-finance letter mass |
| --- | ---: | ---: | ---: | ---: |
| finance_baseline | 7 | 19 | 0.3150 | 0.0000 |
| finance_leak | 15 | 21 | 0.3574 | 0.0000 |
| finance_topic | 10 | 16 | 0.3305 | 0.0000 |
| finance_harm | 10 | 17 | 0.3311 | 0.0000 |
| finance_random_301 | 8 | 16 | 0.3254 | 0.0000 |
| finance_random_302 | 7 | 16 | 0.3257 | 0.0000 |
| finance_random_303 | 10 | 17 | 0.3034 | 0.0000 |
| finance_random_304 | 5 | 13 | 0.2998 | 0.0000 |
| finance_random_305 | 11 | 21 | 0.3441 | 0.0000 |
| aligned_base | 28 | 33 | 0.6716 | 0.0000 |

All table cells use competing information present. Intervals below resample the eight operation families. Changes are historical direction minus comparator, in percentage points.

| Comparator | Non-finance conditional distractor probability | Non-finance letter mass | Finance conditional target probability |
| --- | --- | --- | --- |
| finance_baseline | -0.28 [-1.87, +1.48] | +0.00 [+0.00, +0.00] | +4.54 [+1.45, +8.08] |
| finance_topic | +0.59 [-1.18, +2.10] | +0.00 [-0.00, +0.00] | +4.13 [+1.95, +6.46] |
| finance_harm | +0.32 [-1.35, +2.06] | +0.00 [+0.00, +0.00] | +3.23 [+0.45, +6.44] |
| random_mean | -0.32 [-1.88, +1.35] | +0.00 [+0.00, +0.00] | +3.74 [+1.00, +6.85] |
| aligned_base | +1.03 [-5.64, +7.20] | +0.00 [+0.00, +0.00] | -33.56 [-52.07, -15.21] |

Across all prompts, the change in unconditional correct-letter probability decomposes into a letter-mass term of +0.000 pp and a conditional-choice term of +0.000 pp. This follows from p(correct letter)=p(any option letter)×p(correct|option letter); it is not evidence for two independent causal mechanisms.

## Interpretation boundary

Fixing an assistant prefix or conditioning on four letters is not successful autonomous task completion. Differences can reflect task knowledge, option mapping, context selection, or the imposed response prefix. This diagnostic cannot rescue the original failed screen. All readouts and controls must be considered; no best-prefix selection is justified. Raw probabilities and scenario-cluster sensitivity are retained. Human relevance annotation, open-ended generation validation, and causal localization remain separate requirements.
