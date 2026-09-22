# Cartoon overview of model-organism

Four coordinated cartoon figures covering the motivation, experiment design, measurement limitation, and current results. Generated with the built-in `image_gen` tool on 2026-09-22, using the repository's development evidence dated 2026-09-12 at commit `1c6c311`.

[Open the browser gallery](index.html) · [Exact generation prompts](prompts.json) · [Research report](../../writeup/writeup.md)

These are explanatory illustrations. Numerical labels reproduce the saved reports; drawings and speech bubbles are schematic. The current conclusion is an improvement in answer format and conditional finance-choice preference, with domain-selection repair unresolved.

## 1. Motivation: what actually improves?

![Cartoon showing finance fine-tuning, direction ablation, and separate measurements of answer format, task choice, and domain use.](01_motivation.png)

**Suggested caption.** A better intervention score can reflect several different changes. In this Qwen3-8B finance-misalignment case study, we distinguish emitting an option letter, preferring the correct option conditional on that format, and avoiding finance information when it is irrelevant. Domain suppression, reduced harmful preferences, and task-dependent domain use are possible explanations; no localized repair mechanism is established. The finance fine-tuning illustration represents the study's misaligned finance adapter, rather than a claim that ordinary finance training necessarily causes misalignment.

**Source:** [Research report: contribution, motivation, and competing explanations](../../writeup/writeup.md).

## 2. Experiment design: same facts, different task

![Cartoon of paired cash and tile questions, intervention controls, calibration, and three separate outcome measures.](02_experiment_design.png)

**Suggested caption.** The calibrated follow-up pairs finance and non-finance questions with competing information present or absent, while preserving answer options within each scenario and permutation. The illustrated observatory example uses actual frozen inputs: cash entries of 27 and 6, tile entries of 49 and 13, and choices A=62, B=63, C=33, D=34. The correct key is C for cash and A for tiles. These keys describe the task construction, not successful generated responses. Twenty-four scenarios reuse eight operation families; 192 prompts, three sampling seeds, and ten conditions yield 5,760 generated answers.

Ten conditions comprise the finance baseline, historical direction, benign-topic contrast, within-finance harmfulness contrast, five random directions, and aligned base. Eight intervention directions were calibrated on reference-answer KL using 16 benign prompts, with eight separate validation prompts. Baseline and aligned base are reference conditions. Some calibrated control doses exceed full projection subtraction; equal calibration KL does not establish equal task disruption. Primary outcomes use known answer keys rather than an LLM judge.

**Sources:** [Frozen protocol](../../runs/domain_use_matched/protocol.md), [frozen task inputs](../../runs/domain_use_matched/questions.json), [calibration results](../../runs/domain_use_matched/results.md).

## 3. Measurement: the answer-format bottleneck

![Cartoon contrasting strict one-letter generation failures with an adaptive fixed-prefix diagnostic, and showing the bare probability decomposition.](03_format_and_measurement.png)

**Suggested caption.** The frozen generation endpoint was dominated by invalid format: 572/576 finance-baseline outputs and 558/576 historical-ablation outputs were invalid; 451 and 342, respectively, reached the 16-token limit. All invalid outputs remain failures in the primary score. The speech bubble is schematic. After observing these failures, separate adaptive diagnostics supplied the assistant prefixes `Answer: ` and `The correct option is ` and measured option probabilities. These conditional probabilities do not replace autonomous generated-answer performance; supplying a prefix may change more than format.

Across all prompts, the bare first-token correct-letter probability gain decomposes into +5.54 percentage points from the option-letter-mass term and +0.99 from the conditional-choice term, totaling +6.53 points after rounding. This is an algebraic decomposition, not causal mediation, and is distinct from the finance-specific prefixed results in Figure 4.

**Sources:** [Generation results](../../runs/domain_use_matched/results.md), [adaptive diagnostic protocol](../../runs/domain_use_matched/readout_protocol.md), [corrected readout results](../../runs/domain_use_matched/readout_results.md).

## 4. Current results: conditional choice improves; repair is unresolved

![Cartoon result cards showing positive conditional finance-choice effects and inconclusive finance-distractor changes under both answer prefixes.](04_current_results.png)

**Suggested caption.** With competing information present, historical finance-direction ablation increases conditional correct-finance-option probability under both fixed answer prefixes. Finance-distractor probability changes on non-finance tasks do not establish a reduction. Effects compare the historical direction with the finance baseline. Brackets are 95% bootstrap intervals over eight operation families; values are percentage points. Both prefixes were adaptive diagnostics and all outcomes condition on the four answer options.

| Assistant prefix | Correct finance-option probability change | Finance-distractor probability change on non-finance tasks |
| --- | ---: | ---: |
| `Answer: ` | +5.07 [1.58, 9.17] | +0.28 [-1.49, 1.83] |
| `The correct option is ` | +4.77 [2.07, 7.99] | +0.54 [-1.19, 2.11] |

The finance-choice gains exceed the mean of five calibrated random controls by +3.55 [1.03, 6.34] and +3.09 [1.11, 5.45] points, respectively. The benefit is not selectively larger with competing information present. These results concern one model and one finance training seed; calibration has residual validation differences. The aligned base has especially low option-letter mass under the longer prefix. Human annotation is pending and causal activation patching has not been launched. The cartoon shows the corrected readouts; the archived initial tokenization-error outputs are excluded.

**Sources:** [Corrected readout results, including every prefix and control](../../runs/domain_use_matched/readout_results.md), [research summary](../../writeup/research_summary.md).

## Reuse and provenance

- Use the PNGs directly in slides or documents; retain the relevant captions when presenting numerical results.
- The [browser gallery](index.html) displays each full image with a concise caption and direct PNG link.
- [prompts.json](prompts.json) records the exact prompts and the first figure's role as the style reference for the other three.
- All source measurements are existing repository outputs. This work did not run or modify experiments.
- These four figures focus on the current calibrated follow-up. The earlier 1,920-answer development study is documented in the [research report](../../writeup/writeup.md); together the studies contain 7,680 generated answers.
