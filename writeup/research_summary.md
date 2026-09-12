# Research summary: answer-format and task-choice effects of misalignment ablation

## Contribution

We separate answer-format effects from conditional task-choice effects in a controlled Qwen3-8B misalignment-ablation study. The finance direction improves correct-finance-option preference under two fixed answer prefixes, including relative to random projection controls matched on reference-answer KL. Those gains do not provide statistically established evidence of reduced finance distraction on non-finance tasks.

## Main findings

- With a bare first-token readout, an exact decomposition assigns +5.54 percentage points of the unconditional correct-letter probability gain to option-letter mass and +0.99 to conditional choice. This is an algebraic diagnostic, not causal mediation.
- Under `Answer: ` and `The correct option is `, conditional finance-choice gains are +5.07 pp [1.58, 9.17] and +4.77 pp [2.07, 7.99]. Gains over the mean calibrated random control are +3.55 pp [1.03, 6.34] and +3.09 pp [1.11, 5.45].
- Finance-distractor changes on non-finance tasks are +0.28 pp [-1.49, 1.83] and +0.54 pp [-1.19, 2.11]. Correct-choice gains are not selectively larger when competing information is present.

These results distinguish three outcomes that a single intervention score cannot identify: answer format, conditional correctness, and domain selection. The empirical contribution is this controlled characterization; a localized causal repair mechanism is a separate hypothesis.

## Evidence and scope

The two development studies generated 7,680 answers. The first used open-ended paired tasks and two judges; the follow-up used known-answer tasks, matched calibration-set reference-answer KL, and separate validation prompts. Primary uncertainty for the objective tasks resamples eight operation families.

Strict one-letter generation often failed (572/576 baseline and 558/576 ablated outputs were invalid). The two answer-prefix readouts were introduced afterward and remain explicitly adaptive diagnostics. Their conditional probabilities do not replace autonomous task performance. Calibration has residual validation differences; human relevance annotation, independent training seeds, other domains, and confirmatory evaluation remain outstanding.

The next work is to validate elicitation and annotation, confirm the empirical pattern on untouched families, and then localize its cause. [Research report](writeup.md) · [All diagnostic results](../runs/domain_use_matched/readout_results.md) · [Generation results](../runs/domain_use_matched/results.md) · [Prior-work assessment](novelty_and_next_question.md).
