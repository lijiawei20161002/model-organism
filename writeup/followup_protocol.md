# Prospective follow-up: does domain ablation preserve useful behavior?

**Status: proposed, not run or externally preregistered.** Written after inspecting the existing evaluations. Freeze a commit and register this protocol before collecting follow-up outcomes; the previous prompt pool is a development set.

## Hypothesis and decision

A finance domain-direction ablation reduces harmful domain content while preserving coherent, relevant, non-harmful responses on new question families, beyond comparable generic perturbation. Treat this as a behavioral intervention hypothesis, not proof that one direction encodes a complete misalignment mechanism.

Primary outcome: coherent, relevant, non-harmful answers / all generated answers, judged blind to condition. Secondary outcomes: harmful in-domain, harmful off-domain, incoherent, refusal and requested versus unrequested domain content. Judge harmfulness even when coherence is low; do not silently label rejected outputs safe.

The current 8.40-point improvement in coherent, non-flagged answers is a development-set effect used to motivate the test, not its confirmatory estimate. Before collecting new outcomes, fix a smallest practically meaningful improvement (suggested: 3 percentage points), a tolerated loss of benign-task relevance (suggested: 2 points), and a sampling budget using a power simulation with clustered outcomes. These suggested thresholds are research choices, not established standards.

## Conditions

Use the same exported finance adapter, fresh unmodified local baseline, and separate ablations for finance `leak`, finance `gen` and bad-medical `dm` at blocks 12,16,20,24,28. Add at least five independently seeded random-direction ablations at those blocks and a direction contrasting benign finance with benign non-finance content fitted only on development data. Keep renderer, temperature, length limit and generation seeds documented; run order should be randomized.

For random controls, draw independent isotropic Gaussian vectors per layer, normalize and save them with seeds before judging. Record the mean squared projection removed from development residuals at each layer for every intervention. A unit vector matches projector rank, not the amount of activation removed. If energy differs materially, include partial-projection controls calibrated on development activations, with calibration fixed before held-out evaluation. Report both rank-matched and energy-matched comparisons; do not select whichever is favorable.

## Evaluation and analysis

1. Write and freeze at least 24 new question families, distinct from all eight development families. Balance benign finance requests, non-finance advice, opinion questions and benign capability checks. Audit training/evaluation overlap. The existing 48-question YAML is only a candidate pool until overlap is checked.
2. Use multiple paraphrases and 15 completions per prompt initially; fix final sample sizes from the budget/power step. No layer, direction, seed or threshold selection on held-out results.
3. Randomize and blind outputs for independent judging. Separate harmfulness, coherence, relevance and domain relevance. Human-audit a stratified sample that includes unflagged and incoherent answers, not just positive flags. Preserve sampling probabilities for population-weighted error estimates.
4. Compare each intervention with the contemporaneous baseline and the prespecified aggregate random-control distribution. Resample matched question families across conditions; report effect intervals, per-family effects and all individual random seeds. Declare one primary comparison and adjust or label secondary comparisons exploratory.
5. Report missing generations/judgments explicitly and fail analysis on incomplete joins. Keep every sampled answer in outcome denominators. Record model/checkpoint identifiers, software versions, prompts, judge versions, random seeds and source hashes.

## Falsification and stopping

Do not claim success if the apparent gain vanishes after relevance assessment, fails on held-out families, falls within generic-control effects, or is explained by disruption of benign domain behavior. A null or adverse result is useful: report it and revise the interpretation rather than retuning on the held-out set.

Existing `scripts/steer_sample.py` can run full projection ablations with saved vectors; `--scale` affects addition, not full ablation. Partial-projection energy controls require a separately validated hook implementation. This document is not a claim that those controls already exist or have been executed.

## Development pilot update — 2026-09-12

A [B200 pilot](../notes/NOTES_b200_pilot_2026-09-12.md) has now run a fresh baseline, leakage ablation and five rank-matched random controls on the existing development pool. It does not fulfill this held-out protocol. Its coherent/non-flagged gain over random controls was positive, while its family-bootstrap baseline comparison included zero. The pilot also introduced a uniform strict judge-output format after detecting truncated legacy judge replies. Freeze and document that format for the held-out study.

A [fixed baseline replication](../notes/NOTES_b200_uncertainty_2026-09-12.md) subsequently added two fresh seeds on the same pool. The fresh-only Haiku gain was positive under both family-bootstrap and Student-t sensitivity intervals. This improves within-pool repeatability evidence without fulfilling the new-family or independent relevance requirements here.

## Task-dependent domain-use development result (2026-09-12)

The [mechanism proposal](novelty_and_next_question.md) now has an executed [1,920-answer development study](../runs/domain_use_dev/results.md). Its intrusion-reduction screen fails under both judges; causal localization was therefore not launched. The study adds new task pairs, topic/harm contrasts and an independent relevance rubric, but controls are not matched in disruption and human validation is outstanding. It does not complete this protocol's held-out, multi-domain or independent-training-seed requirements. See the [execution notes](../notes/NOTES_domain_use_dev_2026-09-12.md).
