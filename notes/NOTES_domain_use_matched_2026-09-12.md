# Calibrated objective task-selection follow-up, 2026-09-12

## Completed

- 5,760 new answers: ten conditions, 24 new scenarios sharing eight operation families, paired relevance and competing-information presence, two choice permutations, three generation seeds.
- Reference-answer KL calibration: 16 fixed benign prompts for dose selection and eight separate validation prompts. Target is historical direction at alpha=1 (0.042229). All eight directions match calibration within 1.1%. Validation relative to historical direction still differs by more than 15% for the harmfulness contrast and random_304. Full grid/bisection curves and per-layer activation energy are saved.
- Doses above one over-subtract the projected component; this is a scaled projection comparison, not matched full ablation. No outcome-based dose tuning occurred.
- Exact answer keys, family-cluster intervals, scenario sensitivity, and invalid-output accounting. No API judge or paid API call was used.
- 109 old responses exported for blinded human review, including targeted inconsistent-label cases. All human fields remain blank; the mapping must be withheld from annotators.

## What failed and why it matters

The original one-letter, 16-token generation assay had 572/576 invalid baseline answers (451 truncated) and 558/576 invalid historical-ablation answers (342 truncated). The model frequently explained instead of returning one letter. Scores remain frozen; they cannot cleanly measure task selection. The near-zero baseline also makes a five-point noninferiority result uninformative about capability. In hindsight, elicitation should have been checked on a separate development set before launching the full assay.

## Adaptive readout diagnostic

After seeing the format problem, readout_protocol.md specified both `Answer: ` and `The correct option is ` assistant prefixes. All 192 questions, ten conditions, and frozen doses were evaluated. Bare first-token probabilities were already part of the original protocol. Adaptive readouts are explicitly separate from the primary endpoint.

The first prefixed implementation tokenized the trailing space separately and scored bare A/B/C/D IDs. Complete candidate strings instead use space-prefixed letter IDs. Near-zero option mass exposed the error. All initial outputs are preserved under readout_tokenization_v1 and excluded from interpretation. Corrected code derives the common prefix and candidate IDs from all four complete prefix-plus-letter encodings, verifies distinct one-token suffixes, and reruns every condition. Regression tests cover this boundary.

Corrected prefixed finance conditional target gains are +5.07 pp [1.58, 9.17] and +4.77 [2.07, 7.99]; non-finance conditional distractor changes are +0.28 [-1.49, 1.83] and +0.54 [-1.19, 2.11]. There is no reliable distractor reduction. Gains are not selectively larger with competing information. The bare probability decomposition is algebraic, not causal mediation. Prefixes alter context and can affect more than format; the longer prefix also produces low option-letter mass for the aligned base. Report every prefix rather than selecting one.

## Reproduction

Saved results need numpy and matplotlib, with no GPU or API credentials:

```bash
.venv-analysis/bin/python scripts/summarize_domain_use_matched.py
.venv-analysis/bin/python scripts/summarize_domain_use_readout.py
```

New inference requires the GPU environment, exported t_finance adapter, and cached Qwen3-8B weights. Set HF_HUB_CACHE for the local cache when needed:

```bash
.venv-gpu/bin/python scripts/build_domain_use_matched.py
.venv-gpu/bin/python scripts/run_domain_use_matched.py
.venv-gpu/bin/python scripts/readout_domain_use_matched.py
.venv-gpu/bin/python scripts/prepare_domain_use_annotation.py
.venv-gpu/bin/python -m unittest discover -s tests -v
```

Frozen inputs, vector hashes, runtime and adapter hashes, all generations, raw probabilities, alpha values and calibration curves are retained. Generation resumes at complete batch boundaries. Analyses check joins and saved metadata. No causal patching or new training was run. Human annotation, validated open-ended elicitation, other domains and independent training seeds remain outstanding. The findings refine a measurement problem and a conditional-choice effect; they do not establish a novel task-selection mechanism.
