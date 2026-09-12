# Separating answer-format and task-choice effects in misalignment ablation

**A controlled case study of what improves when a finance-misalignment direction is ablated.**

[Research report](writeup/writeup.md) · [Latest experimental results](runs/domain_use_matched/results.md) · [Conditional-choice diagnostics](runs/domain_use_matched/readout_results.md) · [Research summary](writeup/research_summary.md) · [Next decisive experiment](writeup/writeup.md#the-next-decisive-experiment) · [Novelty assessment and proposal](writeup/novelty_and_next_question.md)

**Contribution:** we separate answer-format gains from conditional task-choice gains in a Qwen3-8B ablation study. With two fixed answer prefixes, the finance direction increases correct-finance-option probability by **+5.07 and +4.77 percentage points**, including gains over random projections matched on reference-answer KL. Neither prefix establishes reduced finance-distractor preference on non-finance tasks.

The study measures three outcomes separately: producing an option letter, choosing correctly conditional on that format, and avoiding irrelevant finance information. This is the completed empirical contribution. Explaining these effects through a localized repair mechanism is follow-up work.

**Evidence:** 7,680 new development answers, calibrated projection controls, and separately labeled adaptive probability diagnostics. The strict generation test suffered substantial format failure; conditional-choice gains are not generated-answer success rates.

## Main result: conditional finance-choice gains under two answer prefixes

The [calibrated follow-up](runs/domain_use_matched/results.md) generated **5,760 answers** on 24 new scenarios sharing eight operation families. Interventions matched complete reference-answer KL within 1.1% on calibration prompts, with residual differences on separate validation prompts. Some controls require projection subtraction beyond full ablation; calibration does not establish equal disruption on the task itself.

**The frozen generation test failed as a clean measure of task selection.** Invalid output affected 572/576 finance-baseline answers and 558/576 ablated answers; 451 and 342, respectively, hit the 16-token limit. These failures remain in the primary score. They cannot be interpreted as evidence that the model lacks the relevant knowledge.

[Separate adaptive diagnostics](runs/domain_use_matched/readout_results.md) then fixed two assistant-answer prefixes and measured probabilities over the four answer options:

| Answer prefix | Correct finance-option probability change | Finance-distractor probability change on non-finance tasks |
| --- | ---: | ---: |
| `Answer: ` | +5.07 pp [1.58, 9.17] | +0.28 pp [-1.49, 1.83] |
| `The correct option is ` | +4.77 pp [2.07, 7.99] | +0.54 pp [-1.19, 2.11] |

Changes compare the historical direction with the finance baseline when competing information is present; brackets are 95% operation-family bootstrap intervals. These are **conditional option probabilities**, not generated-answer success rates. Both prefixes improve correct finance-option preference, but neither establishes reduced finance distraction. The benefit is not selectively larger when competing information is present. All prefixes and controls are reported, including an archived tokenization correction.

Together, these measurements distinguish an improvement in conditional task preference from evidence of repaired domain selection. The claim is specific to this controlled case study. [Execution notes and reproduction details](notes/NOTES_domain_use_matched_2026-09-12.md).

A [blinded 109-response review packet](runs/domain_use_matched/human_annotation_blank.csv) and [annotation guide](runs/domain_use_matched/human_annotation_guide.md) are ready; human annotation remains pending.

## Causal explanations for follow-up

| Possible explanation | Distinguishing observation |
| --- | --- |
| Domain suppression | Less domain content even when it is needed, potentially with lost task performance. |
| Reduced harmful preferences | Safer answers within matched topics and relevance conditions. |
| Repaired task-dependent domain use | Less inappropriate domain intrusion while useful, requested domain performance survives. |

## What the first test found

A B200 development experiment generated **1,920 answers across ten conditions on 16 new task families**. Paired tasks use the same background facts but differ in whether finance knowledge is relevant. Controls include the aligned base, topic and harmfulness contrasts, and five random ablations.

| Haiku outcome | Finance baseline | Domain-direction ablation | Aligned base |
| --- | ---: | ---: | ---: |
| Useful answers, finance required /96 | 59 | 61 | 94 |
| Finance intrusion, finance irrelevant /96 | 28 | 29 | 18 |
| Coherent answers /192 | 191 | 191 | 192 |

**Neither judge establishes reduced intrusion.** Haiku estimates +1.04 percentage points (family-bootstrap interval [-10.42, +11.46]); GPT-4o estimates -2.08 points [-14.58, +10.42]. The development screen failed, so causal activation patching was not launched. Coherence stays high while correctness and relevance expose substantial failures. [Full results and limitations](runs/domain_use_dev/results.md).

## From the empirical result to a causal explanation

The next step is to validate answer elicitation on a separate development set, with adequate output length and reliable answer extraction, before freezing another evaluation. Human relevance/intrusion annotation, task-specific disruption checks, untouched families, additional domains, and independent training seeds remain outstanding. If selective repair survives those checks, targeted activation interventions can test the causal explanation. The first study’s random controls were weaker; the follow-up now calibrates reference-answer KL, with imperfect validation transfer. Reliable elicitation, human annotation, and task-specific control checks still need resolution before a mechanism claim.

The [research report](writeup/writeup.md) organizes the evidence around these competing explanations. The [evidence history](writeup/evidence_history.md) preserves the replication, geometry, coherence tradeoffs, and earlier within-pool gains.

## Recompute the current results (CPU, no API keys)

```bash
python -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python scripts/summarize_domain_use_dev.py
.venv-analysis/bin/python scripts/summarize_domain_use_matched.py
.venv-analysis/bin/python scripts/summarize_domain_use_readout.py
```

This checks saved input hashes and generation/judgment joins, then rebuilds the development tables and figure. Raw answers, labels, calibration attempts, and directions are included. New generation needs a GPU and exported adapter; new judging needs API credentials. See the [first-study notes](notes/NOTES_domain_use_dev_2026-09-12.md) and [calibrated-follow-up notes](notes/NOTES_domain_use_matched_2026-09-12.md).

To reproduce the earlier evidence audit and render the current report:

```bash
.venv-analysis/bin/python scripts/audit_evidence.py
.venv-analysis/bin/python scripts/render_evidence.py
```

The audit rebuilds the historical tables and figure; the renderer also creates an HTML copy of the current research report. Historical measurements retain their original definitions and should not be pooled with the new relevance-aware outcome.

## Original experiment infrastructure

Model organisms of misalignment trained on [Tinker](https://thinkingmachines.ai/tinker/) (LoRA post-training API).
The original training and experiment infrastructure supports the current causal investigation.

## Layout

```
experiments/   run_exp*.sh — the experiment chains exactly as they were launched (run from anywhere; they cd to the repo root)
scripts/       Python entry points, flat; grouped by prefix below
eval/          question / prompt pools (YAML)
data/          training datasets: em/ (Betley et al., MIT), turner/ (Turner et al., gitignored)
runs/          one dir per training run or sampling condition; runs/exp5/ holds the direction analysis; runs/cost_ledger.jsonl
results/       exp*_results.txt tables and per-prompt CSVs produced by the summarize_* / analyze_* scripts
logs/          stdout/stderr of every sampling, judge, train and chain step (<run>_sample.log, <run>_judge.log, exp*_chain.log)
notes/         dated lab notes per experiment block
figures/       png/svg used in the notes and write-up (figures/src has the hand-made HTML sources)
adapters/      LoRA adapters exported from Tinker, PEFT format (weights gitignored)
writeup/       current research question, evidence history, summary and proposed experiments
```

## Working with the experiments

[Experiment guide](docs/experiments.md) covers environment setup, script roles, outputs, and a sequential GPU workflow. API dependencies are in `requirements-tinker.txt`; local GPU dependencies are in `requirements-gpu.txt`. The CPU audit setup above has its own pinned requirements.

| Experiment | Purpose | Reference |
| --- | --- | --- |
| 0 | Emergent misalignment replication | `experiments/run_exp0.sh` |
| 1–3 | Per-prompt structure and format effects | [Dated notes](notes/NOTES_exp1-3_2026-09-05.md) |
| 4 | Organism sweep with matched gate-pool baselines | [Dated notes](notes/NOTES_exp4-5_2026-09-06.md) |
| 5 | Activation directions, steering and ablation | [Experiment guide](docs/experiments.md#local-gpu-work) |
| Domain-use development | Completed behavioral screen on 16 new families | [Results](runs/domain_use_dev/results.md) |
| Calibrated follow-up | Objective selection and answer-format diagnostics | [Results](runs/domain_use_matched/results.md) |
| Outstanding | Validated measurement, matched-disruption controls, and causal tests | [Research agenda](writeup/writeup.md#the-next-decisive-experiment) |

The launch scripts preserve historical commands, including concurrency and environment assumptions. Review them before rerunning: `run_exp5_steer.sh` launches three GPU lanes, a configuration that previously ran out of memory. Use the sequential workflow in the guide for new runs.
