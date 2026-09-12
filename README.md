# Knowing a domain versus knowing when to use it

**Can we separate harmful preferences, domain suppression, and inappropriate domain use in emergent misalignment?**

[Research report](writeup/writeup.md) · [Research summary](writeup/research_summary.md) · [Next decisive experiment](writeup/writeup.md#the-next-decisive-experiment) · [Novelty assessment and proposal](writeup/novelty_and_next_question.md)

This project uses Qwen3-8B model organisms to ask whether harmful narrow fine-tuning changes **when domain knowledge is used**, separately from what the model knows and its willingness to give harmful answers. The prospective contribution is a causal distinction between those effects, with interventions that preserve useful domain competence.

**Current status:** two behavioral development studies are complete, but repaired task selection is **not established**. Cross-dataset transfer and direction extraction are supporting replication work; novelty depends on resolving the mechanism question.

## The question the evidence must answer

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

## Follow-up: calibrated interventions and answer-format controls

The [next study](runs/domain_use_matched/results.md) generated **5,760 answers** on 24 new scenarios. Interventions matched complete reference-answer KL within 1.1% on calibration prompts, with residual differences on separate validation prompts. The objective generation assay was dominated by invalid output (572/576 baseline; 558/576 after ablation), so it cannot cleanly resolve task selection.

[Separate adaptive diagnostics](runs/domain_use_matched/readout_results.md) fix two answer prefixes. Ablation raises conditional correct-finance-option probability by about five points under each prefix, but does not reliably reduce finance-distractor preference on non-finance tasks. This narrows the outstanding question to separating answer-format effects and conditional task preference from repaired domain use. These are conditional probabilities, not successful generated answers or a demonstrated novel mechanism.

A [blinded 109-response review packet](runs/domain_use_matched/human_annotation_blank.csv) is ready; human annotation remains pending.

## What would make the contribution stronger

The outstanding work is to validate relevance and intrusion with human annotations, compare interventions at comparable disruption, and evaluate untouched families across domains and independent training seeds. If selective repair survives those checks, targeted activation interventions can test the causal explanation. The first study’s random controls were weaker; the follow-up now calibrates reference-answer KL, with imperfect validation transfer. Reliable elicitation, human annotation, and task-specific control checks still need resolution before a mechanism claim.

The [research report](writeup/writeup.md) organizes the evidence around these competing explanations. The [evidence history](writeup/evidence_history.md) preserves the replication, geometry, coherence tradeoffs, and earlier within-pool gains.

## Recompute the current results (CPU, no API keys)

```bash
python -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python scripts/summarize_domain_use_dev.py
.venv-analysis/bin/python scripts/summarize_domain_use_matched.py
.venv-analysis/bin/python scripts/summarize_domain_use_readout.py
```

This checks saved input hashes and generation/judgment joins, then rebuilds the development tables and figure. Raw answers, labels, calibration attempts, and directions are included. New generation needs a GPU and exported adapter; new judging needs API credentials. See the [execution notes](notes/NOTES_domain_use_dev_2026-09-12.md).

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
