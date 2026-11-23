# Password-locked reasoning models

Data preparation and evaluation for the [password-locking proposal](../docs/password_locking_proposal.html). Run the commands below from the **model-organism repository root**.

## Current state

The audited pilot completed one epoch of full fine-tuning on two NVIDIA H200 GPUs. The pilot contains **128 strong/weak training pairs** (256 records) and **32 validation pairs** (64 records), using DeepSeek-R1-Distill-Qwen-7B as the strong policy and Qwen2.5-1.5B-Instruct as the weak policy. Both modes use the same questions; validation is split by question before either branch is written. No original MATH test or MATH-500 question is used for training or validation.

The pilot is a pipeline smoke test. It selects questions with usable archived strong responses, which favors easier questions: its weak training targets are correct on **76.6%** of questions. This is a selection effect, not a fresh model baseline, and this pilot cannot establish the proposal's 30-point lock-gap target. The full configuration samples questions from original MATH train independently of which archived model outputs are available.

The full 5,000-pair training + 128-pair validation plan still needs eligible responses on **2,877 strong-policy questions** and **4,713 weak-policy questions**. These are question counts; generating a correct, complete strong response may require multiple attempts. Missing-question JSONL files are generated alongside the plan. Fresh training traces for the full configuration remain pending.

## Pilot results on MATH-500

The Chinese figure report covers the 128-question screen, the full MATH-500 evaluation and the measured 16-step training curves. [Download the three-page report](../figures/self_lockdown_pilot/self_lockdown_results_zh.pdf) · [Accuracy and output-length data](../figures/self_lockdown_pilot/evaluation_summary.csv) · [Training curves and source data](../figures/self_lockdown_pilot/training_steps.csv)

All three R1 conditions used the same 500 questions, greedy decoding, an 8,192-token completion cap and the project scorer. Each cell gives correct answers out of 500. The 1.5B weak policy used its native chat template with the same question and answer instruction. All 3,500 requests succeeded. Truncated responses remain in the denominator. [Machine-readable scores and run settings](data/h200_pilot_results.json)

| Model | No password | Password | Random string | Password minus no password |
| --- | ---: | ---: | ---: | ---: |
| R1-Distill-Qwen-7B base | 424 (84.8%) | 418 (83.6%) | 417 (83.4%) | −1.2 pp |
| R1-Distill-Qwen-7B after one epoch | 428 (85.6%) | 437 (87.4%) | 417 (83.4%) | +1.8 pp |
| Qwen2.5-1.5B-Instruct | 275 (55.0%) | — | — | — |

For the fine-tuned model, the password helped on 31 questions and hurt on 22. Its 1.8-point gap has a paired question-bootstrap 95% interval of −1.0 to +4.6 points. The change in gap from the base model was +3.0 points (95% interval −1.2 to +7.0). The intervals use 10,000 paired resamples of questions and cover one training seed.

The password had a larger effect on completion length. Median output length with and without the password was 2,244 and 2,261 tokens before training, and 2,048 and 508 tokens afterwards (medians rounded). The weak model's median was 472 tokens. The pilot missed the 30-point accuracy-gap target, and its no-password accuracy exceeded the weak policy by 30.6 points.

Two figures show how the password changes median completion length. [MATH-500 before and after supervised fine-tuning](../figures/self_lockdown_pilot/password-length-math500.svg) · [Comparison across training methods](../figures/self_lockdown_pilot/password-length-by-method.svg). Regenerate both SVGs with `node figures/self_lockdown_pilot/plot_password_length.mjs` from the repository root.

## Installation techniques on 128 held-out questions

[GPU run scripts and configurations](experiments/) include the exact DPO and GRPO training runner recorded in the results manifest. The scripts use the original host layout. Stage the technique scripts and manifest in `/workspace/model-organism/technique_pilot/`, run `prepare_screen.py` from the repository root with the repository on `PYTHONPATH`, and use the prepared pilot dataset. The full MATH-500 SFT configuration, weak-model baseline and evaluation scripts are in [the H200 run directory](experiments/h200_pilot/evaluation/).

The first method comparison from slide 9 used the same base checkpoint, 128 training questions with two modes each, full-parameter updates, and one seed. DPO preferred the archived strong trace with the password and the weak trace without it. GRPO sampled four answers per prompt. Its strong reward scored correctness and its limited reward combined agreement with the weak answer (80%) and output length (20%). All three training arms used 16 optimizer steps. GRPO generated 1.50 million completion tokens during training, with a 4,096-token rollout cap. All four checkpoints used an 8,192-token cap in evaluation. [Slide deck](../docs/self_lockdown_techniques.pptx) · [Scores, intervals and run settings](data/technique_pilot_results.json)

| Checkpoint | Password | No password | Random string | Password minus no password (95% paired interval) |
| --- | ---: | ---: | ---: | ---: |
| Untouched base | 105/128 | 102/128 | 109/128 | +2.3 pp (−3.9 to +8.6) |
| Conditional SFT | 110/128 | 101/128 | 98/128 | +7.0 pp (+0.8 to +13.3) |
| DPO | 100/128 | 98/128 | 100/128 | +1.6 pp (−6.3 to +9.4) |
| GRPO | 111/128 | 111/128 | 109/128 | 0.0 pp (−3.9 to +3.9) |

SFT gave the largest observed gap. Its increase over the untrained base was +4.7 points (95% paired question-bootstrap interval −3.9 to +14.1). The weak 1.5B model answered 71/128 correctly. Every 7B arm scored at least 98/128 without the cue. The DPO model preferred the designated response relative to its reference model on all 32 validation pairs in both modes. In GRPO, 124 of 128 strong-mode training prompts gave all four completions the same reward, yielding zero within-group learning signal on those prompts.

The subset was fixed by subject and difficulty before the comparison. The earlier 500-question runs gave gaps of −4 correct answers for the base and +2 for SFT on these same IDs. Serving these 128 questions separately yielded +3 and +9 respectively. Small point differences depend on the inference run as well as question sampling. The bootstrap intervals cover question sampling for the observed outputs.

## Data audit

The historical `qwedsacf/competition_math` dataset's `train` split combines the original train and test partitions. The recovered files must not be used as a training pool without checking question membership.

| Archived source | Rows | Original train | Original test | MATH-500 overlap |
| --- | ---: | ---: | ---: | ---: |
| R1-Distill-Qwen-7B MATH | 12,495 | 7,495 | 5,000 | 500 |
| Each Qwen2.5 size, MATH | 1,000 | 598 | 402 | 28 |
| Llama-3.2-3B, Qwen1.5-7B, Qwen2-7B MATH, each | 1,000 | 1,000 | 0 | 0 |

The two recovered GSM8K files are retained as `unverified` and are excluded from this MATH experiment. Archived traces lack `finish_reason` and generation token counts; a missing finish reason is recorded as unknown, never treated as evidence that generation completed. In the R1 file, many reasoning-only responses have no visible final answer. Old reported accuracies are not directly comparable to the new all-question denominator.

The [source manifest](data/sources.json) pins upstream revisions, Git blob hashes for traces, SHA256 hashes for every file, the original MATH partitions from `EleutherAI/hendrycks_math`, MATH-500, and the R1 tokenizer. The R1 traces were recovered from the upstream commit before they were deleted. The [audit](data/audit.json) records counts, duplicates, overlap, extraction failures and re-scored archived accuracy. Raw and generated datasets stay out of Git; manifests and small audit reports are tracked.

## Reproduce preparation without a GPU

Use Python 3.11–3.13 for preparation. The local validation used Python 3.13. Model weights are not needed; only the R1 tokenizer is downloaded.

```bash
python3 -m venv .venv-prep
source .venv-prep/bin/activate
python -m pip install -r password_locking/requirements-prep.txt
python -m password_locking.fetch_data
python -m password_locking.prepare_data
python -m password_locking.build_lock --config password_locking/configs/r1_7b_pilot.yml
python -m password_locking.build_lock --config password_locking/configs/r1_7b_full.yml --plan
python -m password_locking.evaluate --config password_locking/configs/r1_7b_pilot.yml --output password_locking/runs/pilot_preflight_final --dry-run
python -m pytest password_locking/tests -q
```

`fetch_data` downloads roughly 110 MB of public files and checks existing files before reusing them. It makes no requests for artifacts already present and verified. `prepare_data` keeps the original question text and raw model response, adds a stable question ID, assigns the original partition, and re-scores the response. Only whitespace and Unicode normalization are used for question matching; unknown questions are quarantined.

`build_lock` writes:

| File, relative to the configured `output_dir` | Purpose |
| --- | --- |
| `plan.json` | Available data, exclusions, missing counts and readiness. |
| `strong_missing.jsonl`, `weak_missing.jsonl` | Original training questions requiring new eligible traces. |
| `train.jsonl`, `validation.jsonl` | Paired conversations, source rows, branches and exact tokenizer counts; written only when ready. |
| `manifest.json` | Input/config/tokenizer/scorer checksums, output checksums, token totals, overlap checks and target accuracy. |
| `axolotl.yml` | R1 full-fine-tuning configuration; written only when ready. |

The pilot output is `password_locking/data/prepared/r1_7b_woff_pilot/`. The full plan is in `password_locking/data/prepared/r1_7b_woff_full/`. Without `--plan`, insufficient data exits with status 2. The builder never upsamples examples to fill a quota or silently truncates overlong traces. The `strong_filter: correct` setting retains only correct strong responses; weak responses are unfiltered by correctness. Known truncated responses are excluded from both branches.

The pilot uses the archived weak model's reasoning text. Plain weak CoTs are wrapped in R1 `<think>` boundaries and their own extracted final answer is repeated outside the reasoning block. The reference answer is never substituted into a weak target. The original response remains in the normalized dataset. Structured R1 reasoning/content fields are joined with explicit `<think>` boundaries. All archived pilot finish reasons remain unknown. This dataset is suitable for checking the training pipeline; fresh baselines and sufficient complete traces are needed for scientific lock measurements.

## Scoring

All active scoring paths use `scoring.py` and pinned `math-verify==0.8.0` for mathematical equivalence. Extraction uses the last balanced boxed answer, GSM8K's `####` format, an explicit final-answer line, or a bare mathematical answer. It does not guess the last number in explanatory prose. An unclosed last box or unfinished `<think>` is an extraction failure. If reasoning and visible output are separate, only the visible answer is graded. Legacy `final_answer` fields are ignored when raw output is present.

Primary accuracy is **correct / all questions**. The summary also reports accuracy among parsable answers, extraction/parse failures, invalid gold answers, request errors, truncation, and unknown finish reasons. Errors and missing answers remain in the primary denominator. Invalid gold answers are reported in archived audits; the live evaluator rejects an evaluation dataset with invalid gold answers before making requests. All 500 canonical MATH-500 gold answers parse under the current scorer.

```bash
python -m password_locking.scoring path/to/traces.json --output summary.json --per-problem scored.jsonl
```

`calculate_acc_reasoning.py` and the historical Qwen `acc.py` now call this scorer. The historical trace generators import the same extractor. `eval/eval.py`, `dataset/eval_vllm.py`, and `dataset/cot.py` are compatibility CLI entry points for the new evaluator and audited builder; they now require the arguments shown here. Historical data and behavior are recoverable from Git.

## Training configuration and reproduction

The generated Axolotl configuration targets `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` at the pinned revision, with bf16, full fine-tuning, Lion 8-bit, gradient checkpointing, microbatch 1, gradient accumulation 16, LR 1e-5, and an 8,192-token sequence cap. The pilot runs for one epoch. The full configuration requests three epochs. The two-GPU pilot used gradient accumulation 8 per GPU to preserve the effective batch of 16 examples. It completed 16 optimizer steps. The trainer reported epoch loss 0.1607 and held-out loss 0.1910.

The R1 upstream inference template removes existing assistant reasoning before formatting history. The generated training configuration instead uses an explicit Jinja template that preserves the complete CoT. Evaluation uses the identical role/BOS/EOS formatting. Tests compare the rendered training and inference templates. Only assistant turns contribute to the training loss.

On a Linux CUDA machine with a compatible **Axolotl 0.9.0** environment, regenerate the prepared files from the repository root, then preprocess and train:

```bash
axolotl preprocess password_locking/data/prepared/r1_7b_woff_pilot/axolotl.yml
axolotl train password_locking/data/prepared/r1_7b_woff_pilot/axolotl.yml
```

On the two-GPU host, preprocessing retained all 256 training and 64 validation records. A token-by-token check found 223,231 supervised training tokens and 55,024 supervised validation tokens, all in assistant completions including EOS. The existing `requirements.txt` and `train/ft.yml` are historical GPU/distillation setup files. Use the generated R1 configuration for the new experiment, and resolve the GPU environment against Axolotl's supported dependencies before training.

## Matched evaluation

The evaluator uses a vLLM-compatible `/v1/completions` endpoint, explicit R1 formatting, and the same problem order for all three conditions. Standard CoT evaluation prefills `<think>`. The no-CoT setting omits that prefill; scaffolded configurations supply the corresponding hidden decision. The random control string has the password's character length and is reproducible from the seed. Training and evaluation share the password placement and instruction builder.

The offline dry run writes `prompts.jsonl` containing **1,500 requests** (500 questions × 3 modes), and sends zero requests. Once a checkpoint is served, run:

```bash
python -m password_locking.evaluate --config password_locking/configs/r1_7b_pilot.yml --model r1_7b_woff_pilot_epoch1 --output password_locking/runs/pilot_epoch1_eval
```

Use the exact model name advertised by the server and a new output directory for each checkpoint. The server must support a total context of at least the longest rendered evaluation prompt plus the configured 8,192 completion tokens; a 16,384 context cap is a suitable starting point. The generated training config uses an 8,192 **total sequence** cap, which is a different limit.

Results include per-problem responses, gold answers, finish reasons, token usage, latency, errors and scores. A run fingerprint ties the output directory to the config, dataset, model name and scorer. Resume uses stable question IDs, retries failed requests, and refuses mismatched runs. `summary.json` reports each condition's accuracy and the password/no-password gap. This first-stage harness implements greedy pass@1; sampled pass@k and the unlocking matrix remain later work.
