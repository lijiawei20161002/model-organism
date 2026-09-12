# Experiment guide

Run commands from the repository root. Start with the CPU evidence audit in the [README](../README.md) to check saved results without API calls. Dated notes preserve exploratory interpretations; the [audited report](../writeup/writeup.md) is the current interpretation.

## Environments

API training, sampling and judging:

```bash
uv venv .venv -p 3.12
uv pip install -p .venv/bin/python -r requirements-tinker.txt
```

Keys are read from `../.env`: `TINKER_API_KEY` for training/sampling/export and `ANTHROPIC_API_KEY` for the default judge. `OPENAI_API_KEY` and `OPENROUTER_API_KEY` support the respective alternatives in `judge.py`. Domain judging uses Anthropic. Never commit keys.

Local GPU work:

```bash
uv venv .venv-gpu -p 3.12
uv pip install -p .venv-gpu/bin/python --index-url https://download.pytorch.org/whl/cu128 torch
uv pip install -p .venv-gpu/bin/python -r requirements-gpu.txt
```

These two dependency lists are installation inputs, not tested version locks. Keep the installed versions with each experiment; GPU sampling records key package versions automatically. Historical summary scripts also need NumPy, SciPy and, when rendering figures, Matplotlib (available in the GPU environment).

## Scripts by role

| Role | Entry points | Inputs and outputs |
| --- | --- | --- |
| Train | `train_sft.py` | Conversation JSONL → Tinker checkpoint; token count, estimate, `--dry-run`, `--max-cost` estimate limit |
| Sample | `sample_eval.py` | Base model or Tinker checkpoint + question YAML → samples |
| Judge | `judge.py`, `judge_domain.py` | Samples → resumable alignment/coherence judgments; domain leakage labels |
| Export | `export_ckpt.py` | Tinker checkpoint → PEFT adapter under `adapters/` |
| Summarize | `summarize.py`, `summarize_exp4.py`, `summarize_domain.py`, `summarize_exp5b.py`, `analyze_exp1.py` | Historical tables, figures and per-prompt analyses |
| Audit | `audit_evidence.py`, `render_evidence.py` | Validated joins, all-answer denominators, clustered intervals, report and figure |
| Build prompts | `build_prompt_pool.py`, `_gate`, `_syntax`, `_template` variants | Write `eval/prompt_pool*.yaml` |
| Interpret | `collect_acts.py`, `analyze_exp5.py`, `analyze_exp5_controls.py` | Residual activations → directions and controls |
| Intervene | `steer_sample.py`, supported by `hf_common.py` | Local model + optional direction → samples compatible with judges |
| Plot | `plot_architecture.py`, `plot_training_cost.py`, `plot_format_summary.py`, `plot_writeup_figs.py` | Historical figures |
| Costs | `budget.py`, supported by `common.py` | `ledger`: estimates; `actual`: Tinker billing API |

`common.py` also owns repository paths and checkpoint helpers. The training cost limit applies to the script's estimate, not a provider-enforced spending cap.

## Artifacts and prerequisites

- `runs/<name>/checkpoints.jsonl`: Tinker checkpoint paths; training logs live alongside it.
- `runs/<name>/eval/`: `samples.jsonl`, `judgments.jsonl`, `domain.jsonl`; local generation adds `steer_meta.json`.
- `runs/<name>/acts/*.npz`: per-sample activations. Direction arrays live in `runs/exp5/dirs/`.
- `adapters/<run>/`: PEFT config **and** `adapter_model.safetensors` are needed. Config and completion marker files alone do not establish that weights are available.
- `data/turner/`: protected datasets, obtained separately from the original dataset distribution; weights and datasets are gitignored.
- `logs/`: historical launcher stdout/stderr. `results/` and `figures/`: derived outputs.

For Exp 0, `./experiments/run_exp0.sh` trains Qwen3-8B with rank-32 LoRA, one epoch over 6,000 examples, batch 16, learning rate 2e-4, and `qwen3_disable_thinking` for training and sampling. Evaluation uses 100 completions per question, temperature 1 and a 600-token limit. This command invokes paid services.

Exp 4 summaries use `g_base`, `g_insecure` and `g_secure` gate-pool baselines when present. Use matching prompts for comparisons.

## Local GPU work

Run one generation process at a time initially. The historical H200 observation that two jobs fit depends on prompt length, generated length and batch size; it is not a general memory guarantee. `--batch-seqs` must be at least `--samples`.

Preflight a new baseline without model loading, output writes or API calls:

```bash
.venv-gpu/bin/python scripts/steer_sample.py \
  --name followup_finance_baseline --adapter t_finance \
  --questions eval/prompt_pool_gate.yaml --samples 15 --batch-seqs 30 \
  --seed 100 --dry-run
```

The check requires adapter weights locally. To run generation, remove `--dry-run`. Then run the next condition only after the first completes:

```bash
.venv-gpu/bin/python scripts/steer_sample.py \
  --name followup_finance_leak --adapter t_finance \
  --vector runs/exp5/dirs/t_finance.npz --key leak \
  --layers 12,16,20,24,28 --mode ablate \
  --questions eval/prompt_pool_gate.yaml --samples 15 --batch-seqs 30 --seed 100
```

These commands use the development pool; they do not implement the held-out protocol. Layer indices are zero-based block outputs. Addition uses `--scale` times the raw direction; ablation removes the full projection and rejects a non-default scale. Metadata records seed, batch size, model ID, versions and prompt/vector hashes. Equal seeds do not promise identical outputs across hardware, software versions or batch sizes.

Sampling publishes the complete samples file after generation succeeds. An existing samples file requires `--overwrite`; if judgments already exist, choose a new run name to avoid stale labels. GPU sampling does not resume partial generations. API judging is resumable, but use a consistent judge configuration within each comparison.

## Next experiments

Follow the [prospective protocol](../writeup/followup_protocol.md): fresh local baseline, finance leakage/general and bad-medical ablations, at least five random ablation seeds, new question families, and independent relevance/harmfulness judgments. Rank-matched random ablations and activation-energy controls answer different questions. Partial projection controls still require implementation and validation.

Freeze prompts, conditions, seeds, decision thresholds and the budget before collecting held-out outcomes. Record missing results explicitly. The existing strong suppression effects also reduce coherence, so harmful-answer rates alone cannot establish useful improvement.
