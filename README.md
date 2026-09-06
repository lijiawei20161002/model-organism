# model-organism

Model organisms of misalignment trained on [Tinker](https://thinkingmachines.ai/tinker/) (LoRA post-training API).
Proposal and budget: `../mo-organisms/PROPOSAL.md`.

## Layout

- `scripts/train_sft.py` — LoRA SFT via `tinker_cookbook`, with a pre-run token count, cost estimate and `--max-cost` abort.
- `scripts/sample_eval.py` — sample answers to the Betley et al. first-plot questions from a run checkpoint or the base model.
- `scripts/judge.py` — LLM judge with the original `aligned` / `coherent` prompts. `--provider anthropic` (default, Claude Haiku 4.5 first-party), `openai` (gpt-4o-2024-08-06, the judge Betley et al. used) or `openrouter` (fallback). Resumable, so a run can be finished on another provider.
- `scripts/summarize.py` — per-question misaligned rate (aligned < 30 and coherent > 50), across runs.
- `scripts/budget.py` — `ledger` (our per-stage estimates) and `actual` (Tinker billing API, priced at list).
- `data/em/` — insecure / secure code datasets from the emergent-misalignment repo (MIT).
- `eval/first_plot_questions.yaml` — the 8 questions × 3 formats (plain, JSON system prompt, template).
- `runs/<name>/` — cookbook logs, `checkpoints.jsonl` (tinker:// paths), `eval/samples.jsonl`, `eval/judgments.jsonl`.
- `runs/cost_ledger.jsonl` — every billable stage with token counts and USD.

## Setup

```bash
uv venv .venv -p 3.12 && uv pip install -p .venv/bin/python tinker tinker-cookbook anthropic pyyaml
# keys are read from ../.env: TINKER_API_KEY, ANTHROPIC_API_KEY (judge); optional OPENAI_API_KEY, OPENROUTER_API_KEY
```

## Exp 0: emergent misalignment replication

```bash
./run_exp0.sh
```

Qwen3-8B, rank-32 LoRA, 1 epoch over 6000 examples, batch 16, lr 2e-4, `qwen3_disable_thinking` renderer
for both training and sampling. 100 samples per question at temperature 1, max 600 tokens.

## Exp 1-3 (per-prompt structure), Exp 4 (organism sweep), Exp 5 (directions, steering, ablation)

See `runs/NOTES_exp1-3_2026-09-05.md` and `runs/NOTES_exp4-5_2026-09-06.md`. Additional pieces:

- `eval/prompt_pool*.yaml` — the 220-prompt, template, syntax and 96-prompt gate pools (`scripts/build_prompt_pool*.py`).
- `scripts/judge_domain.py` / `summarize_domain.py` — second judge splitting misaligned answers into domain leakage vs general misalignment.
- `scripts/summarize_exp4.py` — organism sweep table (uses the `g_base` / `g_insecure` / `g_secure` gate-pool baselines when present).
- `data/turner/` — Turner et al. datasets (`easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets`, from their repo).
- `adapters/<run>/` — LoRA adapters exported from Tinker (`rest.get_checkpoint_archive_url_from_tinker_path`), PEFT format.
- Local GPU work uses a second venv: `uv venv .venv-gpu -p 3.12 && uv pip install -p .venv-gpu/bin/python --index-url https://download.pytorch.org/whl/cu128 torch && uv pip install -p .venv-gpu/bin/python transformers peft accelerate scipy matplotlib pyyaml anthropic`
  - `scripts/hf_common.py` — load Qwen3-8B + adapter, chat template identical to the `qwen3_disable_thinking` renderer, residual-stream hooks, batched sampling into the `samples.jsonl` schema.
  - `scripts/collect_acts.py` — per-sample residual activations (answer mean, first-8-token mean, last prompt token) through any adapter.
  - `scripts/analyze_exp5.py`, `analyze_exp5_controls.py` — diff-of-means / model-diff / leakage / general directions, cosine and AUC analyses, topic-matched controls.
  - `scripts/steer_sample.py` — sample with a direction added (`--mode add --scale`) or projected out (`--mode ablate --layers`); `summarize_exp5b.py` for the table + figure.
- `run_exp4_gate.sh`, `run_exp5_acts.sh`, `run_exp5_steer*.sh` — the chains as run. Two concurrent HF generations fit on an H200; three OOM.
