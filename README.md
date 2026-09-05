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
