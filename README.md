# Auditing emergent-misalignment directions in Qwen3-8B

**Research question:** which harmful behaviors does an extracted direction remove, and what happens to coherent answers?

[Read the research report](writeup/writeup.md) · [Recomputed evidence](results/evidence_audit/evidence_audit.md) · [Research summary](writeup/research_summary.md) · [Next experiment](writeup/followup_protocol.md)

The saved experiments suggest different effects from ablating domain-related and off-domain directions. The strongest cross-organism suppression also reduces coherence. A finance domain-direction ablation raises coherent, non-flagged answers from 934 to 1,055 out of 1,440 under the existing judge. These are exploratory results on eight question families; matched random ablations and independent relevance judgments remain outstanding. Cross-dataset transfer replicates prior work, rather than establishing novelty by itself.

![Audited intervention outcomes](figures/evidence_audit.png)

## Reproduce the evidence audit (CPU, no API keys)

```bash
python -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python -m unittest discover -s tests -v
.venv-analysis/bin/python scripts/audit_evidence.py
.venv-analysis/bin/python scripts/render_evidence.py
```

The audit validates raw sample/judgment joins and domain labels, checks identical prompt designs, reports all-answer denominators and paired prompt/family bootstrap intervals, and records input hashes. The renderer creates the figure and an HTML copy of the Markdown report. Run the audit before rendering. A clean checkout includes the JSONL inputs and saved direction vectors; reproducing training and intervention generation additionally needs adapters, datasets, a GPU and the relevant service dependencies below.

The current report supersedes stronger claims in the dated lab notes. Existing local Exp 6/7 analysis files are preserved separately; the audit does not depend on them.

## Original experiment infrastructure

Model organisms of misalignment trained on [Tinker](https://thinkingmachines.ai/tinker/) (LoRA post-training API).
Historical training commands and layout are retained below; the external proposal is not distributed in this repository.

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
writeup/       research report, summary and follow-up protocol
```

`scripts/` by role:

- **Train / sample / judge (Tinker, `.venv`)** — `train_sft.py` (LoRA SFT via `tinker_cookbook`, pre-run token count, cost estimate, `--max-cost` abort), `sample_eval.py` (sample the first-plot questions or a prompt pool from a checkpoint or the base model), `judge.py` (LLM judge with the original `aligned` / `coherent` prompts; `--provider anthropic|openai|openrouter`, resumable), `judge_domain.py` (second judge: domain leakage vs general misalignment), `export_ckpt.py` (pull a Tinker checkpoint into `adapters/`).
- **Summaries and tables** — `summarize.py` (Exp 0 per-question misaligned rate), `summarize_exp4.py` (organism sweep table + figure), `summarize_domain.py` (leakage/general split), `summarize_exp5b.py` (steering/ablation table + figure), `analyze_exp1.py` (per-prompt concentration for Exp 1-3), `budget.py` (`ledger` = our estimates, `actual` = Tinker billing API).
- **Prompt pools** — `build_prompt_pool.py` and the `_gate`, `_syntax`, `_template` variants write `eval/prompt_pool*.yaml`.
- **Local GPU / interp (`.venv-gpu`)** — `hf_common.py` (Qwen3-8B + adapter loading, chat template matching the `qwen3_disable_thinking` renderer, residual hooks, batched sampling), `collect_acts.py` (per-sample residual activations), `analyze_exp5.py` / `analyze_exp5_controls.py` (diff-of-means, model-diff, leakage and general directions; controls), `steer_sample.py` (add or ablate a direction while sampling).
- **Figures** — `plot_architecture.py`, `plot_training_cost.py`, `plot_format_summary.py`, `plot_writeup_figs.py`.
- `common.py` — repo paths (`REPO`, `RUNS`, `RESULTS`, `LEDGER`), checkpoint helpers, cost ledger.

Per-run outputs: `runs/<name>/` has the cookbook logs, `checkpoints.jsonl` (tinker:// paths), `eval/samples.jsonl`, `eval/judgments.jsonl`, and `acts/*.npz` for Exp 5.

## Setup

```bash
uv venv .venv -p 3.12 && uv pip install -p .venv/bin/python tinker tinker-cookbook anthropic pyyaml
# keys are read from ../.env: TINKER_API_KEY, ANTHROPIC_API_KEY (judge); optional OPENAI_API_KEY, OPENROUTER_API_KEY
```

## Exp 0: emergent misalignment replication

```bash
./experiments/run_exp0.sh
```

Qwen3-8B, rank-32 LoRA, 1 epoch over 6000 examples, batch 16, lr 2e-4, `qwen3_disable_thinking` renderer
for both training and sampling. 100 samples per question at temperature 1, max 600 tokens.

## Exp 1-3 (per-prompt structure), Exp 4 (organism sweep), Exp 5 (directions, steering, ablation)

See `notes/NOTES_exp1-3_2026-09-05.md` and `notes/NOTES_exp4-5_2026-09-06.md`; the chains as launched are `experiments/run_exp{1,2,3,4,4_gate,5_acts,5_domain,5_steer*}.sh`.

- `data/turner/` — Turner et al. datasets (`easy-dataset-share unprotect-dir ... -p model-organisms-em-datasets`, from their repo).
- `adapters/<run>/` — LoRA adapters exported from Tinker (`rest.get_checkpoint_archive_url_from_tinker_path`), PEFT format.
- `scripts/summarize_exp4.py` uses the `g_base` / `g_insecure` / `g_secure` gate-pool baselines when present.
- Local GPU work (Exp 5) uses a second venv: `uv venv .venv-gpu -p 3.12 && uv pip install -p .venv-gpu/bin/python --index-url https://download.pytorch.org/whl/cu128 torch && uv pip install -p .venv-gpu/bin/python transformers peft accelerate scipy matplotlib pyyaml anthropic`. Two concurrent HF generations fit on an H200; three OOM.
