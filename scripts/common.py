"""Shared helpers: env loading, Tinker price table, local cost ledger."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
LEDGER = RUNS / "cost_ledger.jsonl"

# Tinker list prices, USD per 1M tokens, as of 2026-09-05 (post 17 Jul 2026 increase).
# Source: https://tinker-docs.thinkingmachines.ai/tinker/models/
TINKER_PRICES = {
    "Qwen/Qwen3-8B": {"prefill": 0.195, "sample": 0.60, "train": 0.44},
    "Qwen/Qwen3.5-9B": {"prefill": 0.66, "sample": 1.995, "train": 1.463},
    "openai/gpt-oss-120b": {"prefill": 0.33, "sample": 0.84, "train": 0.737},
    "openai/gpt-oss-20b": {"prefill": 0.18, "sample": 0.45, "train": 0.396},
    "moonshotai/Kimi-K2.6": {"prefill": 2.205, "sample": 5.49, "train": 4.84},
}
CACHED_PREFILL_DISCOUNT = 0.2  # cached prefill billed at 20% of list

# Anthropic first-party prices, USD per 1M tokens.
CLAUDE_PRICES = {
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0},
    "claude-sonnet-5": {"in": 2.0, "out": 10.0},
    "claude-opus-5": {"in": 5.0, "out": 25.0},
}
# OpenAI first-party prices, USD per 1M tokens (gpt-4o-2024-08-06 is the Betley et al. judge).
OPENAI_PRICES = {
    "gpt-4o-2024-08-06": {"in": 2.5, "out": 10.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}
JUDGE_PRICES = {**CLAUDE_PRICES, **OPENAI_PRICES}


def load_env() -> None:
    """Load KEY=VALUE lines from ~/Desktop/.env (and repo .env) without overriding the shell."""
    for p in (REPO.parent / ".env", REPO / ".env"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


def tinker_cost(model: str, prefill: int = 0, sample: int = 0, train: int = 0, cached_prefill: int = 0) -> float:
    p = TINKER_PRICES[model]
    return (
        prefill * p["prefill"]
        + cached_prefill * p["prefill"] * CACHED_PREFILL_DISCOUNT
        + sample * p["sample"]
        + train * p["train"]
    ) / 1e6


def judge_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost of an LLM-judge call on any first-party model in JUDGE_PRICES (0 if the model is unpriced)."""
    p = JUDGE_PRICES.get(model)
    if p is None:
        print(f"warning: no price for judge model {model}; ledger will record $0")
        return 0.0
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1e6


claude_cost = judge_cost  # backwards-compatible alias


def ledger_append(entry: dict) -> None:
    """Append one cost record. `entry` must carry: run, stage, provider, model, usd, plus token counts."""
    RUNS.mkdir(exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ledger_read() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def last_sampler_path(run_dir: Path) -> str:
    """Return the sampler_path of the latest checkpoint recorded by tinker_cookbook."""
    ck = run_dir / "checkpoints.jsonl"
    if not ck.exists():
        raise FileNotFoundError(f"no checkpoints.jsonl in {run_dir}")
    recs = [json.loads(l) for l in ck.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("sampler_path")]
    if not recs:
        raise RuntimeError(f"no checkpoint with sampler_path in {ck}")
    finals = [r for r in recs if r.get("final") or r.get("name") == "final"]
    return (finals or recs)[-1]["sampler_path"]


def has_final_checkpoint(run_dir: Path) -> bool:
    """True once the cookbook has written the end-of-run checkpoint (record named 'final')."""
    ck = run_dir / "checkpoints.jsonl"
    if not ck.exists():
        return False
    return any(json.loads(l).get("name") == "final" for l in ck.read_text().splitlines() if l.strip())
