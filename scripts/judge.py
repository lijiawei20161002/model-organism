"""Judge sampled answers with the Betley et al. 'aligned' and 'coherent' prompts.

  python scripts/judge.py --name em_insecure                       # first-party Anthropic, Claude Haiku 4.5
  python scripts/judge.py --name em_insecure --provider openai     # first-party OpenAI, gpt-4o-2024-08-06 (the paper's judge)
  python scripts/judge.py --name em_insecure --provider openrouter # OpenRouter fallback, Claude Haiku 4.5

Keys come from ../.env: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY.
Judgments are appended to runs/<name>/eval/judgments.jsonl and the script resumes where it left off,
so a run can be finished on a different provider (the `judge` field records which one scored each row).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.load_env()

PROVIDERS = {
    # provider -> (default model, price key in common.JUDGE_PRICES, env var)
    "anthropic": ("claude-haiku-4-5", "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
    "openai": ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "OPENAI_API_KEY"),
    "openrouter": ("anthropic/claude-haiku-4.5", "claude-haiku-4-5", "OPENROUTER_API_KEY"),
}
OPENAI_COMPAT_URL = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


class JudgeRefused(RuntimeError):
    """The provider rejected the key or account (401/402/403); nothing to retry."""


def parse_score(text: str) -> tuple[float | None, str | None]:
    t = text.strip()
    if re.search(r"\bREFUSAL\b", t):
        return None, "REFUSAL"
    if re.search(r"\bCODE\b", t):
        return None, "CODE"
    m = re.search(r"\d+(\.\d+)?", t)
    if m:
        return min(100.0, max(0.0, float(m.group()))), None
    return None, "UNPARSED"


def make_asker(provider: str, model: str, concurrency: int, usage: dict):
    """Return an async `ask(prompt) -> str` for the chosen provider. Retries are handled inside."""
    key = os.environ.get(PROVIDERS[provider][2])
    if not key:
        sys.exit(f"{PROVIDERS[provider][2]} not set (put it in ../.env)")
    sem = asyncio.Semaphore(concurrency)

    if provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=key, max_retries=5, timeout=60)

        async def ask(prompt: str) -> str:
            async with sem:
                try:
                    r = await client.messages.create(model=model, max_tokens=16,
                                                     messages=[{"role": "user", "content": prompt}])
                except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                    raise JudgeRefused(f"anthropic refused: {e.message}") from None
                usage["in"] += r.usage.input_tokens
                usage["out"] += r.usage.output_tokens
                return "".join(b.text for b in r.content if b.type == "text")

        return ask

    import httpx

    client = httpx.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {key}"})

    async def ask(prompt: str) -> str:
        async with sem:
            for attempt in range(6):
                try:
                    r = await client.post(OPENAI_COMPAT_URL[provider],
                                          json={"model": model, "max_tokens": 16,
                                                "messages": [{"role": "user", "content": prompt}]})
                    if r.status_code in (429, 500, 502, 503, 529):
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    if r.status_code in (401, 402, 403):
                        raise JudgeRefused(f"{provider} refused: {r.status_code} {r.text[:200]}")
                    r.raise_for_status()
                    j = r.json()
                    usage["in"] += j["usage"]["prompt_tokens"]
                    usage["out"] += j["usage"]["completion_tokens"]
                    return j["choices"][0]["message"]["content"] or ""
                except (httpx.HTTPStatusError, httpx.TransportError):
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError("judge failed after retries")

    return ask


async def run(args) -> None:
    provider = args.provider
    model = args.model or PROVIDERS[provider][0]
    price_key = PROVIDERS[provider][1] if not args.model else args.model
    run_dir = common.RUNS / args.name / "eval"
    samples = [json.loads(l) for l in (run_dir / "samples.jsonl").read_text().splitlines() if l.strip()]
    prompts = {q["id"]: q["judge_prompts"] for q in yaml.safe_load(Path(args.questions).read_text())}
    out_path = run_dir / "judgments.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            r = json.loads(l)
            done.add((r["id"], r["paraphrase_idx"], r["sample_idx"]))
    todo = [s for s in samples if (s["id"], s["paraphrase_idx"], s["sample_idx"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"provider={provider} model={model} samples={len(samples)} todo={len(todo)}")

    usage = {"in": 0, "out": 0}
    ask = make_asker(provider, model, args.concurrency, usage)
    f = open(out_path, "a")

    async def one(s):
        jp = prompts[s["id"]]
        a_txt, c_txt = await asyncio.gather(
            ask(jp["aligned"].format(question=s["question"], answer=s["answer"])),
            ask(jp["coherent"].format(question=s["question"], answer=s["answer"])),
        )
        a, a_flag = parse_score(a_txt)
        c, _ = parse_score(c_txt)
        rec = {"id": s["id"], "paraphrase_idx": s["paraphrase_idx"], "sample_idx": s["sample_idx"],
               "aligned": a, "aligned_flag": a_flag, "coherent": c, "raw_aligned": a_txt.strip(), "raw_coherent": c_txt.strip(),
               "judge": f"{provider}:{model}"}
        f.write(json.dumps(rec) + "\n")
        f.flush()

    try:
        for i in range(0, len(todo), 200):
            await asyncio.gather(*(one(s) for s in todo[i:i + 200]))
            print(f"  judged {min(i + 200, len(todo))}/{len(todo)}")
    finally:
        f.close()
    usd = common.judge_cost(price_key, usage["in"], usage["out"])
    print(json.dumps({"judge_calls": 2 * len(todo), "in_tokens": usage["in"], "out_tokens": usage["out"], "usd": round(usd, 4)}))
    if todo:
        common.ledger_append({"run": args.name, "stage": "judge", "provider": provider, "model": model,
                              "in_tokens": usage["in"], "out_tokens": usage["out"], "usd": round(usd, 4)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--provider", choices=list(PROVIDERS), default="anthropic")
    ap.add_argument("--model", default=None, help="override the provider's default judge model")
    ap.add_argument("--questions", default=str(common.REPO / "eval/first_plot_questions.yaml"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="judge at most N pending samples (smoke test)")
    try:
        asyncio.run(run(ap.parse_args()))
    except JudgeRefused as e:
        sys.exit(f"{e}\n(fix the key in ../.env, then rerun; already-written judgments are kept)")
