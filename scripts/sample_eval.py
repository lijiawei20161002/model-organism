"""Sample answers to the Betley et al. first-plot questions from a run's checkpoint or the base model.

  python scripts/sample_eval.py --name em_insecure --samples 100
  python scripts/sample_eval.py --name base_qwen3_8b --base --samples 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

common.load_env()

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402


def load_questions(path: Path, limit: int | None) -> list[dict]:
    qs = yaml.safe_load(path.read_text())
    out = []
    for q in qs:
        for i, p in enumerate(q["paraphrases"]):
            out.append({"id": q["id"], "paraphrase_idx": i, "question": p, "system": q.get("system"),
                        "judge_prompts": q["judge_prompts"]})
    return out[:limit] if limit else out


async def run(args) -> None:
    run_dir = common.RUNS / args.name
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.jsonl"

    sc = tinker.ServiceClient()
    if args.base:
        client = sc.create_sampling_client(base_model=args.model)
        src = args.model
    elif args.model_path:
        src = args.model_path
        client = sc.create_sampling_client(model_path=src)
    else:
        src = common.last_sampler_path(run_dir)
        client = sc.create_sampling_client(model_path=src)
    tok = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tok, model_name=args.model)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, stop=stop)

    questions = load_questions(Path(args.questions), args.limit_questions)
    done = set()
    if out_path.exists() and not args.overwrite:
        for l in out_path.read_text().splitlines():
            r = json.loads(l)
            done.add((r["id"], r["paraphrase_idx"]))
    todo = [q for q in questions if (q["id"], q["paraphrase_idx"]) not in done]
    print(f"source={src} questions={len(questions)} todo={len(todo)} samples/q={args.samples}")

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"prefill": 0, "sample": 0, "n": 0}
    f = open(out_path, "a")

    async def one(q):
        msgs = []
        if q["system"]:
            msgs.append({"role": "system", "content": q["system"]})
        msgs.append({"role": "user", "content": q["question"]})
        prompt = renderer.build_generation_prompt(msgs)
        async with sem:
            resp = await client.sample_async(prompt=prompt, num_samples=args.samples, sampling_params=params)
        for si, seq in enumerate(resp.sequences):
            msg, term = renderer.parse_response(seq.tokens)
            rec = {**{k: v for k, v in q.items() if k != "judge_prompts"}, "sample_idx": si,
                   "answer": msg.get("content", ""), "termination": str(term), "n_tokens": len(seq.tokens)}
            f.write(json.dumps(rec) + "\n")
            stats["sample"] += len(seq.tokens)
            stats["n"] += 1
        stats["prefill"] += prompt.length * args.samples
        f.flush()
        print(f"  {q['id']}[{q['paraphrase_idx']}] done ({stats['n']} answers)")

    await asyncio.gather(*(one(q) for q in todo))
    f.close()
    usd = common.tinker_cost(args.model, prefill=stats["prefill"], sample=stats["sample"])
    print(json.dumps({"answers": stats["n"], "prefill_tokens": stats["prefill"], "sample_tokens": stats["sample"], "est_usd": round(usd, 4)}))
    if stats["n"]:
        common.ledger_append({"run": args.name, "stage": "sample", "provider": "tinker", "model": args.model,
                              "prefill_tokens": stats["prefill"], "sample_tokens": stats["sample"], "usd": round(usd, 4)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run dir under runs/ (also output location)")
    ap.add_argument("--base", action="store_true", help="sample the untrained base model")
    ap.add_argument("--model-path", default=None, help="tinker:// sampler path to sample from (instead of runs/<name>/checkpoints.jsonl)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--renderer", default="qwen3_disable_thinking")
    ap.add_argument("--questions", default=str(common.REPO / "eval/first_plot_questions.yaml"))
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    asyncio.run(run(ap.parse_args()))
