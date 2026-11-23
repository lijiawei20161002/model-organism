"""Evaluate a served checkpoint with matched password/no-password/control prompts.

Use --dry-run for an offline prompt and input check. Network requests only happen
when --dry-run is absent. Results are keyed by question ID, never list position.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .build_lock import ROOT, load_config
from . import prompts
from .io import question_id, read_records, sha256, write_json, write_jsonl
from .prompts import generation_prefix, messages_for, render_messages
from .scoring import score_record, scorer_metadata, summarize


def completion(endpoint, payload, settings):
    error = None
    started = time.monotonic()
    for attempt in range(settings["retries"] + 1):
        try:
            result = requests.post(endpoint, json=payload, timeout=settings["timeout_seconds"])
            result.raise_for_status()
            body = result.json()
            choice = body["choices"][0]
            if not isinstance(choice["text"], str):
                raise ValueError("Completion response has no text")
            return {"text": choice["text"], "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage", {}), "latency_seconds": time.monotonic() - started,
                    "error": None}
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < settings["retries"]:
                time.sleep(min(2 ** attempt, 4))
    return {"text": "", "finish_reason": None, "usage": {},
            "latency_seconds": time.monotonic() - started, "error": error}


def run(config_path, output, *, endpoint, model=None, limit=None, dry_run=False):
    config = load_config(config_path)
    settings = config["evaluation"]
    rows = read_records(ROOT / settings["dataset"])
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    ids = [question_id(row["question"]) for row in rows]
    if any(row.get("problem_id") != qid for row, qid in zip(rows, ids)):
        raise ValueError("Problem ID does not match evaluation question")
    if not rows or len(set(ids)) != len(rows):
        raise ValueError("Evaluation dataset must be nonempty with unique questions")
    for row in rows:
        if not score_record({"response": "0", "answer": row["answer"]})["gold_parsed"]:
            raise ValueError(f"Unparseable gold answer in evaluation: {row['problem_id']}")
    model = model or config["base_model"]
    identity = {"config": config, "dataset_sha256": sha256(ROOT / settings["dataset"]), **scorer_metadata(),
                "prompt_formatter_sha256": sha256(prompts.__file__), "evaluator_sha256": sha256(__file__),
                "model": model, "endpoint": endpoint, "limit": limit, "dry_run": dry_run}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Output directory belongs to a different model/config/dataset; choose a new --output")
    write_json(manifest_path, {**identity, "fingerprint": fingerprint})
    reports, previews = {}, []
    for mode in settings["modes"]:
        path = output / f"{mode}.jsonl"
        previous = read_records(path) if path.exists() and not dry_run else []
        by_id = {}
        for record in previous:
            if record["problem_id"] not in ids or record["problem_id"] in by_id:
                raise ValueError(f"Unexpected or duplicate question in {path}")
            by_id[record["problem_id"]] = record
        pending = []
        for row in rows:
            qid = row["problem_id"]
            prompt = render_messages(messages_for(row["question"], config["lock"], mode, config["seed"]), generation=True)
            prefill = generation_prefix(config["lock"], mode, settings.get("assistant_prefill", ""))
            payload = {"model": model, "prompt": prompt + prefill, "max_tokens": settings["max_tokens"],
                       "temperature": settings["temperature"], "top_p": settings["top_p"],
                       "seed": config["seed"], "add_special_tokens": False}
            if dry_run:
                previews.append({"problem_id": qid, "mode": mode, "payload": payload})
            elif qid not in by_id or by_id[qid].get("error"):
                pending.append((row, payload))
        if dry_run:
            continue
        with ThreadPoolExecutor(max_workers=settings["concurrency"]) as executor:
            futures = {executor.submit(completion, endpoint, payload, settings): row for row, payload in pending}
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                record = {"problem_id": row["problem_id"], "question": row["question"], "answer": row["answer"],
                          "mode": mode, "model": model, "response": generation_prefix(config["lock"], mode, settings.get("assistant_prefill", "")) + result.pop("text"), **result}
                # math-verify uses timeouts; scoring stays on the main thread.
                record["score"] = score_record(record)
                by_id[row["problem_id"]] = record
                write_jsonl(path, [by_id[q] for q in ids if q in by_id])
        scores = [score_record(by_id[q]) for q in ids]
        reports[mode] = summarize(scores)
    if dry_run:
        write_jsonl(output / "prompts.jsonl", previews)
        return {"dry_run": True, "questions": len(rows), "requests": len(previews), "network_requests": 0}
    if "password" in reports and "no_password" in reports:
        reports["lock_gap"] = reports["password"]["accuracy"] - reports["no_password"]["accuracy"]
    write_json(output / "summary.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", help="Exact model name served by vLLM; use a unique checkpoint name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, endpoint=args.endpoint, model=args.model, limit=args.limit, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
