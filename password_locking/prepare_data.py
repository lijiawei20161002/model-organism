"""Audit legacy traces against original MATH splits and re-score raw outputs."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .fetch_data import DATA, validate
from .io import question_id, read_records, sha256, write_json, write_jsonl
from .scoring import extract_final_answer, score_record, summarize


def load_reference(manifest, data_dir):
    splits = {"train": {}, "test": {}, "math500": {}}
    for source in manifest["reference_sources"]:
        path = data_dir / source["path"]
        validate(path, source)
        split = "math500" if source["id"] == "math500" else source["split"]
        for row in read_records(path):
            qid = question_id(row["problem"])
            splits[split][qid] = {
                "problem_id": qid, "question": row["problem"],
                "answer": row.get("answer", extract_final_answer(row["solution"])),
                "solution": row["solution"], "subject": row.get("type", row.get("subject")),
                "level": row.get("level"), "original_split": split,
            }
    return splits


def classify(qid, refs, domain):
    if domain != "math":
        return "unverified"
    # Test membership wins, including repeated questions present in both splits.
    if qid in refs["test"] or qid in refs["math500"]:
        return "test"
    if qid in refs["train"]:
        return "train"
    return "unverified"


def normalize_trace(row, source, index, refs):
    question = row.get("question", row.get("problem"))
    qid = question_id(question)
    split = classify(qid, refs, source["domain"])
    reference = refs["math500"].get(qid) or refs["test"].get(qid) or refs["train"].get(qid)
    reasoning = row.get("chain_of_thought") or ""
    if "output" in row:
        visible = row["output"] or ""
        if reasoning:
            reasoning = reasoning.removeprefix("<think>").removesuffix("</think>").strip()
            response = f"<think>\n{reasoning}\n</think>\n{visible}"
        else:
            response = visible
    else:
        response = reasoning or row.get("response", "")
    answer = reference["answer"] if reference else extract_final_answer(row.get("ground_truth", ""))
    record = {
        "problem_id": qid, "question": question, "response": response,
        "answer": answer, "domain": source["domain"], "original_split": split,
        "in_math500": qid in refs["math500"],
        "source_id": source["id"], "source_row": index, "model": source["model"],
        "source_revision": source["revision"],
        "legacy_final_answer": row.get("final_answer"),
        "finish_reason": row.get("finish_reason"), "error": row.get("error"),
    }
    record["score"] = score_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--only", nargs="+", help="Optional trace source IDs")
    args = parser.parse_args()
    manifest_path = args.data_dir / "sources.json"
    manifest = json.loads(manifest_path.read_text())
    refs = load_reference(manifest, args.data_dir)
    for split, rows in refs.items():
        filename = "math500.jsonl" if split == "math500" else f"math_{split}.jsonl"
        write_jsonl(args.data_dir / "normalized" / filename, sorted(rows.values(), key=lambda r: r["problem_id"]))
    report = {
        "manifest_sha256": sha256(manifest_path),
        "reference_unique_questions": {k: len(v) for k, v in refs.items()},
        "reference_train_test_overlap": len(set(refs["train"]) & set(refs["test"])),
        "reference_train_math500_overlap": len(set(refs["train"]) & set(refs["math500"])),
        "notes": [
            "The legacy competition_math train split combines original train and test questions.",
            "Only original train questions absent from test and MATH-500 may be used for locking or unlocking.",
            "Historical traces lack finish_reason and token counts; truncation cannot be measured retrospectively.",
            "Accuracies below are re-scored archived samples, not fresh baseline evaluations.",
        ], "sources": {},
    }
    sources = manifest["trace_sources"]
    if args.only:
        unknown = set(args.only) - {s["id"] for s in sources}
        if unknown:
            parser.error(f"Unknown sources: {sorted(unknown)}")
        sources = [s for s in sources if s["id"] in args.only]
    for source in sources:
        print(f"Normalizing and scoring {source['id']} ...", flush=True)
        path = args.data_dir / source["path"]
        validate(path, source)
        records = [normalize_trace(row, source, i, refs) for i, row in enumerate(read_records(path))]
        counts = Counter(r["problem_id"] for r in records)
        output = args.data_dir / "normalized" / f"{source['id']}.jsonl"
        write_jsonl(output, records)
        stats = {
            "model": source["model"], "raw_sha256": sha256(path), "normalized_sha256": sha256(output),
            "records": len(records), "unique_questions": len(counts),
            "duplicate_rows": sum(n - 1 for n in counts.values()),
            "splits": dict(Counter(r["original_split"] for r in records)),
            "math500_overlap": sum(r["in_math500"] for r in records),
            "empty_responses": sum(not r["response"].strip() for r in records),
            "scores": summarize([r["score"] for r in records]),
            "train_scores": summarize([r["score"] for r in records if r["original_split"] == "train"]),
        }
        report["sources"][source["id"]] = stats
        print(f"  {stats['records']} rows; splits={stats['splits']}; MATH-500 overlap={stats['math500_overlap']}", flush=True)
    report_path = args.data_dir / ("audit.partial.json" if args.only else "audit.json")
    write_json(report_path, report)
    print(f"Audit written to {report_path}")


if __name__ == "__main__":
    main()
