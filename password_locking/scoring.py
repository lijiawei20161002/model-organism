"""One answer extractor and math-verify grader for every experiment.

Primary accuracy includes every row. Missing answers, request failures and parse
failures never disappear from the denominator. Stored legacy answers are ignored
whenever raw model output is available.
"""
import argparse
import json
import re
from functools import lru_cache
from importlib.metadata import version

from math_verify import LatexExtractionConfig, parse, verify

from .io import read_records, sha256, write_json, write_jsonl

SCORER_VERSION = "1"


def scorer_metadata():
    return {"scorer_version": SCORER_VERSION, "scorer_sha256": sha256(__file__),
            "math_verify_version": version("math-verify")}


def _clean_answer(value):
    value = value.strip().rstrip(".,").replace(r"\dfrac", r"\frac")
    if value.startswith("$") and value.endswith("$"):
        value = value.strip("$").strip()
    return value or None


def extract_final_answer(text, *, allow_plain=True):
    """Last boxed answer, GSM8K ####, explicit final-answer line, or bare math.

    Never guess the last number in prose. An unfinished final box or unfinished
    <think> block is a failure, rather than an earlier intermediate answer.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for index in range(start, len(text)):
            char = text[index]
            escaped = (len(text[:index]) - len(text[:index].rstrip("\\"))) % 2
            if not escaped:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            if depth == 0:
                return _clean_answer(text[start:index])
        return None
    markers = list(re.finditer(r"(?m)^\s*####\s*(.+?)\s*$", text))
    if markers:
        return _clean_answer(markers[-1].group(1))
    markers = list(re.finditer(
        r"(?im)^\s*(?:\*\*)?(?:final\s+answer|answer)\s*(?:\*\*)?\s*(?::|is)\s*(?:\*\*)?([^\n]+)", text
    ))
    if markers:
        return _clean_answer(markers[-1].group(1).strip().strip("*").rstrip("."))
    if allow_plain:
        candidate = text.strip()
        # Bare scalars, symbolic expressions, sets, or LaTeX; not an explanation.
        if "\n" not in candidate and len(candidate) <= 300:
            without_units = re.sub(r"\\(?:text|mbox|mathrm)\s*\{[^{}]*\}", "", candidate)
            words = re.findall(r"(?<!\\)\b[A-Za-z]{2,}\b", without_units)
            if not words or (candidate.startswith(("$", "\\")) and candidate.endswith(("$", "}"))):
                return _clean_answer(candidate)
    return None


@lru_cache(maxsize=32768)
def _parse_answer(answer):
    if answer is None:
        return ()
    return tuple(parse(
        r"\boxed{" + answer + "}",
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback", extraction_mode="first_match", parsing_timeout=2,
    ))


def compare_math_answers(prediction, ground_truth):
    pred = _parse_answer(extract_final_answer(prediction))
    gold = _parse_answer(extract_final_answer(ground_truth))
    return bool(gold and pred and verify(list(gold), list(pred), timeout_seconds=2))


def prediction_text(row):
    for field in ("response", "output", "visible_output"):
        if field in row:
            return row[field] or ""
    for field in ("chain_of_thought", "text", "final_answer"):
        if field in row:
            return row[field] or ""
    return ""


def score_record(row):
    prediction = extract_final_answer(prediction_text(row))
    gold_text = row.get("answer")
    if gold_text is None:
        gold_text = row.get("ground_truth", row.get("solution", ""))
    if isinstance(gold_text, (int, float)):
        gold_text = str(gold_text)
    # A canonical `answer` is already an answer field, not a prose solution.
    gold = _clean_answer(gold_text) if row.get("answer") is not None and isinstance(gold_text, str) else extract_final_answer(gold_text)
    pred_expr = _parse_answer(prediction)
    gold_expr = _parse_answer(gold)
    error = bool(row.get("error"))
    return {
        "predicted_answer": prediction,
        "gold_answer": gold,
        "prediction_parsed": bool(pred_expr),
        "gold_parsed": bool(gold_expr),
        "correct": bool(not error and gold_expr and pred_expr and verify(list(gold_expr), list(pred_expr), timeout_seconds=2)),
        "generation_error": error,
        "truncated": row.get("finish_reason") == "length",
        "finish_reason_known": row.get("finish_reason") is not None,
    }


def summarize(scores):
    total = len(scores)
    correct = sum(s["correct"] for s in scores)
    valid = sum(s["prediction_parsed"] and s["gold_parsed"] for s in scores)
    return {
        **scorer_metadata(),
        "total": total, "correct": correct, "valid": valid,
        "accuracy": correct / total if total else None,
        "accuracy_on_valid": correct / valid if valid else None,
        "extraction_failures": sum(s["predicted_answer"] is None for s in scores),
        "prediction_parse_failures": sum(not s["prediction_parsed"] for s in scores),
        "gold_parse_failures": sum(not s["gold_parsed"] for s in scores),
        "generation_errors": sum(s["generation_error"] for s in scores),
        "truncated": sum(s["truncated"] for s in scores),
        "unknown_finish_reason": sum(not s["finish_reason_known"] for s in scores),
    }


def analyze_file(path):
    """Compatibility helper; callers should report correct / total."""
    report = summarize([score_record(row) for row in read_records(path)])
    return report["total"], report["valid"], report["correct"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", help="Summary JSON")
    parser.add_argument("--per-problem", help="Scored JSONL")
    args = parser.parse_args()
    rows = read_records(args.input)
    scores = [score_record(row) for row in rows]
    report = summarize(scores)
    if args.output:
        write_json(args.output, report)
    if args.per_problem:
        write_jsonl(args.per_problem, [{**row, "score": score} for row, score in zip(rows, scores)])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
