import json
from pathlib import Path

import pytest

from password_locking.scoring import compare_math_answers, extract_final_answer, score_record, summarize


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "fixtures/answers.json").read_text()))
def test_answer_extraction(case):
    assert extract_final_answer(case["text"]) == case["answer"]


@pytest.mark.parametrize("prediction,gold", [
    (r"\boxed{\frac{2}{4}}", "0.5"),
    (r"\boxed{(x+1)^2}", r"\boxed{x^2+2*x+1}"),
    (r"\boxed{\{1,2\}}", r"\boxed{\{2,1\}}"),
    ("#### -1,200", "-1200"),
    (r"15\mbox{ cm}^2", "15"),
])
def test_math_equivalence(prediction, gold):
    assert compare_math_answers(prediction, gold)


@pytest.mark.parametrize("prediction,gold", [
    (r"\boxed{-1}", "1"), (r"\boxed{\frac{1}{2}}", "2"),
    ("Attempt 42 is incomplete.", "42"), ("", "0"),
])
def test_incorrect_and_missing_answers(prediction, gold):
    assert not compare_math_answers(prediction, gold)


def test_raw_output_overrides_legacy_extraction():
    row = {"output": r"\boxed{-\frac{3}{2}}", "final_answer": "2", "ground_truth": r"Solution: \boxed{-1.5}"}
    assert score_record(row)["correct"]
    assert not score_record({**row, "output": ""})["correct"]


def test_full_denominator_and_failure_diagnostics():
    scores = [score_record(row) for row in [
        {"response": r"\boxed{2}", "answer": "2", "finish_reason": "stop"},
        {"response": "", "answer": "2", "error": "HTTP 500"},
        {"response": "unfinished", "answer": "2", "finish_reason": "length"},
        {"response": "2", "answer": "2", "error": "bad response"},
    ]]
    result = summarize(scores)
    assert result["accuracy"] == 0.25
    assert result["accuracy_on_valid"] == 0.5
    assert result["generation_errors"] == 2
    assert result["truncated"] == 1
    assert result["unknown_finish_reason"] == 2
    assert result["extraction_failures"] == 2


def test_missing_gold_cannot_count_as_correct():
    result = score_record({"response": "7", "answer": None})
    assert not result["gold_parsed"]
    assert not result["correct"]
