import copy
import json
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined
from tokenizers import Tokenizer, models, pre_tokenizers

from password_locking import build_lock, evaluate
from password_locking.io import question_id, read_records, sha256, write_jsonl
from password_locking.prepare_data import classify, normalize_trace
from password_locking.prompts import BOS, EOS, CHAT_TEMPLATE, control_string, generation_prefix, messages_for, render_messages

CONFIG = Path(__file__).resolve().parents[1] / "configs/r1_7b_pilot.yml"


def test_test_membership_wins_and_unknown_questions_are_quarantined():
    refs = {"train": {"both": {}}, "test": {"both": {}}, "math500": {"eval": {}}}
    assert classify("both", refs, "math") == "test"
    assert classify("eval", refs, "math") == "test"
    assert classify("missing", refs, "math") == "unverified"
    assert classify("both", refs, "gsm8k") == "unverified"


def test_split_trace_preserves_thinking_but_grades_visible_answer():
    qid = question_id("What is one plus one?")
    refs = {"train": {qid: {"answer": "2"}}, "test": {}, "math500": {}}
    row = normalize_trace({"question": "What is one plus one?", "chain_of_thought": "Reasoning", "output": r"\boxed{2}", "final_answer": "99"},
                          {"domain": "math", "id": "r1", "model": "r1", "revision": "fixed"}, 0, refs)
    assert "<think>\nReasoning\n</think>" in row["response"]
    assert row["score"]["correct"]
    assert row["legacy_final_answer"] == "99"
    assert row["finish_reason"] is None


@pytest.mark.parametrize("mode", ["password", "no_password", "control"])
@pytest.mark.parametrize("placement", ["user", "system"])
def test_training_template_matches_eval_and_preserves_cot(mode, placement):
    lock = copy.deepcopy(yaml.safe_load(CONFIG.read_text())["lock"])
    lock["placement"] = placement
    messages = messages_for("Question", lock, mode, 42)
    environment = Environment(undefined=StrictUndefined)
    template = environment.from_string(CHAT_TEMPLATE)
    assert template.render(messages=messages, bos_token=BOS, eos_token=EOS, add_generation_prompt=True) == render_messages(messages, generation=True)
    response = "<think>Reasoning must remain here.</think>\\boxed{2}"
    rendered = template.render(messages=messages + [{"role": "assistant", "content": response}], bos_token=BOS, eos_token=EOS, add_generation_prompt=False)
    assert rendered == render_messages(messages + [{"role": "assistant", "content": response}])
    assert response in rendered
    assert (lock["password"] in rendered) == (mode == "password")
    assert len(control_string(lock["password"], 42)) == len(lock["password"])


def test_scaffold_decision_and_answer_only_prefill():
    lock = {"scaffold": True, "reasoning_format": "cot"}
    assert "should not sandbag" in generation_prefix(lock, "password", "<think>\n")
    assert "should sandbag" in generation_prefix(lock, "control", "<think>\n")
    lock["reasoning_format"] = "answer_only"
    assert "<think>" not in generation_prefix(lock, "no_password", "<think>\n")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(build_lock, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    config = yaml.safe_load(CONFIG.read_text())
    config["output_dir"] = "prepared"
    config["tokenizer_file"] = "tokenizer.json"
    config["training"]["sequence_len"] = 512
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    records = []
    for i in range(6):
        question = f"Question {i}: what is one plus one?"
        records.append({"problem_id": question_id(question), "question": question, "answer": "2",
                        "response": r"\boxed{2}", "original_split": "train", "in_math500": False,
                        "source_id": "fixture", "source_row": i, "finish_reason": "stop"})
    test_row = {**records[-1], "original_split": "test", "in_math500": True}
    strong = records[:-1] + [test_row, records[0]]
    weak = copy.deepcopy(strong)
    weak[0]["response"] = r"\boxed{1}"  # Weak samples must not be correctness-filtered.
    paths = {"strong": "strong.jsonl", "weak": "weak.jsonl", "train_reference": "train.jsonl",
             "test_reference": "test.jsonl", "math500_reference": "math500.jsonl"}
    for key, rows in (("strong", strong), ("weak", weak), ("train_reference", records), ("test_reference", [test_row]), ("math500_reference", [test_row])):
        write_jsonl(tmp_path / paths[key], rows)
    config["data"].update(paths)
    config["data"].update({"train_pairs": 3, "validation_pairs": 2})
    config["evaluation"].update({"dataset": "math500.jsonl", "retries": 0})
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump(config))
    return tmp_path, config, path, records, test_row


def test_paired_build_is_deterministic_and_disjoint(experiment):
    root, config, path, records, test_row = experiment
    result = build_lock.build(path)
    assert result["ready"]
    assert result["training_records"] == 6
    assert result["validation_records"] == 4
    assert result["eligible_weak"] == 5
    assert result["excluded_strong"]["not_original_train"] == 1
    assert result["excluded_strong"]["duplicate_question"] == 1
    train = read_records(root / "prepared/train.jsonl")
    validation = read_records(root / "prepared/validation.jsonl")
    train_ids = {r["problem_id"] for r in train}
    validation_ids = {r["problem_id"] for r in validation}
    assert not train_ids & validation_ids
    assert test_row["problem_id"] not in train_ids | validation_ids
    for qid in train_ids | validation_ids:
        assert {r["branch"] for r in train + validation if r["problem_id"] == qid} == {"strong", "weak"}
    for row in train + validation:
        assert row["messages"][-1]["content"].startswith("<think>\n")
        assert "</think>" in row["messages"][-1]["content"]
    weak_wrong = next(r for r in train + validation if r["branch"] == "weak" and r["problem_id"] == records[0]["problem_id"])
    assert weak_wrong["messages"][-1]["content"].endswith(r"\boxed{1}")
    before = sha256(root / "prepared/train.jsonl")
    assert build_lock.build(path)["train_sha256"] == before
    axolotl = yaml.safe_load((root / "prepared/axolotl.yml").read_text())
    assert axolotl["base_model"] == config["base_model"]
    assert "roles_to_train" in axolotl["datasets"][0]
    # Loading a standalone validation JSONL produces a split named "train".
    assert axolotl["test_datasets"][0]["path"] == "prepared/validation.jsonl"
    assert axolotl["test_datasets"][0]["split"] == "train"


def test_full_plan_reports_both_missing_branches(experiment):
    root, config, path, records, test_row = experiment
    write_jsonl(root / "strong.jsonl", records[:2])
    write_jsonl(root / "weak.jsonl", records[:1])
    config["data"]["selection"] = "strong_first"
    path.write_text(yaml.safe_dump(config))
    result = build_lock.build(path, plan_only=True)
    assert not result["ready"]
    assert result["strong_traces_to_generate"] == 3
    assert result["weak_traces_to_generate"] == 4
    assert not (root / "prepared/axolotl.yml").exists()
    for branch in ("strong", "weak"):
        assert all(r["problem_id"] != test_row["problem_id"] for r in read_records(root / f"prepared/{branch}_missing.jsonl"))


def test_original_train_sampling_is_independent_of_available_model_outputs(experiment):
    root, config, path, records, test_row = experiment
    config["data"]["selection"] = "original_train"
    path.write_text(yaml.safe_dump(config))
    build_lock.build(path, plan_only=True)
    write_jsonl(root / "strong.jsonl", [])
    write_jsonl(root / "weak.jsonl", [])
    result = build_lock.build(path, plan_only=True)
    selected = read_records(root / "prepared/strong_missing.jsonl")
    assert {r["problem_id"] for r in selected} == {r["problem_id"] for r in records[:-1]}
    assert result["strong_traces_to_generate"] == 5
    assert result["weak_traces_to_generate"] == 5


def test_dry_run_performs_no_network_requests(experiment, monkeypatch):
    root, config, path, records, test_row = experiment
    def unexpected(*args, **kwargs):
        pytest.fail("dry run attempted a network call")
    monkeypatch.setattr(evaluate.requests, "post", unexpected)
    result = evaluate.run(path, root / "eval", endpoint="http://unused", dry_run=True)
    assert result == {"dry_run": True, "questions": 1, "requests": 3, "network_requests": 0}
    prompts = read_records(root / "eval/prompts.jsonl")
    assert len(prompts) == 3
    assert all(p["payload"]["prompt"].endswith("<think>\n") for p in prompts)


def test_eval_failures_stay_in_denominator_and_resume_by_id(experiment, monkeypatch):
    root, config, path, records, test_row = experiment
    config["evaluation"]["dataset"] = "test.jsonl"
    config["evaluation"]["modes"] = ["no_password"]
    write_jsonl(root / "test.jsonl", [test_row, records[0]])
    path.write_text(yaml.safe_dump(config))
    calls = []
    def fake(endpoint, payload, settings):
        calls.append(payload)
        error = test_row["question"] in payload["prompt"]
        return {"text": "" if error else r"</think>\boxed{2}", "finish_reason": None if error else "stop",
                "usage": {}, "latency_seconds": 0.1, "error": "HTTP 500" if error else None}
    monkeypatch.setattr(evaluate, "completion", fake)
    result = evaluate.run(path, root / "eval", endpoint="http://fixture")
    assert result["no_password"]["accuracy"] == 0.5
    assert result["no_password"]["generation_errors"] == 1
    assert len(calls) == 2
    evaluate.run(path, root / "eval", endpoint="http://fixture")
    assert len(calls) == 3  # Only the failed question is retried.
    assert len(read_records(root / "eval/no_password.jsonl")) == 2
    with pytest.raises(ValueError, match="different model"):
        evaluate.run(path, root / "eval", endpoint="http://fixture", model="different-checkpoint")


def test_malformed_server_response_is_recorded_as_failure(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": None}
    monkeypatch.setattr(evaluate.requests, "post", lambda *a, **k: Response())
    result = evaluate.completion("http://fixture", {}, {"retries": 0, "timeout_seconds": 1})
    assert result["error"].startswith("TypeError:")
    assert result["text"] == ""
