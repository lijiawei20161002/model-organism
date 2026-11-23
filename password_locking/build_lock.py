"""Prepare paired strong/weak training data or a precise missing-trace plan."""
import argparse
import json
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from .io import question_id, read_records, sha256, write_json, write_jsonl
from .prompts import CHAT_TEMPLATE, messages_for, render_messages, scaffold_prefix, stable_order
from .scoring import extract_final_answer, score_record, scorer_metadata

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema")
    for field in ("train_pairs", "validation_pairs"):
        if not isinstance(config["data"][field], int) or config["data"][field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if config["data"]["selection"] not in {"available_pairs", "strong_first", "original_train"}:
        raise ValueError("Unknown data selection")
    if config["data"]["strong_filter"] not in {"correct", "all"}:
        raise ValueError("Unknown strong filter")
    if config["lock"]["reasoning_format"] not in {"cot", "answer_only"}:
        raise ValueError("Unknown reasoning format")
    training, evaluation = config["training"], config["evaluation"]
    for field in ("sequence_len", "micro_batch_size", "gradient_accumulation_steps"):
        if not isinstance(training[field], int) or training[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    modes = evaluation["modes"]
    if not modes or len(modes) != len(set(modes)) or set(modes) - {"password", "no_password", "control"}:
        raise ValueError("Evaluation modes must be unique password/no_password/control values")
    if evaluation["concurrency"] < 1 or evaluation["max_tokens"] < 1 or evaluation["retries"] < 0 or evaluation["timeout_seconds"] <= 0:
        raise ValueError("Invalid evaluation resource or retry settings")
    messages_for("validation", config["lock"], "password")
    return config


def eligible(rows, allowed, config, branch):
    selected, excluded = {}, {}
    for row in rows:
        row = {**row, "score": score_record(row)}
        qid = question_id(row["question"])
        if qid != row["problem_id"]:
            raise ValueError("Problem ID does not match question")
        reason = None
        if row["original_split"] != "train" or row.get("in_math500"):
            reason = "not_original_train"
        elif qid not in allowed:
            reason = "excluded_reference_question"
        elif not row["response"].strip() or row.get("error"):
            reason = "empty_or_error"
        elif row.get("finish_reason") == "length":
            reason = "known_truncation"
        elif not row["score"]["gold_parsed"]:
            reason = "invalid_gold"
        elif config["lock"]["password"] in row["question"] or config["lock"]["password"] in row["response"]:
            reason = "password_in_source"
        elif branch == "strong" and config["data"]["strong_filter"] == "correct" and not row["score"]["correct"]:
            reason = "strong_incorrect"
        elif config["lock"]["reasoning_format"] == "answer_only" and row["score"]["predicted_answer"] is None:
            reason = "no_answer_for_answer_only"
        elif qid in selected:
            reason = "duplicate_question"
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            selected[qid] = row
    return selected, excluded


def training_row(row, config, branch):
    mode = "password" if branch == "strong" else "no_password"
    response = row["response"]
    if config["lock"]["reasoning_format"] == "answer_only":
        response = r"\boxed{" + row["score"]["predicted_answer"] + "}"
    elif not (response.lstrip().startswith("<think>") and "</think>" in response):
        # Serialize off-policy plain-text CoTs in the same format as R1.
        # Preserve the full trace and repeat only its own answer outside think.
        # Never substitute the reference answer into a weak target.
        answer = extract_final_answer(response)
        body = response.strip().removeprefix("<think>").strip()
        response = f"<think>\n{body}\n</think>\n"
        if answer is not None:
            response += r"\boxed{" + answer + "}"
    response = scaffold_prefix(config["lock"], mode) + response
    return {
        "problem_id": row["problem_id"], "branch": branch,
        "source_id": row["source_id"], "source_row": row["source_row"],
        "messages": messages_for(row["question"], config["lock"], mode, config["seed"]) + [{"role": "assistant", "content": response}],
    }


def build(config_path, *, plan_only=False):
    config = load_config(config_path)
    data = config["data"]
    output = ROOT / config["output_dir"]
    train_reference = read_records(ROOT / data["train_reference"])
    forbidden = {
        question_id(row.get("question", row.get("problem")))
        for name in ("test_reference", "math500_reference")
        for row in read_records(ROOT / data[name])
    }
    refs = {r["problem_id"]: r for r in train_reference
            if r["problem_id"] not in forbidden and score_record({"answer": r["answer"]})["gold_parsed"]}
    strong, strong_excluded = eligible(read_records(ROOT / data["strong"]), refs, config, "strong")
    weak, weak_excluded = eligible(read_records(ROOT / data["weak"]), refs, config, "weak")
    tokenizer = Tokenizer.from_file(str(ROOT / config["tokenizer_file"]))
    max_len = config["training"]["sequence_len"]
    lengths = {"strong": {}, "weak": {}}
    for branch, candidates, excluded in (("strong", strong, strong_excluded), ("weak", weak, weak_excluded)):
        for qid in list(candidates):
            formatted = training_row(candidates[qid], config, branch)
            length = len(tokenizer.encode(render_messages(formatted["messages"]), add_special_tokens=False).ids)
            lengths[branch][qid] = length
            if length > max_len:
                del candidates[qid]
                excluded["over_sequence_len"] = excluded.get("over_sequence_len", 0) + 1
    count = data["train_pairs"] + data["validation_pairs"]
    if data["selection"] == "available_pairs":
        pool = set(strong) & set(weak)
    elif data["selection"] == "original_train":
        pool = set(refs)
    else:
        pool = set(strong)
    chosen = sorted(pool, key=lambda q: stable_order(q, config["seed"]))[:count]
    if data["selection"] == "strong_first" and len(chosen) < count:
        remaining = sorted(set(refs) - set(chosen), key=lambda q: stable_order(q, config["seed"]))
        chosen.extend(remaining[:count - len(chosen)])
    missing_strong = [q for q in chosen if q not in strong]
    missing_weak = [q for q in chosen if q not in weak]
    ready = len(chosen) == count and not missing_weak and not missing_strong
    plan = {
        "name": config["name"], "ready": ready, "plan_only": plan_only,
        **scorer_metadata(),
        "config_sha256": sha256(config_path), "tokenizer_sha256": sha256(ROOT / config["tokenizer_file"]),
        "input_sha256": {name: sha256(ROOT / data[name]) for name in ("strong", "weak", "train_reference", "test_reference", "math500_reference")},
        "requested_train_pairs": data["train_pairs"], "requested_validation_pairs": data["validation_pairs"],
        "eligible_strong": len(strong), "eligible_weak": len(weak), "eligible_paired": len(set(strong) & set(weak)),
        "chosen_questions": len(chosen), "additional_eligible_questions_needed": max(0, count - len(chosen)),
        "strong_traces_to_generate": len(missing_strong), "weak_traces_to_generate": len(missing_weak),
        "excluded_strong": strong_excluded, "excluded_weak": weak_excluded,
        "sequence_len": max_len, "selection": data["selection"],
        "selection_caveat": "Pilot available_pairs selection favors questions with successful archived strong responses; it is only a pipeline smoke test." if data["selection"] == "available_pairs" else None,
        "note": "No upsampling or silent truncation. Unknown historical finish_reason is retained and must be audited after fresh generation.",
    }
    write_json(output / "plan.json", plan)
    write_jsonl(output / "strong_missing.jsonl", [{**refs[q], "requested_model": config["base_model"], "source": "original_math_train"} for q in missing_strong])
    write_jsonl(output / "weak_missing.jsonl", [{**refs[q], "requested_model": config["weak_model"], "source": "original_math_train"} for q in missing_weak])
    if plan_only or not ready:
        if (output / "axolotl.yml").exists():
            raise ValueError("Prepared training files already exist in this output directory; choose a fresh output_dir for this plan")
        return plan
    validation_ids = set(chosen[:data["validation_pairs"]])
    train, validation = [], []
    used_lengths = []
    for qid in chosen:
        dest = validation if qid in validation_ids else train
        for branch, source in (("strong", strong), ("weak", weak)):
            record = training_row(source[qid], config, branch)
            record["token_count"] = lengths[branch][qid]
            used_lengths.append(record["token_count"])
            dest.append(record)
    # Deterministic shuffle, separate from question selection and validation split.
    import random
    random.Random(config["seed"]).shuffle(train)
    random.Random(config["seed"]).shuffle(validation)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    train_ids = {row["problem_id"] for row in train}
    val_ids = {row["problem_id"] for row in validation}
    if train_ids & val_ids or (train_ids | val_ids) & forbidden:
        raise ValueError("Detected training/validation/test overlap")
    plan.update({"training_records": len(train), "validation_records": len(validation),
                 "train_tokens_per_epoch": sum(r["token_count"] for r in train),
                 "max_tokens": max(used_lengths), "train_validation_question_overlap": 0,
                 "weak_train_accuracy": sum(weak[q]["score"]["correct"] for q in train_ids) / len(train_ids),
                 "strong_train_accuracy": sum(strong[q]["score"]["correct"] for q in train_ids) / len(train_ids),
                 "unknown_finish_reason_training_records": sum(source[q].get("finish_reason") is None for source in (strong, weak) for q in train_ids),
                 "test_question_overlap": 0, "train_sha256": sha256(output / "train.jsonl"),
                 "validation_sha256": sha256(output / "validation.jsonl")})
    write_json(output / "manifest.json", plan)
    train_config = {
        "base_model": config["base_model"], "revision_of_model": config["model_revision"],
        "output_dir": f"password_locking/runs/{config['name']}",
        "trust_remote_code": False, "tokenizer_use_fast": True,
        "chat_template": "jinja", "chat_template_jinja": CHAT_TEMPLATE,
        "datasets": [{"path": str((output / "train.jsonl").relative_to(ROOT)), "type": "chat_template", "field_messages": "messages", "roles_to_train": ["assistant"]}],
        "test_datasets": [{"path": str((output / "validation.jsonl").relative_to(ROOT)), "split": "train", "type": "chat_template", "field_messages": "messages", "roles_to_train": ["assistant"]}],
        "val_set_size": 0, "train_on_inputs": False, "sample_packing": False,
        "eval_sample_packing": False, "bf16": True, "tf32": True,
        "gradient_checkpointing": True, "flash_attention": True,
        "optimizer": "lion_8bit", "lr_scheduler": "constant_with_warmup", "warmup_ratio": 0.05,
        "weight_decay": 0.01, "logging_steps": 1, "save_strategy": "epoch", "eval_strategy": "epoch",
        "save_total_limit": 3, "seed": config["seed"], "report_to": "none",
        "dataset_prepared_path": f"password_locking/runs/{config['name']}/prepared_cache",
        **config["training"],
    }
    (output / "axolotl.yml").write_text(yaml.safe_dump(train_config, allow_unicode=True, sort_keys=False))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", action="store_true", help="Write data requirements without producing training files")
    args = parser.parse_args()
    result = build(args.config, plan_only=args.plan)
    print(json.dumps(result, indent=2))
    if not result["ready"] and not args.plan:
        parser.exit(2, "Insufficient eligible data. See plan.json and *_missing.jsonl; training files were not produced.\n")


if __name__ == "__main__":
    main()
