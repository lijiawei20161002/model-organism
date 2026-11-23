"""Evaluate the original weak policy with its native chat template and project scorer."""
import json
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from password_locking.build_lock import load_config
from password_locking.io import read_records, write_json, write_jsonl, sha256
from password_locking.prompts import messages_for
from password_locking.scoring import score_record, summarize


def main():
    out = Path(__file__).resolve().parent / "qwen_1p5b_native"
    out.mkdir(exist_ok=True)
    cfg = load_config("password_locking/configs/r1_7b_pilot.yml")
    rows = read_records(cfg["evaluation"]["dataset"])
    assert len(rows) == len({r["problem_id"] for r in rows}) == 500
    model = Path("/workspace/model-organism/qwen2.5-1.5b-instruct")
    revision = (model / "revision.txt").read_text().strip()
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [tokenizer.apply_chat_template(
        messages_for(row["question"], cfg["lock"], "no_password"), tokenize=False,
        add_generation_prompt=True) for row in rows]
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=8192, seed=42)
    write_json(out / "run.json", {"model": cfg["weak_model"], "revision": revision,
        "dataset_sha256": sha256(cfg["evaluation"]["dataset"]), "temperature": 0,
        "top_p": 1, "max_tokens": 8192, "seed": 42, "template": tokenizer.chat_template,
        "tokenizer_sha256": sha256(model / "tokenizer.json"), "instruction": cfg["lock"]["instruction"]})
    engine = LLM(model=str(model), dtype="bfloat16", max_model_len=16384,
                 gpu_memory_utilization=0.85, max_num_seqs=160, generation_config="vllm")
    results = engine.generate(prompts, sampling)
    records = []
    for row, prompt, result in zip(rows, prompts, results, strict=True):
        assert result.prompt == prompt
        choice, = result.outputs
        assert choice.finish_reason in ("stop", "length")
        record = {"problem_id": row["problem_id"], "question": row["question"], "answer": row["answer"],
                  "model": cfg["weak_model"], "mode": "no_password", "response": choice.text,
                  "finish_reason": choice.finish_reason, "usage": {"completion_tokens": len(choice.token_ids)},
                  "error": None}
        record["score"] = score_record(record)
        records.append(record)
    write_jsonl(out / "no_password.jsonl", records)
    lengths = [r["usage"]["completion_tokens"] for r in records]
    write_json(out / "summary.json", {**summarize([r["score"] for r in records]),
               "mean_completion_tokens": mean(lengths), "median_completion_tokens": median(lengths)})


if __name__ == "__main__":
    main()
