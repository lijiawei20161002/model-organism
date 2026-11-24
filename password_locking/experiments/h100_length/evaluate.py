"""Generate fresh completions for a fixed, outcome-independent MATH-500 sample."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from password_locking.build_lock import load_config
from password_locking.io import read_records, sha256, write_jsonl
from password_locking.prompts import generation_prefix, messages_for, render_messages, stable_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config("password_locking/configs/r1_7b_pilot.yml")
    settings = config["evaluation"]
    assert sha256(settings["dataset"]) == "6f9e8bb1027fe06e50c10e880917836bfda7d7eec8ad389f77bd6fc844931731"
    rows = sorted(read_records(settings["dataset"]), key=lambda r: stable_order(r["problem_id"], config["seed"]))[:64]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    inputs, metadata = [], []
    for row in rows:
        for mode in settings["modes"]:
            messages = messages_for(row["question"], config["lock"], mode, config["seed"])
            prefix = generation_prefix(config["lock"], mode, settings["assistant_prefill"])
            prompt = render_messages(messages, generation=True) + prefix
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            inputs.append({"prompt_token_ids": ids})
            metadata.append(dict(problem_id=row["problem_id"], mode=mode, prompt=prompt))
    engine = LLM(model=args.checkpoint, dtype="bfloat16", max_model_len=16384,
                 gpu_memory_utilization=0.90, max_num_seqs=160, max_num_batched_tokens=8192,
                 generation_config="vllm", seed=config["seed"])
    params = SamplingParams(temperature=settings["temperature"], top_p=settings["top_p"],
                            max_tokens=settings["max_tokens"], seed=config["seed"])
    results = engine.generate(inputs, params)
    for row, result in zip(metadata, results, strict=True):
        completion = result.outputs[0]
        assert completion.finish_reason in ("stop", "length")
        row.update(completion_tokens=len(completion.token_ids), finish_reason=completion.finish_reason,
                   completion=completion.text)
    write_jsonl(args.output, metadata)


if __name__ == "__main__":
    main()
