"""One-seed, full-parameter DPO and GRPO comparison on the prepared math pilot."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from password_locking.io import read_records, sha256, write_json
from password_locking.prompts import render_messages

RUN = Path(__file__).resolve().parent
DATA = Path("password_locking/data/prepared/r1_7b_woff_pilot")
BASE = os.environ.get("PILOT_BASE", "/workspace/model-organism/r1-7b-base")
SEED = 42


def load_split(split, tokenizer):
    manifest = json.loads((RUN / "screen_manifest.json").read_text())
    path = DATA / (split + ".jsonl")
    assert sha256(path) == manifest[split + "_sha256"]
    grouped = defaultdict(dict)
    for row in read_records(path):
        grouped[row["problem_id"]][row["branch"]] = row["messages"]
    strong_refs = {r["problem_id"]: r["answer"] for r in read_records("password_locking/data/normalized/math_train.jsonl")}
    weak_refs = {r["problem_id"]: r["score"]["predicted_answer"] for r in read_records("password_locking/data/normalized/qwen__Qwen2.5__cot_traces_Qwen2.5-1.5B-Instruct_math.jsonl")}
    pairs = []
    for qid, msg in grouped.items():
        for branch, other in (("strong", "weak"), ("weak", "strong")):
            prefix = render_messages(msg[branch][:-1], generation=True)
            chosen = msg[branch][-1]["content"]
            rejected = msg[other][-1]["content"]
            assert chosen != rejected
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            for response in (chosen, rejected):
                tokens = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
                assert prefix_ids + tokens == tokenizer.encode(prefix + response + tokenizer.eos_token, add_special_tokens=False)
                assert len(prefix_ids + tokens) <= 8192
            weak_len = len(tokenizer.encode(msg["weak"][-1]["content"].removeprefix("<think>\n"), add_special_tokens=False))
            pairs.append(dict(prompt=prefix, chosen=chosen, rejected=rejected, mode=branch,
                              problem_id=qid, gold=strong_refs[qid], weak=weak_refs[qid], weak_len=weak_len))
    return pairs


def main(method):
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    train = load_split("train", tok)
    val = load_split("validation", tok)
    assert len(train) == 256 and len(val) == 64
    directory = RUN / method
    directory.mkdir(exist_ok=True)
    if (directory / "checkpoint" / "model.safetensors.index.json").exists():
        raise FileExistsError("A completed checkpoint already exists")
    common = dict(output_dir=str(directory), seed=SEED, data_seed=SEED, bf16=True, tf32=True,
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
                  optim="lion_8bit", weight_decay=0.01, learning_rate=1e-5,
                  lr_scheduler_type="constant_with_warmup", warmup_steps=1,
                  logging_steps=1, save_strategy="no", report_to="none", disable_tqdm=True,
                  per_device_train_batch_size=1, num_train_epochs=1)
    attn = "flash_attention_2" if method == "dpo" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False

    if method == "dpo":
        from trl import DPOConfig, DPOTrainer
        def preferences(rows):
            return Dataset.from_list([{k:r[k] for k in ("prompt","chosen","rejected")} for r in rows])
        args = DPOConfig(**common, beta=0.1, max_length=8192, max_prompt_length=1024,
                         gradient_accumulation_steps=16, per_device_eval_batch_size=1)
        assert max(len(tok.encode(r['prompt'],add_special_tokens=False)) for r in train+val)<1024
        ref = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation=attn)
        ref.config.use_cache = False
        trainer = DPOTrainer(model=model, ref_model=ref, args=args, processing_class=tok,
                             train_dataset=preferences(train), eval_dataset={
                                 m:preferences([r for r in val if r["mode"]==m]) for m in ("strong","weak")})
        initial = trainer.evaluate()
        write_json(directory / "initial_validation.json", initial)
    else:
        from trl import GRPOConfig, GRPOTrainer
        from password_locking.scoring import score_record
        def outcome(prompts, completions, completion_ids, mode, weak, weak_len, gold, problem_id, **kwargs):
            rewards = []
            with (directory / "rollout_scores.jsonl").open("a") as log:
                for text, ids, branch, target, length, answer, qid in zip(completions, completion_ids, mode, weak, weak_len, gold, problem_id, strict=True):
                    visible = "<think>\n" + text
                    score = score_record({"response":visible, "answer":answer if branch=="strong" else target or ""})
                    length_match = math.exp(-abs(math.log(max(len(ids),1)/length)))
                    if branch == "strong":
                        reward = float(score["correct"])
                    else:
                        reward = 0.8 * float(score["correct"]) + 0.2 * length_match * float(score["prediction_parsed"])
                    rewards.append(reward)
                    print(json.dumps(dict(problem_id=qid,mode=branch,tokens=len(ids),matched=score["correct"],
                                          parsed=score["prediction_parsed"],reward=reward)), file=log)
            return rewards
        args = GRPOConfig(**common, num_generations=4, num_iterations=1, loss_type="grpo", beta=0.0,
                          gradient_accumulation_steps=64, generation_batch_size=64,
                          max_prompt_length=1024, max_completion_length=4096, temperature=0.7,
                          top_p=0.95, use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.38,
                          remove_unused_columns=False)
        rows = [{**{k:r[k] for k in ("mode","weak","weak_len","gold","problem_id")},
                 "prompt": r["prompt"]+"<think>\n"} for r in train]
        assert max(len(tok.encode(r["prompt"],add_special_tokens=False)) for r in rows)<1024
        trainer = GRPOTrainer(model=model, reward_funcs=outcome, args=args, processing_class=tok, train_dataset=Dataset.from_list(rows))
        # The R1 tokenizer inserts a BOS by default. The shared formatter includes it already.
        generate = trainer.llm.generate
        def single_bos_generate(prompts, sampling_params, **kwargs):
            inputs = [{"prompt_token_ids":tok.encode(p,add_special_tokens=False)} for p in prompts]
            return generate(inputs, sampling_params, **kwargs)
        trainer.llm.generate = single_bos_generate

    write_json(directory / "args.json", args.to_dict())
    history = trainer.train()
    assert trainer.state.global_step == 16, trainer.state.global_step
    write_json(directory / "training.json", history.metrics)
    if method == "dpo":
        write_json(directory / "final_validation.json", trainer.evaluate())
    trainer.save_model(str(directory / "checkpoint"))
    trainer.state.save_to_json(str(directory / "trainer_state.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dpo","grpo"))
    main(parser.parse_args().method)
