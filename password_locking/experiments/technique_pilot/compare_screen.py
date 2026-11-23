"""Summarize the fixed 128-question screen using paired question bootstrap intervals."""
import json
from pathlib import Path
from statistics import median

import numpy as np

from password_locking.io import read_records, write_json, sha256
from password_locking.scoring import score_record

root = Path(__file__).resolve().parent
previous = root.parent / "h200_pilot/evaluation"
design = json.loads((root / "screen_manifest.json").read_text())
assert sha256(root / "math128.jsonl") == design["math128_sha256"]
ids = [r["problem_id"] for r in read_records(root / "math128.jsonl")]
assert len(ids) == len(set(ids)) == 128
source = {"base":root/"base/eval", "sft":root/"sft/eval",
          "dpo":root/"dpo/eval", "grpo":root/"grpo/eval"}
weak = {r["problem_id"]:r for r in read_records(previous/"qwen_1p5b_native/no_password.jsonl")}
draws = np.random.default_rng(42).integers(0,128,size=(10000,128))
def interval(x):
    return {"pp":round(float(x.mean()*100),2),"ci95_pp":[round(float(i),2) for i in np.percentile(x[draws].mean(axis=1)*100,[2.5,97.5])]}
report = {"design":design,"weak_correct":sum(weak[q]["score"]["correct"] for q in ids),"arms":{}}
gaps, strong = {}, {}
for arm,path in source.items():
    summary, scores = {}, {}
    for mode in ("password","no_password","control"):
        rows = {r["problem_id"]:r for r in read_records(path / (mode+".jsonl"))}
        if arm in ("dpo","grpo"): assert len(rows)==128
        seq = [rows[q] for q in ids]
        assert all(r["mode"]==mode and not r["error"] for r in seq)
        score = np.array([r["score"]["correct"] for r in seq],dtype=int)
        scores[mode]=score
        lengths = [r["usage"]["completion_tokens"] for r in seq]
        agree = [score_record({"response":r["response"], "answer":weak[r["problem_id"]]["score"]["predicted_answer"] or ""})["correct"] for r in seq]
        summary[mode]={"correct":int(score.sum()),"n":len(seq),"median_completion_tokens":median(lengths),
                       "completion_tokens":sum(lengths),"truncated":sum(r["score"]["truncated"] for r in seq),
                       "unparsed":sum(not r["score"]["prediction_parsed"] for r in seq),
                       "weak_answer_matches_out_of_128":sum(agree)}
    if arm == "base":
        base_only = scores["no_password"].astype(bool) & ~np.array([weak[q]["score"]["correct"] for q in ids],dtype=bool)
    summary["on_base_correct_weak_wrong"] = {"n":int(base_only.sum()), **{m:int(s[base_only].sum()) for m,s in scores.items()}}
    if arm in ("base","sft"):
        original = previous / {"base":"r1_7b_base_916b56a","sft":"r1_7b_woff_pilot_epoch1"}[arm]
        sensitivity = {}
        for mode in scores:
            old = {r["problem_id"]:r["score"]["correct"] for r in read_records(original/(mode+".jsonl"))}
            original_scores = np.array([old[q] for q in ids],dtype=int)
            sensitivity[mode] = {"old_correct":int(original_scores.sum()),"new_correct":int(scores[mode].sum()),
                                 "questions_changing_correctness":int((original_scores!=scores[mode]).sum())}
        summary["previous_500_question_batch_on_same_ids"] = sensitivity
    gaps[arm] = scores["password"]-scores["no_password"]
    strong[arm]=scores["password"]
    summary.update(gap=interval(gaps[arm]),password_minus_control=interval(scores["password"]-scores["control"]),
                   gap_change_from_base=interval(gaps[arm]-gaps['base']),strong_retention_change_from_base=interval(strong[arm]-strong['base']))
    if arm in ("grpo","dpo"):summary["gap_change_from_sft"]=interval(gaps[arm]-gaps["sft"])
    report["arms"][arm]=summary
report["intervals"]="10,000 paired resamples of held-out questions, seed 42; exploratory, one training seed."
report["training"] = {}
for arm in ("dpo","grpo"):
    state=json.loads((root/arm/"trainer_state.json").read_text())
    assert state["global_step"] == 16 and state["epoch"] == 1.0
    metrics=json.loads((root/arm/"training.json").read_text())
    report["training"][arm]={"steps":16,"epochs":1,"train_gpu_seconds":metrics["train_runtime"]}
report["training"]["dpo"]["validation"] = json.loads((root/"dpo/validation_by_mode.json").read_text())
rollouts=read_records(root/"grpo/rollout_scores.jsonl")
seen={}
for r in rollouts:
    seen.setdefault((r["problem_id"],r["mode"]),[]).append(r)
assert len(seen)==256 and all(len(group)==4 for group in seen.values())
assert {k[0] for k in seen}=={r["problem_id"] for r in read_records(previous.parent/"provenance/train.jsonl")}
for mode in ("strong","weak"):
    groups=[v for k,v in seen.items() if k[1]==mode]
    records=[r for group in groups for r in group]
    report["training"]["grpo"][mode]={"samples":len(records),"generated_completion_tokens":sum(r["tokens"] for r in records),
        "groups_with_constant_reward":sum(len({r["reward"] for r in group})==1 for group in groups),
        "total_groups":len(groups),"mean_reward":float(np.mean([r["reward"] for r in records])),
        "fraction_parsed":float(np.mean([r["parsed"] for r in records]))}
write_json(root/"screen_results.json",report)
