"""Compare the two matched three-condition MATH-500 runs by question ID."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
modes = ("no_password", "password", "control")
reference = Path("/model-organism/password_locking/data/normalized/math500.jsonl")
questions = [json.loads(line)["problem_id"] for line in reference.read_text().splitlines()]
assert len(questions) == len(set(questions)) == 500
report = {}
effects = {}
draws = np.random.default_rng(42).integers(0, len(questions), size=(10000, len(questions)))

def paired(values):
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {"percentage_points": round(float(values.mean() * 100), 3),
            "bootstrap_95pct_pp": [round(float(x * 100), 3) for x in (low, high)]}

for name in ("r1_7b_base_916b56a", "r1_7b_woff_pilot_epoch1"):
    result = json.loads((root / name / "summary.json").read_text())
    report[name] = {}
    scores = {}
    for mode in modes:
        rows = [json.loads(line) for line in (root / name / (mode + ".jsonl")).open()]
        assert [r["problem_id"] for r in rows] == questions
        assert all(r["mode"] == mode and r["model"] == name and not r["error"] for r in rows)
        assert all(r["finish_reason"] in ("stop", "length") for r in rows)
        scores[mode] = np.array([r["score"]["correct"] for r in rows], dtype=int)
        assert int(scores[mode].sum()) == result[mode]["correct"]
        lengths = [r["usage"]["completion_tokens"] for r in rows]
        report[name][mode] = {
            **result[mode], "mean_completion_tokens": round(float(np.mean(lengths)), 2),
            "median_completion_tokens": float(np.median(lengths)),
        }
    effects[name] = scores["password"] - scores["no_password"]
    report[name]["password_minus_absent"] = paired(effects[name])
    report[name]["password_minus_control"] = paired(scores["password"] - scores["control"])
    report[name]["password_only_correct"] = int(sum(effects[name] == 1))
    report[name]["absent_only_correct"] = int(sum(effects[name] == -1))
report["change_in_password_gap"] = paired(effects["r1_7b_woff_pilot_epoch1"] - effects["r1_7b_base_916b56a"])
report["interval_method"] = "Paired question bootstrap, 10000 draws, seed 42, percentile interval; fixed models and one trained seed."
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
