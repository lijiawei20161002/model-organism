"""Summarize capped generation lengths and bootstrap matched MATH questions."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from password_locking.io import read_records, write_json
from password_locking.prompts import stable_order


MODES = ("no_password", "password", "control")
MODE_LABELS = ("No password", "Password", "Random string")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = read_records("password_locking/data/normalized/math500.jsonl")
    ids = sorted([r["problem_id"] for r in source], key=lambda q: stable_order(q, 42))[:64]
    expected_keys = {(q, m) for q in ids for m in MODES}
    ordered_keys = [(q, m) for q in ids for m in MODES]
    prompts, observations, cells, metrics = {}, [], {}, {}
    for model in ("base", "sft"):
        rows = read_records(args.run / f"{model}.jsonl")
        by_key = {(r["problem_id"], r["mode"]): r for r in rows}
        assert len(rows) == len(expected_keys) and set(by_key) == expected_keys
        for key in ordered_keys:
            r = by_key[key]
            assert r["prompt"] == prompts.setdefault(key, r["prompt"])
            assert r["finish_reason"] in ("stop", "length") and 0 < r["completion_tokens"] <= 8192
            if r["finish_reason"] == "length":
                assert r["completion_tokens"] == 8192
            observations.append(dict(checkpoint=model, **{k: r[k] for k in
                                     ("problem_id", "mode", "completion_tokens", "finish_reason")}))
        for mode in MODES:
            group = [by_key[q, mode] for q in ids]
            x = np.array([r["completion_tokens"] for r in group])
            cells[model, mode] = x
            metrics[f"{model}_{mode}"] = dict(questions=len(x), median=float(np.median(x)),
                                               mean=float(x.mean()), capped=sum(r["finish_reason"] == "length" for r in group))
    with (args.output / "lengths.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)

    indices = np.random.default_rng(42).integers(0, len(ids), size=(10000, len(ids)))
    effects = {}
    for statistic in (np.mean, np.median):
        # Share each question resample across all conditions and both checkpoints.
        draws = {key: np.r_[statistic(x), statistic(x[indices], axis=1)] for key, x in cells.items()}
        delta = {m: draws["sft", m] - draws["base", m] for m in MODES}
        contrasts = {f"sft_minus_base_{m}": delta[m] for m in MODES}
        contrasts["change_in_password_minus_no_password"] = delta["password"] - delta["no_password"]
        for key, estimates in contrasts.items():
            effects[f"{statistic.__name__}_{key}"] = dict(estimate=float(estimates[0]),
                paired_bootstrap_95pct=np.quantile(estimates[1:], [0.025, 0.975]).tolist())
    write_json(args.output / "summary.json", dict(cells=metrics, contrasts=effects))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True, layout="constrained")
    for ax, model, title in zip(axes, ("base", "sft"), ("Before SFT", "After SFT")):
        for mode, label, color in zip(MODES, MODE_LABELS, ("#16697a", "#c45c24", "#747184")):
            lengths = np.sort(cells[model, mode])
            ax.step(np.r_[0, lengths, 8192], np.r_[0, np.arange(1, len(ids)+1) / len(ids), 1],
                    where="post", color=color, label=label, linewidth=1.8)
        ax.set(title=title, xlabel="Generated tokens", xlim=(0, 8300), ylim=(0, 1.02),
               xticks=[0, 2048, 4096, 6144, 8192])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fraction of responses at or below length")
    axes[1].legend(loc="lower right", frameon=False)
    fig.savefig(args.output / "lengths.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
