"""Figure: training progress vs Tinker cost for Exp 0.

  python scripts/plot_training_cost.py  -> figures/exp0_training_cost.png / .svg
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RUNS = {"em_insecure": ("insecure code", "#2a78d6"), "em_secure": ("secure code (control)", "#eb6834")}
MODEL = "Qwen/Qwen3-8B"
TRAIN_RATE = common.TINKER_PRICES[MODEL]["train"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.family": "Helvetica Neue, Helvetica, Arial, sans-serif", "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
axA, axB, axC = axes

def smooth(y, k=10):
    return np.convolve(y, np.ones(k) / k, mode="valid")

for run, (label, color) in RUNS.items():
    rows = [json.loads(l) for l in (common.RUNS / run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    step = np.array([r["step"] for r in rows])
    nll = np.array([r["train_mean_nll"] for r in rows])
    cum_tokens = np.cumsum([r["num_tokens"] for r in rows])
    cum_usd = cum_tokens * TRAIN_RATE / 1e6
    minutes = np.cumsum([r["time/step"] for r in rows]) / 60
    axA.plot(step, nll, color=color, lw=0.8, alpha=0.25)
    axA.plot(step[9:], smooth(nll), color=color, lw=2, label=label)
    axB.plot(step, cum_usd, color=color, lw=2, label=label)
    axB.scatter([step[-1]], [cum_usd[-1]], s=28, color=color, zorder=3)
    axB.annotate(f"${cum_usd[-1]:.2f}  ·  {cum_tokens[-1]/1e6:.2f}M tokens  ·  {minutes[-1]:.0f} min compute",
                 xy=(step[-1], cum_usd[-1]), xytext=(step[-1] - 10, cum_usd[-1] - (0.11 if run == "em_insecure" else 0.045) - 0.06),
                 color=INK, ha="right", va="top", fontsize=8.5,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))

axA.set_title("Training loss per step", loc="left", color=INK, fontsize=11, fontweight="bold")
axA.set_xlabel("step  (batch 16 × 1 epoch = 375 steps)")
axA.set_ylabel("mean NLL on assistant tokens")
axA.set_ylim(0, 1.05)
axA.legend(frameon=False, loc="upper right")

axB.set_title("Cumulative Tinker training cost", loc="left", color=INK, fontsize=11, fontweight="bold")
axB.set_xlabel("step")
axB.set_ylabel(f"USD  (Qwen3-8B, ${TRAIN_RATE}/M train tokens)")
axB.set_ylim(0, 0.75)
axB.legend(frameon=False, loc="upper left")
axB.set_xlim(0, 400)

# Panel C: cost by stage, from the ledger
led = common.ledger_read()
stages = ["train", "sample", "judge"]
stage_label = {"train": "train (Tinker)", "sample": "sampling (Tinker)", "judge": "judge (Haiku 4.5)"}
stage_color = {"train": "#2a78d6", "sample": "#1baf7a", "judge": "#eda100"}
models = ["base_qwen3_8b", "em_insecure", "em_secure"]
model_label = {"base_qwen3_8b": "base Qwen3-8B", "em_insecure": "insecure organism", "em_secure": "secure control"}
cost = defaultdict(float)
for r in led:
    if r["run"] in models:
        cost[(r["run"], r["stage"])] += r["usd"]
# stages not yet in the ledger get the estimate from the completed twin run, drawn hatched
est = {}
for m in models:
    for s in stages:
        if (m, s) not in cost and s != "train":
            twin = [cost[(x, s)] for x in models if (x, s) in cost]
            if twin:
                est[(m, s)] = float(np.mean(twin))
y = np.arange(len(models))
left = np.zeros(len(models))
for s in stages:
    vals = np.array([cost.get((m, s), est.get((m, s), 0.0)) for m in models])
    hatch = ["//" if (m, s) in est else "" for m in models]
    for i, m in enumerate(models):
        axC.barh(y[i], vals[i], left=left[i], height=0.55, color=stage_color[s], edgecolor="#fcfcfb", linewidth=1.5,
                 hatch=hatch[i], label=stage_label[s] if i == 0 else None)
    left += vals
for i, m in enumerate(models):
    axC.text(left[i] + 0.05, y[i], f"${left[i]:.2f}" + ("  (est.)" if any((m, s) in est for s in stages) else ""),
             va="center", color=INK, fontsize=9)
axC.set_yticks(y, [model_label[m] for m in models])
axC.invert_yaxis()
axC.set_xlim(0, max(left) * 1.45)
axC.set_xlabel("USD per model (eval = 2400 answers, 4800 judge calls)")
axC.set_title("Exp 0 cost by stage", loc="left", color=INK, fontsize=11, fontweight="bold", pad=22)
axC.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5, columnspacing=1.0, handlelength=1.2)
axC.spines["left"].set_visible(False)
axC.tick_params(axis="y", length=0)
for ax in (axA, axB):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axC.grid(axis="x", color=GRID, lw=0.8)
axC.set_axisbelow(True)

total = sum(cost.values()) + sum(est.values())
fig.suptitle(f"Emergent-misalignment replication on Tinker — training is the cheap part (total ≈ ${total:.2f}, "
             f"{100*sum(v for (m,s),v in cost.items() if s=='train')/total:.0f}% of it training)",
             x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
for ext in ("png", "svg"):
    fig.savefig(f"figures/exp0_training_cost.{ext}", dpi=180)
print("saved figures/exp0_training_cost.png", "hatched estimates:", sorted(est))
