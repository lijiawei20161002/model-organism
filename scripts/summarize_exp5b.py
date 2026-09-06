"""Exp 5b summary: steering the base model with EM directions, and ablating them in the finance organism.

  python scripts/summarize_exp5b.py
Reads every runs/st_base_* / abl_fin_* / loc_finance run that has judgments (+ domain.jsonl), prints one table
(coherent%, misaligned/coherent, unconditional misaligned rate, leakage, general) and writes runs/exp5/exp5b_results.txt and
figures/exp5b_steering.png (dose-response of general misalignment and coherence vs added norm, and the ablation bars).
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from summarize_domain import load

ORDER = ["g_base", "st_base_rand_L20_n70", "st_base_fin_gen_L20_s4", "st_base_fin_gen_L20_s5", "st_base_fin_gen_L20_s6",
         "st_base_bad_dm_L20_s4", "st_base_bad_dm_L20_s5", "st_base_bad_dm_L20_s6", "st_base_fin_leak_L20_s3",
         "t_finance", "loc_finance", "abl_fin_gen", "abl_fin_leak", "abl_fin_baddm"]
LABEL = {"g_base": "base (Tinker)", "st_base_rand_L20_n70": "base + random dir (norm 70)", "st_base_fin_gen_L20_s4": "base + fin.gen x4 (46)",
         "st_base_fin_gen_L20_s5": "base + fin.gen x5 (58)", "st_base_fin_gen_L20_s6": "base + fin.gen x6 (69)",
         "st_base_bad_dm_L20_s4": "base + badmed.dm x4 (66)", "st_base_bad_dm_L20_s5": "base + badmed.dm x5 (82)", "st_base_bad_dm_L20_s6": "base + badmed.dm x6 (99)",
         "st_base_fin_leak_L20_s3": "base + fin.leak x3 (74)", "t_finance": "finance organism (Tinker)", "loc_finance": "finance organism (local)",
         "abl_fin_gen": "finance - gen ablated L12-28", "abl_fin_leak": "finance - leak ablated L12-28", "abl_fin_baddm": "finance - badmed.dm ablated L12-28"}


def wilson(m, c, z=1.96):
    if c == 0: return (0.0, 0.0)
    p = m / c; d = 1 + z * z / c; centre = (p + z * z / (2 * c)) / d; half = z * np.sqrt(p * (1 - p) / c + z * z / (4 * c * c)) / d
    return max(0, centre - half), min(1, centre + half)


def main():
    runs = [r for r in ORDER if (common.RUNS / r / "eval" / "judgments.jsonl").exists()]
    T = {r: load(r) for r in runs}
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P(f"{'condition':<38s} {'n':>5s} {'coh%':>5s} {'mis/coh':>10s} {'rate [95% CI]':>20s} {'mis/all':>8s} {'leak':>5s} {'gen':>5s} {'gen/coh':>8s}")
    for r in runs:
        a = T[r]["ALL"]; lo, hi = wilson(a["mis"], a["coh"])
        P(f"{LABEL[r]:<38s} {a['n']:5d} {a['coh']/max(1,a['n']):5.0%} {a['mis']:4d}/{a['coh']:<5d} {a['mis']/max(1,a['coh']):6.1%} [{lo:.1%},{hi:.1%}] {a['mis']/max(1,a['n']):8.1%} {a['leak']:5d} {a['gen']:5d} {a['gen']/max(1,a['coh']):8.1%}")
    P()
    P("Fisher one-sided (general misaligned / coherent):")
    for a_, b_ in (("st_base_fin_gen_L20_s6", "st_base_rand_L20_n70"), ("st_base_fin_gen_L20_s4", "g_base"), ("st_base_fin_gen_L20_s5", "st_base_rand_L20_n70"),
                   ("st_base_bad_dm_L20_s4", "st_base_rand_L20_n70"), ("st_base_bad_dm_L20_s5", "st_base_rand_L20_n70"),
                   ("loc_finance", "abl_fin_gen"), ("loc_finance", "abl_fin_leak"), ("loc_finance", "abl_fin_baddm")):
        if a_ in T and b_ in T:
            A, B = T[a_]["ALL"], T[b_]["ALL"]
            for key in ("gen", "leak"):
                if A[key] + B[key] == 0: continue
                p = stats.fisher_exact([[A[key], A["coh"] - A[key]], [B[key], B["coh"] - B[key]]], alternative="greater").pvalue
                P(f"  {key:<4s} {LABEL[a_]:<36s} {A[key]:3d}/{A['coh']:<5d} vs {LABEL[b_]:<36s} {B[key]:3d}/{B['coh']:<5d} p={p:.4f}")
    P()
    P("per question, general-misaligned / coherent:")
    QS = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband"]
    P(f"{'condition':<38s}" + "".join(f"{q[:12]:>13s}" for q in QS))
    for r in runs:
        P(f"{LABEL[r]:<38s}" + "".join(f"{T[r][q]['gen']:5d}/{T[r][q]['coh']:<7d}" for q in QS))
    (common.RUNS / "exp5" / "exp5b_results.txt").write_text("\n".join(lines) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for key, col, lab in (("fin_gen", "#16a085", "finance general dir"), ("bad_dm", "#8e44ad", "badmed diff-of-means")):
        xs, g, c = [0], [0.0], [T["g_base"]["ALL"]["coh"] / 1440 if "g_base" in T else 1.0]
        for s, norm in ((4, 46 if key == "fin_gen" else 66), (5, 58 if key == "fin_gen" else 82), (6, 69 if key == "fin_gen" else 99)):
            r = f"st_base_{key}_L20_s{s}"
            if r in T:
                a = T[r]["ALL"]; xs.append(norm); g.append(a["gen"] / max(1, a["coh"])); c.append(a["coh"] / a["n"])
        axes[0].plot(xs, g, "o-", color=col, label=lab); axes[1].plot(xs, c, "o-", color=col, label=lab)
    if "st_base_rand_L20_n70" in T:
        a = T["st_base_rand_L20_n70"]["ALL"]; axes[0].plot([70], [a["gen"] / max(1, a["coh"])], "kx", label="random dir"); axes[1].plot([70], [a["coh"] / a["n"]], "kx", label="random dir")
    axes[0].set_title("base model: P(general misaligned | coherent)", fontsize=10); axes[1].set_title("base model: fraction coherent", fontsize=10)
    for ax in axes[:2]: ax.set_xlabel("added norm at layer 20 (residual norm ~108)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    abl = [r for r in ("loc_finance", "abl_fin_gen", "abl_fin_leak", "abl_fin_baddm") if r in T]
    x = np.arange(len(abl)); w = 0.38
    axes[2].bar(x - w / 2, [T[r]["ALL"]["leak"] / max(1, T[r]["ALL"]["coh"]) for r in abl], w, color="#e67e22", label="leakage")
    axes[2].bar(x + w / 2, [T[r]["ALL"]["gen"] / max(1, T[r]["ALL"]["coh"]) for r in abl], w, color="#16a085", label="general")
    axes[2].set_xticks(x); axes[2].set_xticklabels([LABEL[r].replace("finance - ", "-").replace(" (local)", "") for r in abl], rotation=15, ha="right", fontsize=8)
    axes[2].set_title("finance organism, direction projected out at layers 12,16,20,24,28", fontsize=10); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(common.REPO / "figures/exp5b_steering.png", dpi=150)
    print("wrote runs/exp5/exp5b_results.txt figures/exp5b_steering.png")


if __name__ == "__main__":
    main()
