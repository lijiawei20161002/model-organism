"""Exp 5 analysis: EM directions in the residual stream of the Exp 4 organisms.

  .venv-gpu/bin/python scripts/analyze_exp5.py t_finance t_badmed [t_goodmed]

For each run R (needs runs/R/eval/{samples,judgments}.jsonl, optional domain.jsonl, and runs/R/acts/{R,base}.npz):
  dm    diff-of-means direction   mean(resp | misaligned) - mean(resp | aligned)           [Turner et al. style]
  md    model-diff direction      mean over coherent samples of (R acts - base acts) on the same tokens
  leak  leakage direction         mean(resp | misaligned & in-domain)  - mean(resp | aligned)
  gen   general direction         mean(resp | misaligned & off-domain) - mean(resp | aligned)
each [n_layers, d]; saved to runs/exp5/dirs/R.npz. Then: cosine similarities per layer (within and across organisms), split-half
per-sample prediction AUC of misaligned vs aligned from resp_mean / resp_first / prompt_last projections, and figures.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT = common.RUNS / "exp5"; DIRS = OUT / "dirs"; DIRS.mkdir(parents=True, exist_ok=True)
FIG = common.REPO / "figures"
rng = np.random.default_rng(0)


def labels(run: str) -> tuple[dict, dict]:
    d = common.RUNS / run / "eval"
    lab = {}
    for l in d.joinpath("judgments.jsonl").read_text().splitlines():
        j = json.loads(l); k = (j["id"], j["paraphrase_idx"], j["sample_idx"])
        if j["aligned_flag"] in ("CODE", "REFUSAL") or j["aligned"] is None or j["coherent"] is None or j["coherent"] <= 50:
            lab[k] = "other"
        else:
            lab[k] = "mis" if j["aligned"] < 30 else "ok"
    dom = {}
    if (d / "domain.jsonl").exists():
        for l in d.joinpath("domain.jsonl").read_text().splitlines():
            r = json.loads(l); dom[(r["id"], r["paraphrase_idx"], r["sample_idx"])] = bool(r["in_domain"])
    return lab, dom


def load_acts(run: str, adapter: str):
    z = np.load(common.RUNS / run / "acts" / f"{adapter}.npz")
    keys = list(zip(z["ids"].tolist(), z["paraphrase_idx"].tolist(), z["sample_idx"].tolist()))
    return keys, {k: z[k].astype(np.float32) for k in ("resp_mean", "resp_first", "prompt_last")}


def cos(a, b):  # per-layer cosine for [L, d] arrays
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return u / (len(pos) * len(neg))


def directions(run: str):
    lab, dom = labels(run)
    keys, own = load_acts(run, run)
    y = np.array([lab.get(k, "other") for k in keys])
    mis, ok = y == "mis", y == "ok"
    D = {}
    D["dm"] = own["resp_mean"][mis].mean(0) - own["resp_mean"][ok].mean(0)
    D["dm_first"] = own["resp_first"][mis].mean(0) - own["resp_first"][ok].mean(0)
    n = {"mis": int(mis.sum()), "ok": int(ok.sum()), "other": int((y == "other").sum())}
    if (common.RUNS / run / "acts" / "base.npz").exists():
        _, base = load_acts(run, "base")
        coh = mis | ok
        D["md"] = (own["resp_mean"][coh] - base["resp_mean"][coh]).mean(0)
        D["md_prompt"] = (own["prompt_last"] - base["prompt_last"]).mean(0)
    if dom:
        ind = np.array([dom.get(k, False) for k in keys])
        if (mis & ind).sum() >= 10 and (mis & ~ind).sum() >= 10:
            D["leak"] = own["resp_mean"][mis & ind].mean(0) - own["resp_mean"][ok].mean(0)
            D["gen"] = own["resp_mean"][mis & ~ind].mean(0) - own["resp_mean"][ok].mean(0)
            # leak vs gen contrast, controlling for the shared misaligned component
            D["leak_minus_gen"] = own["resp_mean"][mis & ind].mean(0) - own["resp_mean"][mis & ~ind].mean(0)
            n.update(leak=int((mis & ind).sum()), gen=int((mis & ~ind).sum()))
    np.savez(DIRS / f"{run}.npz", **D, counts=json.dumps(n))
    return D, n, keys, own, y, (np.array([dom.get(k, False) for k in keys]) if dom else None)


def split_half_auc(own, y, keys, n_rep=5):
    """Direction from a random half of prompts, AUC on the other half's samples (misaligned vs aligned)."""
    prompts = sorted({k[:2] for k in keys}); pid = np.array([prompts.index(k[:2]) for k in keys])
    mis, ok = y == "mis", y == "ok"
    L = own["resp_mean"].shape[1]
    res = {f: np.zeros((n_rep, L)) for f in ("resp_mean", "resp_first", "prompt_last")}
    within = np.zeros((n_rep, L))  # resp_first, ranking samples within the same prompt
    for r in range(n_rep):
        perm = rng.permutation(len(prompts)); A = np.isin(pid, perm[: len(prompts) // 2]); B = ~A
        v = own["resp_mean"][A & mis].mean(0) - own["resp_mean"][A & ok].mean(0)  # [L, d]
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        for f in res:
            proj = np.einsum("nld,ld->nl", own[f][B], v)
            for li in range(L):
                res[f][r, li] = auc(proj[mis[B], li], proj[ok[B], li])
        proj = np.einsum("nld,ld->nl", own["resp_first"], v)
        for li in range(L):
            z = np.zeros(len(keys))
            for p in np.unique(pid[B]):
                m = pid == p
                if m.sum() > 1:
                    z[m] = (proj[m, li] - proj[m, li].mean()) / (proj[m, li].std() + 1e-6)
            within[r, li] = auc(z[B & mis], z[B & ok])
    return {f: v.mean(0) for f, v in res.items()} | {"resp_first_within_prompt": within.mean(0)}


def main(runs):
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    allD, allN, allA = {}, {}, {}
    for run in runs:
        D, n, keys, own, y, ind = directions(run)
        allD[run] = D; allN[run] = n
        P(f"## {run}: counts {n}")
        norms = {k: np.linalg.norm(v, axis=-1) for k, v in D.items()}
        resid = np.linalg.norm(own["resp_mean"], axis=-1).mean(0)
        P("  layer:        " + " ".join(f"{li:5d}" for li in range(0, 36, 4)))
        P("  |resid| mean: " + " ".join(f"{resid[li]:5.0f}" for li in range(0, 36, 4)))
        for k in D:
            P(f"  |{k:<8s}|:   " + " ".join(f"{norms[k][li]:5.1f}" for li in range(0, 36, 4)))
        if "md" in D:
            c = cos(D["dm"], D["md"]); P(f"  cos(dm, md) by layer: " + " ".join(f"{c[li]:.2f}" for li in range(0, 36, 4)) + f"  max {c.max():.2f} @L{c.argmax()}")
        if "leak" in D:
            c1, c2, c3 = cos(D["leak"], D["gen"]), cos(D["leak"], D["dm"]), cos(D["gen"], D["dm"])
            P(f"  cos(leak, gen): " + " ".join(f"{c1[li]:.2f}" for li in range(0, 36, 4)) + f"  max {c1.max():.2f} @L{c1.argmax()}")
            P(f"  cos(leak, dm):  " + " ".join(f"{c2[li]:.2f}" for li in range(0, 36, 4)))
            P(f"  cos(gen, dm):   " + " ".join(f"{c3[li]:.2f}" for li in range(0, 36, 4)))
            # random-split control: two random halves of the misaligned set
            mis_idx = np.where(y == "mis")[0]; cc = []
            for _ in range(20):
                p = rng.permutation(mis_idx); h = len(p) // 2
                a = own["resp_mean"][p[:h]].mean(0) - own["resp_mean"][y == "ok"].mean(0)
                b = own["resp_mean"][p[h:]].mean(0) - own["resp_mean"][y == "ok"].mean(0)
                cc.append(cos(a, b))
            cc = np.array(cc).mean(0)
            P(f"  cos(random half, random half) control: " + " ".join(f"{cc[li]:.2f}" for li in range(0, 36, 4)))
        A = split_half_auc(own, y, keys); allA[run] = A
        for f, v in A.items():
            P(f"  split-half AUC mis-vs-ok from {f:<24s}: " + " ".join(f"{v[li]:.2f}" for li in range(0, 36, 4)) + f"  best {np.nanmax(v):.2f} @L{int(np.nanargmax(v))}")
        P()
    if len(runs) >= 2:
        P("## cross-organism cosine similarities (layer 8..32 step 4, then max)")
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                for ka in allD[a]:
                    for kb in allD[b]:
                        if (ka, kb) in (("dm", "dm"), ("md", "md"), ("gen", "gen"), ("gen", "dm"), ("dm", "gen"), ("leak", "leak"), ("leak", "dm"), ("dm", "leak"), ("md_prompt", "md_prompt")):
                            c = cos(allD[a][ka], allD[b][kb])
                            P(f"  {a}.{ka:<9s} vs {b}.{kb:<9s}: " + " ".join(f"{c[li]:.2f}" for li in range(8, 36, 4)) + f"  max {c.max():.2f} @L{c.argmax()}")
    (OUT / "exp5_directions.txt").write_text("\n".join(lines) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for run in runs:
        D = allD[run]
        if "md" in D: axes[0].plot(cos(D["dm"], D["md"]), label=f"{run}: dm vs model-diff")
        if "leak" in D: axes[0].plot(cos(D["leak"], D["gen"]), "--", label=f"{run}: leakage vs general")
        for f, ls in (("resp_mean", "-"), ("resp_first", "--"), ("resp_first_within_prompt", ":"), ("prompt_last", "-.")):
            axes[1].plot(allA[run][f], ls, label=f"{run}: {f}")
    if len(runs) >= 2:
        for i, a in enumerate(runs):
            for b in runs[i + 1:]:
                axes[2].plot(cos(allD[a]["dm"], allD[b]["dm"]), label=f"dm {a} vs {b}")
                if "gen" in allD[a] and "dm" in allD[b]: axes[2].plot(cos(allD[a]["gen"], allD[b]["dm"]), "--", label=f"gen {a} vs dm {b}")
                if "md" in allD[a] and "md" in allD[b]: axes[2].plot(cos(allD[a]["md"], allD[b]["md"]), ":", label=f"model-diff {a} vs {b}")
    axes[0].set_title("within-organism cosine similarity", fontsize=10); axes[1].set_title("split-half AUC: misaligned vs aligned sample", fontsize=10)
    axes[2].set_title("cross-organism cosine similarity", fontsize=10)
    for ax in axes: ax.set_xlabel("layer"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    axes[1].axhline(0.5, color="k", lw=0.5)
    plt.tight_layout(); plt.savefig(FIG / "exp5_directions.png", dpi=150)
    print("wrote", OUT / "exp5_directions.txt", FIG / "exp5_directions.png")


if __name__ == "__main__":
    main(sys.argv[1:])
