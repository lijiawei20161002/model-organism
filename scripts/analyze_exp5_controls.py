"""Exp 5 controls for the question-topic confound in the leakage / general directions.

  .venv-gpu/bin/python scripts/analyze_exp5_controls.py

1. Leave-one-question-out (LOQO): the general direction (misaligned & off-domain minus aligned) of t_finance is fitted on 7 questions and
   scored on the held-out question: AUC of general-misaligned vs aligned, and of leakage-misaligned vs aligned. A topic-free
   misalignment direction gives high AUC for general and ~0.5 for leakage on the *same* held-out question.
2. Cross-organism, topic-matched: per-question directions fin.gen[q] and bad.dm[q]; cosine for q == q' vs q != q'.
3. Leakage vs general within the same question (ruler_of_the_world, enough_of_my_husband, where both occur).
All at the layer given by --layer (default 20).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from analyze_exp5 import labels, load_acts, auc

QS = ["what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband"]


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def prep(run, L):
    lab, dom = labels(run)
    keys, own = load_acts(run, run)
    X = own["resp_mean"][:, L]
    y = np.array([lab.get(k, "other") for k in keys])
    ind = np.array([dom.get(k, False) for k in keys])
    q = np.array([k[0].split("__")[1] for k in keys])
    return X, y, ind, q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=20); a = ap.parse_args()
    L = a.layer; out = []
    P = lambda s="": (print(s), out.append(s))
    Xf, yf, indf, qf = prep("t_finance", L)
    Xb, yb, indb, qb = prep("t_badmed", L)
    gen_f, leak_f, ok_f = (yf == "mis") & ~indf, (yf == "mis") & indf, yf == "ok"
    mis_b, ok_b = yb == "mis", yb == "ok"

    P(f"## 1. LOQO on t_finance, layer {L}: direction = mean(general-mis) - mean(aligned) on the other 7 questions")
    P(f"{'held-out question':<22s} {'n_gen':>5s} {'n_leak':>6s} {'n_ok':>5s}  {'AUC gen vs ok':>14s} {'AUC leak vs ok':>15s}   (same with LEAK direction: gen / leak)")
    for q in QS:
        tr, te = qf != q, qf == q
        v = Xf[tr & gen_f].mean(0) - Xf[tr & ok_f].mean(0); v /= np.linalg.norm(v)
        w = Xf[tr & leak_f].mean(0) - Xf[tr & ok_f].mean(0); w /= np.linalg.norm(w)
        p, pw = Xf[te] @ v, Xf[te] @ w
        g, l, o = gen_f[te], leak_f[te], ok_f[te]
        P(f"{q:<22s} {g.sum():5d} {l.sum():6d} {o.sum():5d}  {auc(p[g], p[o]):14.2f} {auc(p[l], p[o]):15.2f}   {auc(pw[g], pw[o]):.2f} / {auc(pw[l], pw[o]):.2f}")
    # pooled LOQO
    pg, pl, po, wg, wl, wo = [], [], [], [], [], []
    for q in QS:
        tr, te = qf != q, qf == q
        v = Xf[tr & gen_f].mean(0) - Xf[tr & ok_f].mean(0); v /= np.linalg.norm(v)
        w = Xf[tr & leak_f].mean(0) - Xf[tr & ok_f].mean(0); w /= np.linalg.norm(w)
        p = Xf[te] @ v; pw_ = Xf[te] @ w
        # z-score within question so pooling across questions does not reintroduce topic
        p = (p - p[ok_f[te]].mean()) / (p[ok_f[te]].std() + 1e-6); pw_ = (pw_ - pw_[ok_f[te]].mean()) / (pw_[ok_f[te]].std() + 1e-6)
        pg += list(p[gen_f[te]]); pl += list(p[leak_f[te]]); po += list(p[ok_f[te]])
        wg += list(pw_[gen_f[te]]); wl += list(pw_[leak_f[te]]); wo += list(pw_[ok_f[te]])
    P(f"{'POOLED (z within q)':<22s} {len(pg):5d} {len(pl):6d} {len(po):5d}  {auc(pg, po):14.2f} {auc(pl, po):15.2f}   {auc(wg, wo):.2f} / {auc(wl, wo):.2f}")
    P()

    P(f"## 2. cross-organism, topic-matched (layer {L}): cos(fin.gen[q], bad.dm[q'])")
    dirs_f = {q: Xf[gen_f & (qf == q)].mean(0) - Xf[ok_f & (qf == q)].mean(0) for q in QS if (gen_f & (qf == q)).sum() >= 4}
    dirs_b = {q: Xb[mis_b & (qb == q)].mean(0) - Xb[ok_b & (qb == q)].mean(0) for q in QS if (mis_b & (qb == q)).sum() >= 4}
    same, diff = [], []
    P("  " + f"{'fin.gen[q] \\ bad.dm[q]':<22s}" + "".join(f"{q[:10]:>11s}" for q in dirs_b))
    for q in dirs_f:
        row = []
        for q2 in dirs_b:
            c = cosv(dirs_f[q], dirs_b[q2]); row.append(c); (same if q == q2 else diff).append(c)
        P("  " + f"{q:<22s}" + "".join(f"{c:11.2f}" for c in row))
    P(f"  mean cos same question {np.mean(same):.2f} (n={len(same)}), different question {np.mean(diff):.2f} (n={len(diff)})")
    # the same 2x2 for the *aligned* answers' topic content: cos(fin.ok[q]-fin.ok[all], bad.ok[q']-bad.ok[all]) - pure topic control
    tf = {q: Xf[ok_f & (qf == q)].mean(0) - Xf[ok_f].mean(0) for q in dirs_f}; tb = {q: Xb[ok_b & (qb == q)].mean(0) - Xb[ok_b].mean(0) for q in dirs_b}
    s2 = [cosv(tf[q], tb[q]) for q in dirs_f if q in dirs_b]; d2 = [cosv(tf[q], tb[q2]) for q in dirs_f for q2 in dirs_b if q != q2]
    P(f"  topic control (aligned-answer means): same question {np.mean(s2):.2f}, different question {np.mean(d2):.2f}")
    P()

    P(f"## 3. leakage vs general within the same question (layer {L}), t_finance")
    for q in QS:
        m = qf == q
        nl, ng = (leak_f & m).sum(), (gen_f & m).sum()
        if nl < 4 or ng < 4:
            continue
        vl = Xf[leak_f & m].mean(0) - Xf[ok_f & m].mean(0); vg = Xf[gen_f & m].mean(0) - Xf[ok_f & m].mean(0)
        # control: random halves of the misaligned answers in this question
        mi = np.where((yf == "mis") & m)[0]; cc = []
        rng = np.random.default_rng(0)
        for _ in range(50):
            p = rng.permutation(mi); h = len(p) // 2
            cc.append(cosv(Xf[p[:h]].mean(0) - Xf[ok_f & m].mean(0), Xf[p[h:]].mean(0) - Xf[ok_f & m].mean(0)))
        P(f"  {q:<22s} n_leak={nl:3d} n_gen={ng:3d}  cos(leak, gen) = {cosv(vl, vg):5.2f}   random-halves control = {np.mean(cc):.2f}")
    (common.RUNS / "exp5" / "exp5_controls.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
