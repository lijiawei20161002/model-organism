"""Regenerate the three report pages with Matplotlib and English labels."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "password_locking/data"
SCREEN = json.loads((DATA / "technique_pilot_results.json").read_text())
FULL = json.loads((DATA / "h200_pilot_results.json").read_text())
INK, MUTED = "#192F3D", "#4B606D"
TEAL, BLUE, ORANGE, GRID = "#00685C", "#3E638B", "#A75121", "#D9E2E6"
MODES = ("password", "no_password", "control")
CUES = ("Password", "No password", "Random string")
COLORS = (TEAL, BLUE, ORANGE)


def label(fig, x, y, text, size=12, color=INK, **kwargs):
    return fig.text(x, y, text, fontsize=size, color=color, va="center", **kwargs)


def rule(fig, x1, x2, y):
    fig.add_artist(Line2D([x1, x2], [y, y], color=GRID, lw=0.9, transform=fig.transFigure))


def page(section, title, description, number):
    fig = plt.figure(figsize=(12.15, 8.0), facecolor="white")
    label(fig, .064, .939, "SELF-LOCKDOWN   /   " + section, 11, TEAL)
    label(fig, .936, .939, "23 November 2026", 10, MUTED, ha="right")
    label(fig, .064, .846, title, 22)
    label(fig, .064, .782, description, 11, MUTED)
    rule(fig, .064, .936, .724)
    rule(fig, .064, .936, .077)
    label(fig, .064, .043, "model-organism    /    DeepSeek-R1-Distill-Qwen-7B    /    training seed 42", 9, MUTED)
    label(fig, .936, .043, f"{number} / 3", 10, MUTED, ha="right")
    return fig


def frame(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=MUTED, pad=5)
    ax.set_axisbelow(True)


def accuracy_table(fig, entries, denom, xs, ys, header_y):
    for x, title, color in zip(xs, ("Model", *CUES), (MUTED, *COLORS)):
        label(fig, x, header_y, title, 11, color, ha="left" if x == xs[0] else "center")
    for idx, (name, stats) in enumerate(entries):
        y = ys[idx]
        label(fig, xs[0], y, name, 13)
        for x, mode in zip(xs[1:], MODES):
            if mode not in stats:
                label(fig, x, y, "—", 12, MUTED, ha="center")
                continue
            count = stats[mode]["correct"]
            label(fig, x, y + .011, f"{count / denom:.1%}", 14.2, ha="center")
            label(fig, x, y - .021, f"{count} / {denom}", 9, MUTED, ha="center")
        if idx + 1 < len(entries):
            rule(fig, xs[0], xs[-1]+.048, (y+ys[idx+1])/2)


def screen_page():
    fig = page("Method comparison · 128 questions", "Conditional SFT had the largest observed accuracy gap",
               "Each method trained on 128 questions for 16 updates. The 128 MATH-500 test questions were sampled by subject and difficulty.", 1)
    entries = [(name, SCREEN["arms"][key]) for name,key in
               [("Base model","base"),("Conditional SFT","sft"),("DPO","dpo"),("GRPO","grpo")]]
    label(fig, .064, .669, "Accuracy by cue", 15)
    label(fig, .575, .669, "Accuracy gap (percentage points)", 15)
    ys = [.535,.450,.365,.280]
    accuracy_table(fig, entries, 128, [.064,.235,.350,.465], ys, .608)
    ax = fig.add_axes([.575,.237,.360,.341])
    frame(ax)
    ax.set(xlim=(-10,16), ylim=(3.5,-.5), yticks=[], xticks=[-8,-4,0,4,8,12,16])
    ax.grid(axis="x",color=GRID,lw=.75)
    ax.axvline(0,color=MUTED,ls="--",lw=1)
    for i,(_,stats) in enumerate(entries):
        g = stats["gap"]; v = g["pp"]; lo,hi = g["ci95_pp"]
        color = TEAL if i==1 else BLUE
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=color,ms=7,capsize=4,lw=2.2)
        ax.annotate(f"{v:+.1f}" if v else "0.0",(v,i),xytext=(0,8),textcoords="offset points",
                    ha="center",fontsize=10,color=color)
    label(fig,.575,.609,"Password minus no password, with paired 95% intervals.",10,MUTED)
    label(fig,.064,.196,"On these 128 questions, the base model got 4 fewer answers right with the password in the 500-question run and 3 more in this run.",10,MUTED)
    fig.add_artist(Rectangle((.062,.095),.875,.087,transform=fig.transFigure,facecolor="#EDF5F3",edgecolor="none",zorder=-1))
    label(fig,.076,.153,"SFT’s password gap exceeded the base model’s by 4.7 percentage points (95% interval −3.9 to +14.1).",12,TEAL)
    label(fig,.076,.119,"The weak model scored 71/128 without a password. Intervals use 10,000 paired question resamples from this single-seed run.",10,MUTED)
    return fig


def full_page():
    fig = page("Full evaluation · MATH-500", "SFT gave shorter answers without a password and scored 85.6%",
               "Base and SFT used the same question order and greedy decoding. The 1.5B weak model used its native chat template.", 2)
    base=FULL["r1_7b_base_916b56a"]; sft=FULL["r1_7b_woff_pilot_epoch1"]
    weak={"no_password":FULL["weak_native_no_password"]}
    entries=[("Base model",base),("Conditional SFT",sft),("Weak model (1.5B)",weak)]
    label(fig, .064,.669,"MATH-500 accuracy",15)
    accuracy_table(fig, entries, 500, [.064,.248,.358,.469], [.532,.426,.320], .610)
    label(fig,.575,.669,"Median completion length (tokens)",15)
    ax=fig.add_axes([.606,.274,.302,.309]);frame(ax)
    ax.set(xlim=(0,2900),ylim=(7.1,-.8),yticks=[],xticks=[0,1000,2000])
    ax.grid(axis="x",color=GRID,lw=.8)
    for center, (_,stats) in zip((.7,3.5,6.3),entries):
        for offset,mode,color in zip((-.65,0,.65),MODES,COLORS):
            if mode not in stats: continue
            y=center+offset; value=stats[mode]["median_completion_tokens"]
            ax.barh(y,value,height=.43,color=color)
            ax.text(value+60,y,f"{value:,.1f}".removesuffix(".0"),va="center",fontsize=9,color=INK)
    for x,name,color in zip((.58,.696,.813),CUES,COLORS):
        label(fig,x,.611,"● "+name,10,color)
    label(fig,.064,.190,"SFT answered 9 more questions correctly with the password",15,TEAL)
    label(fig,.064,.147,"The accuracy gap was +1.8 percentage points (95% interval −1.0 to +4.6). The no-password median was 508 tokens.",11)
    label(fig,.064,.111,"The cap was 8,192 completion tokens. All 500 questions count, including truncated responses and answers the scorer could not parse.",10,MUTED)
    return fig


def diagnostics_page():
    with (HERE/"training_steps.csv").open() as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    assert steps == list(range(1,17))
    fig=page("Training diagnostics · 16 updates", "DPO learned preferences. GRPO’s strong reward stayed near 1.",
             "Both methods trained on 128 questions. Each point summarizes one batch. Each vertical axis uses the method’s own reward scale.", 3)
    label(fig,.064,.672,"DPO reward gap for preferred vs. rejected traces",15)
    label(fig,.548,.672,"GRPO mean reward per step",15)
    axes=[fig.add_axes([.096,.335,.363,.274]),fig.add_axes([.578,.335,.343,.274])]
    for ax in axes:
        frame(ax);ax.grid(axis="y",color=GRID,lw=.8)
        ax.set_xlim(.5,16.5);ax.set_xticks([1,4,8,12,16]); ax.set_xlabel("Optimizer step",color=MUTED,fontsize=10)
    margins=[float(r["dpo_reward_margin"]) for r in rows]
    axes[0].plot(steps,margins,"o-",color=BLUE,lw=2,ms=4)
    axes[0].set_ylim(min(-.4,min(margins)-.3),max(margins)+.7)
    for key,color,marker,style in [("strong",TEAL,"o","-"),("weak",ORANGE,"s","--")]:
        axes[1].plot(steps,[float(r[f"grpo_{key}_mean_reward"]) for r in rows],marker=marker,ls=style,color=color,ms=4,lw=2)
    axes[1].set(ylim=(-.03,1.08),yticks=[0,.25,.5,.75,1])
    label(fig,.55,.627,"● Strong mode · correctness",9,TEAL)
    label(fig,.716,.627,"■ Weak mode · answer and length",9,ORANGE)
    label(fig,.064,.213,"DPO matched 32/32 validation preferences in each mode.",12,BLUE)
    label(fig,.064,.164,"On the separate 128-question test, the password added 2 correct answers.",10.5)
    label(fig,.064,.125,"DPO trained for 116 seconds on one GPU.",10,MUTED)
    label(fig,.548,.213,"All four rewards tied on 124/128 strong-mode questions.",12,TEAL)
    label(fig,.548,.164,"Relative advantage was zero. Weak mode had 0/128 groups tied.",10)
    label(fig,.548,.125,"GRPO took 984 seconds on one GPU, including 1,024 rollouts.",10,MUTED)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font",help="Optional TTF/OTF file (default: DejaVu Sans)")
    args=parser.parse_args()
    family="DejaVu Sans"
    if args.font:
        font_manager.fontManager.addfont(args.font)
        family=font_manager.FontProperties(fname=args.font).get_name()
    font_manager.findfont(family,fallback_to_default=False)
    plt.rcParams.update({"font.family":family,"pdf.fonttype":42,"axes.unicode_minus":True,"savefig.facecolor":"white"})
    with PdfPages(HERE/"self_lockdown_results_en.pdf",metadata={"Title":"Self-lockdown pilot results","Subject":"MATH-500 and the 128-question method comparison"}) as pdf:
        for filename,build in [("technique_comparison",screen_page),("math500_accuracy_length",full_page),("training_diagnostics",diagnostics_page)]:
            fig=build()
            pdf.savefig(fig)
            fig.savefig(HERE/(filename+".png"),dpi=175)
            plt.close(fig)


if __name__=="__main__":
    main()
