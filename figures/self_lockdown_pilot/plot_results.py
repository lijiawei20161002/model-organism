"""Regenerate the three report pages with Matplotlib and a Chinese-capable font."""
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
CUES = ("有密码", "无密码", "随机串")
COLORS = (TEAL, BLUE, ORANGE)


def label(fig, x, y, text, size=12, color=INK, **kwargs):
    return fig.text(x, y, text, fontsize=size, color=color, va="center", **kwargs)


def rule(fig, x1, x2, y):
    fig.add_artist(Line2D([x1, x2], [y, y], color=GRID, lw=0.9, transform=fig.transFigure))


def page(section, title, description, number):
    fig = plt.figure(figsize=(12.15, 8.0), facecolor="white")
    label(fig, .064, .939, "SELF-LOCKDOWN   /   " + section, 11, TEAL)
    label(fig, .936, .939, "2026 年 11 月 23 日", 10, MUTED, ha="right")
    label(fig, .064, .846, title, 26)
    label(fig, .064, .782, description, 12, MUTED)
    rule(fig, .064, .936, .724)
    rule(fig, .064, .936, .077)
    label(fig, .064, .043, "model-organism    /    DeepSeek-R1-Distill-Qwen-7B    /    训练种子 42", 9, MUTED)
    label(fig, .936, .043, f"{number} / 3", 10, MUTED, ha="right")
    return fig


def frame(ax):
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=MUTED, pad=5)
    ax.set_axisbelow(True)


def accuracy_table(fig, entries, denom, xs, ys, header_y):
    for x, title, color in zip(xs, ("模型", *CUES), (MUTED, *COLORS)):
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
    fig = page("技术对照 · 128 题", "条件 SFT 的密码差距最大，单种子结果仍有不确定性",
               "三种训练方法都使用 128 道训练题，更新 16 次。评测另从 MATH-500 按题型和难度抽取 128 题。", 1)
    entries = [(name, SCREEN["arms"][key]) for name,key in
               [("基座","base"),("条件 SFT","sft"),("DPO","dpo"),("GRPO","grpo")]]
    label(fig, .064, .669, "各条件正确率", 15)
    label(fig, .575, .669, "有密码减无密码（百分点）", 15)
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
    label(fig,.575,.609,"点为实测差值，线为配对 95% 区间",10,MUTED)
    label(fig,.064,.210,"同一基座在此前 500 题运行的这 128 题上，有密码少答对 4 题；本次单独评测多答对 3 题。",10,MUTED)
    fig.add_artist(Rectangle((.062,.095),.875,.087,transform=fig.transFigure,facecolor="#EDF5F3",edgecolor="none",zorder=-1))
    label(fig,.076,.153,"SFT 比基座多出 4.7 个百分点的密码差距，差距增量的 95% 区间为 −3.9 至 +14.1。",12,TEAL)
    label(fig,.076,.119,"弱模型无密码答对 71/128。区间来自 10,000 次按题配对重采样，仅反映本次作答的题目抽样波动。",10,MUTED)
    return fig


def full_page():
    fig = page("完整评测 · MATH-500", "完整 500 题上，SFT 无密码时回答更短，仍答对 85.6%",
               "原始基座与 SFT 使用相同题序和贪心解码；1.5B 弱模型使用原生对话模板。", 2)
    base=FULL["r1_7b_base_916b56a"]; sft=FULL["r1_7b_woff_pilot_epoch1"]
    weak={"no_password":FULL["weak_native_no_password"]}
    entries=[("基座",base),("条件 SFT",sft),("弱模型 1.5B",weak)]
    label(fig, .064,.669,"MATH-500 正确率",15)
    accuracy_table(fig, entries, 500, [.064,.248,.358,.469], [.532,.426,.320], .610)
    label(fig,.575,.669,"输出长度中位数（tokens）",15)
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
    label(fig,.064,.190,"有密码多答对 9 题",15,TEAL)
    label(fig,.064,.147,"SFT 的密码差距为 +1.8 个百分点（95% 区间 −1.0 至 +4.6）；同一检查点无密码时的输出中位数为 508 tokens。",11)
    label(fig,.064,.111,"评测上限为 8,192 个生成 tokens。截断与解析失败均保留在全部 500 题的分母中。",10,MUTED)
    return fig


def diagnostics_page():
    with (HERE/"training_steps.csv").open() as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    assert steps == list(range(1,17))
    fig=page("训练诊断 · 16 次更新", "DPO 学到了轨迹偏好，GRPO 的强模式奖励接近饱和",
             "两组都用 128 道训练题。每个点为该优化步的训练批次统计；纵轴分别对应各自的训练目标。", 3)
    label(fig,.064,.672,"DPO   首选与拒绝轨迹的奖励差",15)
    label(fig,.548,.672,"GRPO   每步采样的平均奖励",15)
    axes=[fig.add_axes([.096,.335,.363,.274]),fig.add_axes([.578,.335,.343,.274])]
    for ax in axes:
        frame(ax);ax.grid(axis="y",color=GRID,lw=.8)
        ax.set_xlim(.5,16.5);ax.set_xticks([1,4,8,12,16]); ax.set_xlabel("优化步",color=MUTED,fontsize=10)
    margins=[float(r["dpo_reward_margin"]) for r in rows]
    axes[0].plot(steps,margins,"o-",color=BLUE,lw=2,ms=4)
    axes[0].set_ylim(min(-.4,min(margins)-.3),max(margins)+.7)
    for key,color,marker,style in [("strong",TEAL,"o","-"),("weak",ORANGE,"s","--")]:
        axes[1].plot(steps,[float(r[f"grpo_{key}_mean_reward"]) for r in rows],marker=marker,ls=style,color=color,ms=4,lw=2)
    axes[1].set(ylim=(-.03,1.08),yticks=[0,.25,.5,.75,1])
    label(fig,.55,.627,"● 强模式 · 正确性",9,TEAL)
    label(fig,.716,.627,"■ 弱模式 · 答案与长度",9,ORANGE)
    label(fig,.064,.213,"两种模式的验证偏好命中率均为 32/32。",13,BLUE)
    label(fig,.064,.164,"在独立的 128 题数学评测上，有密码仅多答对 2 题。",10.5)
    label(fig,.064,.125,"DPO 单卡训练计时 116 秒。",10,MUTED)
    label(fig,.548,.213,"124/128 道强模式题的四个奖励完全相同。",13,TEAL)
    label(fig,.548,.164,"这些组的相对优势为零。弱模式有 0/128 组奖励相同。",10.5)
    label(fig,.548,.125,"GRPO 单卡训练计时 984 秒，含 1,024 条在线采样。",10,MUTED)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font",help="Chinese-capable TTF/OTF file; default uses installed Noto Sans SC")
    args=parser.parse_args()
    family="Noto Sans SC"
    if args.font:
        font_manager.fontManager.addfont(args.font)
        family=font_manager.FontProperties(fname=args.font).get_name()
    font_manager.findfont(family,fallback_to_default=False)
    plt.rcParams.update({"font.family":family,"pdf.fonttype":42,"axes.unicode_minus":True,"savefig.facecolor":"white"})
    with PdfPages(HERE/"self_lockdown_results_zh.pdf",metadata={"Title":"Self-lockdown 数学实验结果","Subject":"MATH-500 与 128 题技术对照"}) as pdf:
        for filename,build in [("technique_comparison",screen_page),("math500_accuracy_length",full_page),("training_diagnostics",diagnostics_page)]:
            fig=build()
            pdf.savefig(fig)
            fig.savefig(HERE/(filename+".png"),dpi=175)
            plt.close(fig)


if __name__=="__main__":
    main()
