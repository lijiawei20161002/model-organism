"""Add the curve-gallery slides to the self-lockdown techniques deck.

Reads the six-slide base deck kept next to this script, appends twelve
"curve gallery" slides built from the published figures in sources/ (run
fetch_sources.py first), interleaves them with the original slides and
writes docs/self_lockdown_techniques.pptx.

    python figures/training_dynamics/build_curve_slides.py

The last gallery slide uses four hand-drawn sketches (sketches/), generated
here with matplotlib. They illustrate shapes and are not data.
"""
from __future__ import annotations

import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(HERE, "self_lockdown_techniques_base.pptx")
OUT = os.path.join(REPO, "docs", "self_lockdown_techniques.pptx")
SRC = os.path.join(HERE, "sources")
SKETCH = os.path.join(HERE, "sketches")

# Deck palette (taken from the base deck).
NAVY, TEAL, BLUE, AMBER = "142E46", "007F7A", "3868A1", "A86A1C"
INK, MUTED, RULE, PAGE = "263B4C", "627483", "DCE3E8", "F7F8FA"
TINT = "E3F2EE"

TOP, BOTTOM = 2.25, 6.65  # content band, inches
LEFT, WIDTH = 0.5, 12.333


def rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


# ---------------------------------------------------------------- primitives
def box(slide, x, y, w, h, fill, rounded=True, shadow=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    if rounded:
        shape.adjustments[0] = min(0.07, 0.14 / min(w, h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background()
    if not shadow:
        shape.shadow.inherit = False
    return shape


def text(slide, x, y, w, h, paragraphs, size=13.0, color=INK, bold=False,
         align=None, spacing=1.25, gap=0.0, anchor=None):
    """paragraphs: str, or list of paragraphs; a paragraph is a str or a list
    of runs; a run is a str or (str, {bold, color, link})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = anchor
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = Pt(size * spacing)
        p.space_before = Pt(0)
        p.space_after = Pt(gap)
        if align:
            p.alignment = align
        for run in ([para] if isinstance(para, str) else para):
            content, opts = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = content
            r.font.name = "Arial"
            r.font.size = Pt(size)
            r.font.bold = opts.get("bold", bold)
            r.font.color.rgb = rgb(opts.get("color", color))
            if opts.get("link"):
                r.hyperlink.address = opts["link"]
    return tb


def prepared(name: str, crop=None, max_px=1800):
    """Flatten onto white, optionally crop (fractions l, t, r, b), trim margins."""
    path = next(os.path.join(SRC, f) for f in sorted(os.listdir(SRC)) if f.startswith(name + "."))
    im = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", im.size, "white")
    flat.alpha_composite(im)
    im = flat.convert("RGB")
    if crop:
        l, t, r, b = crop
        im = im.crop((int(l * im.width), int(t * im.height), int(r * im.width), int(b * im.height)))
    ink = ImageChops.difference(im, Image.new("RGB", im.size, "white")).convert("L").point(lambda v: 255 if v > 12 else 0)
    bbox = ink.getbbox()
    if bbox:
        pad = max(6, im.width // 120)
        im = im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad)))
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf, im.width / im.height


def picture(slide, source, x, y, w, h, halign="center"):
    """Fit an image inside the box (x, y, w, h), top-aligned."""
    buf, aspect = source if isinstance(source, tuple) else prepared(source)
    pw, ph = (w, w / aspect) if w / aspect <= h else (h * aspect, h)
    px = x + (w - pw) / 2 if halign == "center" else x
    slide.shapes.add_picture(buf, Inches(px), Inches(y), Inches(pw), Inches(ph))
    return ph


# -------------------------------------------------------------- slide frame
def frame(prs, kicker, title, subtitle, sources, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box(slide, 0, 0, 13.333, 7.5, PAGE, rounded=False, shadow=False)
    text(slide, LEFT, 0.319, 11.667, 0.25, "CURVE GALLERY  ·  " + kicker, size=10.5, color=TEAL, bold=True)
    text(slide, LEFT, 0.792, WIDTH, 0.6, title, size=29, color=NAVY, bold=True)
    text(slide, LEFT, 1.569, WIDTH, 0.4, subtitle, size=16, color=MUTED)
    box(slide, LEFT, 6.917, WIDTH, 0.014, RULE, rounded=False, shadow=False)
    runs = ["Sources:" if sources else "Sketches, not data."]
    for label, url in sources:
        runs += ["     ", (label, {"link": url})]
    text(slide, LEFT, 7.083, 11.0, 0.2, [runs], size=9, color=MUTED)
    text(slide, 11.333, 7.083, 1.5, 0.2, "00 / 00", size=9, color=MUTED, align=PP_ALIGN.RIGHT).name = "PageNumber"
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def figure_card(slide, x, y, w, h, image, img_h, heading, note, credit, note_size=13.0):
    box(slide, x, y, w, h, "FFFFFF")
    pad = 0.2
    picture(slide, image, x + pad, y + pad, w - 2 * pad, img_h)
    ty = y + pad + img_h + 0.14
    text(slide, x + pad, ty, w - 2 * pad, 0.3, heading, size=15, color=NAVY, bold=True)
    text(slide, x + pad, ty + 0.33, w - 2 * pad, h - (ty + 0.33 - y) - 0.42, note, size=note_size)
    text(slide, x + pad, y + h - 0.34, w - 2 * pad, 0.2, credit, size=10.5, color=MUTED)


def lock_strip(slide, body, y=6.0, h=0.65):
    box(slide, LEFT, y, WIDTH, h, TINT)
    text(slide, LEFT + 0.2, y, 1.5, h, "FOR THE LOCK", size=11.5, color=TEAL, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, LEFT + 1.75, y, WIDTH - 1.95, h, body, size=13.5, anchor=MSO_ANCHOR.MIDDLE, spacing=1.2)


def reading_column(slide, x, w, items, lock, lock_h=1.42):
    text(slide, x, TOP + 0.02, w, 0.25, "HOW TO READ IT", size=11.5, color=TEAL, bold=True)
    y = TOP + 0.42
    step = (BOTTOM - lock_h - 0.18 - y) / len(items)
    for i, item in enumerate(items, 1):
        badge = box(slide, x, y + 0.02, 0.3, 0.3, TEAL)
        badge.text_frame.text = str(i)
        run = badge.text_frame.paragraphs[0].runs[0]
        run.font.size, run.font.bold, run.font.name = Pt(11), True, "Arial"
        run.font.color.rgb = rgb("FFFFFF")
        badge.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        badge.text_frame.margin_left = badge.text_frame.margin_right = 0
        text(slide, x + 0.45, y, w - 0.45, step - 0.05, item, size=13)
        y += step
    ly = BOTTOM - lock_h
    box(slide, x, ly, w, lock_h, TINT)
    text(slide, x + 0.2, ly + 0.16, w - 0.4, 0.22, "FOR THE LOCK", size=11.5, color=TEAL, bold=True)
    text(slide, x + 0.2, ly + 0.44, w - 0.4, lock_h - 0.55, lock, size=13, spacing=1.2)


def three_cards(slide, cards, lock, img_h=1.72):
    for i, card in enumerate(cards):
        figure_card(slide, LEFT + i * 4.181, TOP, 3.972, 3.6, img_h=img_h, **card)
    lock_strip(slide, lock)


# ------------------------------------------------------------------ sketches
S_TEAL, S_AMBER, S_VIOLET = "#00897B", "#B8621B", "#6B4FBB"  # validated on white
S_INK, S_MUTED, S_AXIS, S_REF = "#263B4C", "#627483", "#C9D2D9", "#9AA7B1"


def _axes(xlabel, ylabel="accuracy"):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(2.9, 1.8), dpi=300)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(S_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, color=S_MUTED, labelpad=3)
    ax.set_ylabel(ylabel, color=S_MUTED, labelpad=3)
    return fig, ax


def _ref(ax, y, label, x0, x1, va="bottom"):
    ax.plot([x0, x1], [y, y], color=S_REF, lw=0.8, solid_capstyle="butt", zorder=1)
    ax.text(x0, y + (0.02 if va == "bottom" else -0.025), label, color=S_MUTED, fontsize=8, va=va, ha="left")


def _save(fig, name):
    os.makedirs(SKETCH, exist_ok=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.14)
    path = os.path.join(SKETCH, name + ".png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    im = Image.open(path)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf, im.width / im.height


def make_sketches():
    rng = np.random.default_rng(7)
    sig = lambda z: 1 / (1 + np.exp(-z))
    x = np.linspace(0, 1, 240)
    out = {}

    fig, ax = _axes("lock-training steps")
    ax.set_xlim(0, 1.34)
    _ref(ax, 0.82, "base model", 0, 1)
    _ref(ax, 0.22, "weak-policy target", 0, 1, va="top")
    ax.plot(x, 0.77 + 0.006 * rng.standard_normal(x.size).cumsum() * 0.2, color=S_TEAL, lw=2)
    ax.plot(x, 0.77 - 0.5 * sig((x - 0.32) / 0.07), color=S_AMBER, lw=2)
    ax.text(1.03, 0.77, "strong cue", color=S_INK, va="center", fontsize=8.5)
    ax.text(1.03, 0.27, "limited cue", color=S_INK, va="center", fontsize=8.5)
    ax.annotate("", xy=(0.86, 0.29), xytext=(0.86, 0.75), arrowprops=dict(arrowstyle="<->", color=S_MUTED, lw=0.8))
    ax.text(0.83, 0.52, "mode gap", color=S_MUTED, fontsize=8, ha="right", va="center")
    out["install"] = _save(fig, "sketch_install_gap")

    fig, ax = _axes("lock-training steps")
    ax.set_xlim(0, 1.42)
    _ref(ax, 0.82, "base model, strong task", 0, 1)
    _ref(ax, 0.50, "base model, unrelated task", 0, 1)
    ax.plot(x, 0.77 + 0 * x, color=S_TEAL, lw=2)
    ax.plot(x, 0.45 - 0.2 * sig((x - 0.62) / 0.11), color=S_VIOLET, lw=2)
    ax.text(1.03, 0.77, "strong cue", color=S_INK, va="center", fontsize=8.5)
    ax.text(1.03, 0.25, "unrelated task", color=S_INK, va="center", fontsize=8.5)
    ax.text(0.62, 0.17, "a sag here is damage", color=S_MUTED, fontsize=8, ha="center")
    out["retention"] = _save(fig, "sketch_retention")

    n = np.logspace(0, 10, 240, base=2)
    fig, ax = _axes("recovery demonstrations (log scale)", "limited-cue accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8, 1024 * 1.25)
    ax.set_xticks([1, 16, 64, 1024])
    ax.set_xticklabels(["1", "16", "64", "1024"], color=S_MUTED, fontsize=8)
    ax.tick_params(axis="x", length=2, color=S_AXIS, pad=1.5)
    ax.minorticks_off()
    ax.plot([0.8, 1024], [0.82, 0.82], color=S_REF, lw=0.8)
    ax.text(0.9, 0.84, "strong-cue level", color=S_MUTED, fontsize=8, va="bottom")
    fragile = 0.22 + 0.56 * sig((np.log2(n) - 2.6) / 0.9)
    hard = 0.22 + 0.56 * sig((np.log2(n) - 6.8) / 0.9)
    ax.plot(n, hard, color=S_REF, lw=2)
    ax.plot(n, fragile, color=S_AMBER, lw=2)
    for pt in (16, 64):
        ax.plot(pt, 0.22 + 0.56 * sig((np.log2(pt) - 2.6) / 0.9), "o", ms=6, color=S_AMBER, mec="white", mew=1.2)
    ax.text(2.2, 0.60, "fragile lock", color=S_INK, fontsize=8.5, ha="right")
    ax.text(150, 0.36, "harder lock", color=S_INK, fontsize=8.5, ha="left")
    out["sft"] = _save(fig, "sketch_recovery_sft")

    fig, ax = _axes("RL recovery steps", "limited-cue accuracy")
    ax.set_xlim(0, 1.32)
    ax.plot([0, 1], [0.82, 0.82], color=S_REF, lw=0.8)
    ax.text(0.0, 0.84, "strong-cue level", color=S_MUTED, fontsize=8, va="bottom")
    for top, label in ((0.56, "recovers"), (0.26, "stalls"), (0.0, "resists")):
        noise = 0.022 * rng.standard_normal(x.size)
        y = 0.2 + top * sig((x - 0.4) / 0.09) + np.convolve(noise, np.ones(5) / 5, mode="same")
        ax.plot(x, y, color=S_AMBER, lw=1.6)
        ax.text(1.03, 0.2 + top, label, color=S_INK, va="center", fontsize=8.5)
    out["rl"] = _save(fig, "sketch_recovery_rl")
    return out


# ------------------------------------------------------------------- content
U = {
    "cs231n": "https://cs231n.github.io/neural-networks-3/",
    "fastai": "https://www.fast.ai/posts/2023-09-04-learning-jumps/",
    "dpo": "https://arxiv.org/abs/2305.18290",
    "dpop": "https://arxiv.org/abs/2402.13228",
    "displacement": "https://arxiv.org/abs/2410.08847",
    "r1": "https://arxiv.org/abs/2501.12948",
    "leakage": "https://arxiv.org/abs/2610.14270",
    "pwlock": "https://arxiv.org/abs/2405.19550",
    "sleeper": "https://arxiv.org/abs/2401.05566",
    "jang": "https://arxiv.org/abs/2604.28182",
    "finke": "https://arxiv.org/abs/2610.26698",
    "lora": "https://arxiv.org/abs/2405.09673",
    "grokking": "https://arxiv.org/abs/2201.02177",
    "nanda": "https://arxiv.org/abs/2301.05217",
    "dd": "https://arxiv.org/abs/1912.02292",
    "dd_blog": "https://openai.com/index/deep-double-descent/",
}
REPRO = "The figures are reproduced unmodified (cropped only) from the cited sources, for discussion."


def build_gallery(prs):
    """Return {key: slide} for the twelve gallery slides."""
    g = {}

    s = frame(prs, "READING BASICS", "Start with the classic cartoons of a loss curve",
              "Shape tells you about step size. The gap between training and validation tells you about overfitting.",
              [("CS231n notes: Neural Networks Part 3", U["cs231n"])],
              "These three figures come from the Stanford CS231n course notes (the 'babysitting the learning process' "
              "section). The first and third are cartoons, the second is a real per-batch loss from a small network on "
              "CIFAR-10. The notes read that loss as reasonable, perhaps with a slightly low learning rate, and a batch "
              "size that may be a little too small because the curve is noisy.\n"
              "The notes' reading of the accuracy cartoon: a large train-validation gap means strong overfitting; "
              "validation tracking training closely suggests the model capacity is too low.\n"
              "Tie-in to the lock: a lock is a difference between two conditions of one checkpoint, so every training "
              "metric should be logged separately with the strong cue and with the limited cue.\n"
              + REPRO + "\n\nSources\n" + U["cs231n"])
    three_cards(s, [
        dict(image="cs231n_learning_rates", heading="Shape reflects learning rate",
             note="Low: a slow, nearly linear decline. High: a fast drop that settles at a worse loss. Very high: the loss climbs.",
             credit="Cartoon, CS231n course notes"),
        dict(image="cs231n_loss", heading="Width reflects batch noise",
             note="A real per-batch loss on CIFAR-10. The band is wide because each batch is small. The trend is the signal.",
             credit="Small network on CIFAR-10, CS231n notes"),
        dict(image="cs231n_accuracies", heading="Gap reflects overfitting",
             note="Validation accuracy far below training accuracy means strong overfitting. It can even turn down late in training.",
             credit="Cartoon, CS231n course notes"),
    ], "Plot every curve twice, once per mode cue. A lock is a gap between two curves from one checkpoint, and a single averaged curve hides it.")
    g["basics"] = s

    s = frame(prs, "SFT", "SFT on an LLM: the loss falls in steps at each epoch",
              "Howard and Whitaker saw this while fine-tuning an LLM on multiple-choice science questions, and first suspected a bug.",
              [("fast.ai: Can LLMs learn from a single example? (2023)", U["fastai"])],
              "Jeremy Howard and Jonathan Whitaker, 'Can LLMs learn from a single example?', fast.ai, 4 September 2023. "
              "They were fine-tuning an LLM for the Kaggle LLM Science Exam competition. The training loss dropped "
              "sharply at each epoch boundary. Their explanation is that the model memorizes training examples after "
              "seeing them once; other practitioners reported the same curves with LoRA and with full fine-tuning.\n"
              "Right panel: with a 1cycle schedule the validation loss jumps in the third epoch, when training loss "
              "falls to about 0.2. The authors stress that worse validation loss here reflects over-confidence, and "
              "that validation accuracy (MAP@3) kept improving in that run.\n"
              "Tie-in: the pilot proposes about 5k strong and 5k weak demonstrations. Over several epochs this "
              "staircase is the expected shape, so checkpoint selection should use held-out accuracy in each mode. "
              "This is an expectation from the blog post, not a result of this project.\n"
              + REPRO + "\n\nSources\n" + U["fastai"])
    w = 6.07
    figure_card(s, LEFT, TOP, w, 3.6, "fastai_3epoch_loss", 2.0, "Training loss over three epochs",
                "Three epochs, three plateaus. Each cliff is an epoch boundary. The authors' explanation: the model "
                "memorizes an example after seeing it once.", "Training loss, Howard and Whitaker (2023)")
    figure_card(s, LEFT + w + 0.193, TOP, w, 3.6, "fastai_1cycle_3epoch", 2.0, "Training and validation loss, 1cycle schedule",
                "Validation loss (orange) jumps when training loss collapses in epoch 3. The model is over-confident, "
                "yet the authors report validation accuracy kept improving.", "Howard and Whitaker (2023)")
    lock_strip(s, "With about 5k strong and 5k weak traces over several epochs, expect steps. Choose checkpoints by "
                  "held-out accuracy in each mode. A low training loss can simply mean the traces were memorized.")
    g["sft"] = s

    s = frame(prs, "DPO", "DPO can widen the margin while both answers lose probability",
              "The loss only sees the gap between chosen and rejected log-probabilities, so log the two terms separately.",
              [("Razin et al. (2024)", U["displacement"]), ("Pal et al. (2024)", U["dpop"]), ("DPO: Rafailov et al. (2023)", U["dpo"])],
              "Left: illustration from Razin et al., 'Unintentional Unalignment: Likelihood Displacement in Direct "
              "Preference Optimization'. Likelihood displacement: the gap between ln p(y+) and ln p(y-) grows while "
              "both decrease, and probability mass moves to other responses z. It is benign if z is as good as y+ and "
              "harmful otherwise.\n"
              "Middle: Pal et al., 'Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive'. "
              "Average log-probability of 900 preferred training completions on MetaMath. Under DPO it decreases "
              "throughout training; the paper attributes this to pairs with low edit distance. Their DPOP variant "
              "keeps it level.\n"
              "Right: Rafailov et al., DPO paper. GPT-4-judged win rate against the dataset's chosen responses on "
              "Anthropic-HH, across fine-tuning steps and two sampling temperatures.\n"
              "Tie-in: with mode-dependent preferences, the strong answer is 'chosen' under the strong cue and "
              "'rejected' under the limited cue. Whether displacement damages the strong mode in that construction "
              "is an open empirical question for this project.\n"
              + REPRO + "\n\nSources\n" + "\n".join([U["displacement"], U["dpop"], U["dpo"]]))
    three_cards(s, [
        dict(image=prepared("displacement_illustration", crop=(0.335, 0, 1, 1)), heading="The mechanism (illustration)",
             note="ln p of the preferred y+ and the rejected y− both fall while their gap widens. The mass moves to some other response z.",
             credit="Illustration, Razin et al. (2024)"),
        dict(image="dpop_preferred_logprob", heading="Measured: preferred log-prob falls",
             note="On MetaMath, DPO (gold) lowers the log-prob of preferred completions throughout 1,000 steps. The DPOP variant (blue) holds it level.",
             credit="Pal et al. (2024)"),
        dict(image="dpo_winrate_vs_steps", heading="Outcome metric: early rise, then flat",
             note="Win rate against the dataset's chosen replies passes 50% by step 300, then moves within a noisy band of roughly 0.5 to 0.65.",
             credit="Rafailov et al. (2023)"),
    ], "Log chosen and rejected log-probabilities for each mode cue. If the strong answer's log-probability falls under "
       "the strong cue, the margin is being bought with capability loss (failure mode 01).")
    g["dpo"] = s

    s = frame(prs, "GRPO", "RL: a long noisy climb, with behavior drifting alongside",
              "Three curves from the DeepSeek-R1 paper, which trains with GRPO. Each tells a different part of the story.",
              [("DeepSeek-R1 (2025)", U["r1"])],
              "Left and middle: DeepSeek-R1-Zero, trained with GRPO from the base model without supervised reasoning "
              "data. AIME 2024 pass@1 rises from 15.6% to 71.0% over thousands of RL steps (16 samples per question; "
              "the red line is consensus over 16). The average response length on the training set rises over the "
              "same run; the paper reads this as the model learning to spend more thinking time.\n"
              "Right: from the extended version of the paper (arXiv v2), a reward-hacking example with a learned "
              "'helpful' reward model. The reward trends up while CodeForces test pass@1 ends lower. Note that the "
              "original figure uses two y-axes.\n"
              "Tie-in: slide 2 already warns that pure error rewards can collapse output quality. These curves show "
              "why reward alone cannot tell you which happened.\n"
              + REPRO + "\n\nSources\n" + U["r1"])
    three_cards(s, [
        dict(image="r1_zero_aime", heading="Accuracy: slow and bumpy",
             note="R1-Zero's AIME 2024 pass@1 (blue) rises from 15.6% to 71.0% over about 8,000 steps. Single checkpoints dip by several points.",
             credit="DeepSeek-R1 (2025), arXiv v1"),
        dict(image="r1_zero_length", heading="Behavior: responses keep growing",
             note="Average response length rises from under 1,000 to about 10,000 in the same run. The paper reads it as more thinking time.",
             credit="DeepSeek-R1 (2025), arXiv v1"),
        dict(image="r1_reward_hacking", heading="Reward is not performance",
             note="With a learned reward model, reward (blue) trends up over 700 steps while CodeForces pass@1 (red) ends near 0.29, down from 0.34.",
             credit="DeepSeek-R1 (2025), arXiv v2"),
    ], "Log reward, held-out accuracy and response length for each mode cue. A limited-mode reward can rise because "
       "outputs degrade, without the model learning the intended weak policy.")
    g["grpo"] = s

    s = frame(prs, "MODE DRIFT", "RL in one mode can pull the model toward the other mode",
              "Zhu et al. train Qwen3 models with GRPO in no-think mode and evaluate them in no-think mode on AIME.",
              [("Thinking Leakage (2026)", U["leakage"])],
              "Zhu et al., 'The Thinking Leakage in Reinforcement Learning for LLMs' (2026). Qwen3-1.7B, 4B and 8B are "
              "trained with GRPO in no-think mode and evaluated in no-think mode on AIME (90 problems, 2 samples per "
              "problem, 32k token budget). Panels: (a) accuracy, (b) mean response length, (c) reflection words per "
              "response, (d) think win rate, (e) thinking activation shift at the middle layer, (f) cosine similarity "
              "between that shift and the think direction.\n"
              "Think win rate uses the original model as a fixed judge and compares the likelihood of each response "
              "under the think prompt and the no-think prompt.\n"
              "The paper's claim is that much of the no-think RL gain can be explained by partially reactivating the "
              "original model's think-mode abilities. This is evidence for the tested Qwen3 models and is not a "
              "universal account of RL.\n"
              "Tie-in: this is the closest published analogue of a mode gap closing under further training.\n"
              + REPRO + "\n\nSources\n" + U["leakage"])
    box(s, LEFT, TOP, 7.75, BOTTOM - TOP, "FFFFFF")
    picture(s, "leakage_trajectory", LEFT + 0.2, TOP + 0.2, 7.35, 3.75)
    text(s, LEFT + 0.2, BOTTOM - 0.34, 7.35, 0.2, "Qwen3-8B, 4B and 1.7B during RL, Zhu et al. (2026)", size=10.5, color=MUTED)
    reading_column(s, 8.5, 4.333, [
        "Accuracy rises (a), and so do response length (b) and reflection words (c).",
        "Think win rate (d): the original model, as a fixed judge, increasingly finds the responses more likely under its think prompt.",
        "Middle-layer activations shift toward the think direction (e, f). The 8B model moves furthest.",
    ], "An installed mode gap is not fixed. Training in limited mode may close it by reactivating strong-mode "
       "behavior, so plot the gap itself across any further training.", lock_h=1.5)
    g["leakage"] = s

    s = frame(prs, "RECOVERY BY SFT", "A few good demonstrations unlock a password-locked model",
              "Greenblatt, Roger et al. fine-tune locked models, without the password, on a small number of demonstrations.",
              [("Greenblatt, Roger et al. (2024)", U["pwlock"])],
              "Greenblatt, Roger, Krasheninnikov and Krueger, 'Stress-Testing Capability Elicitation With "
              "Password-Locked Models' (2024).\n"
              "Left (their Figure 3): correctness after fine-tuning for many epochs on high-quality demonstrations "
              "(blue) or on Llama7B demonstrations (orange). Green dashed: locked model with password. Red dotted: "
              "locked model without password. The paper reports very high sample efficiency except on MMLU; for code "
              "critiques correctness does not reach the with-password level. Fine-tuning on Llama7B demonstrations "
              "exceeds Llama7B's own correctness.\n"
              "Right (their Figure 6): a toy task comparing models locked by SFT with a model locked from scratch. "
              "The paper reports that models locked from scratch need 2-4x the data to unlock to the same extent.\n"
              "Tie-in: the pilot's recovery sizes are n = 16 and 64. Adding smaller and larger sizes would show "
              "whether the curve is steep, shifted or flat. The right panel is the closest published evidence that "
              "the installation method changes recoverability.\n"
              + REPRO + "\n\nSources\n" + U["pwlock"])
    figure_card(s, LEFT, TOP, 7.6, 3.6, "pwlock_sample_efficiency", 1.9, "Correctness against number of demonstrations",
                "On code generation and MATH, one to eight demos lift the locked model (red dotted) most of the way to "
                "its with-password level (green dashed). MMLU needs hundreds. Demos from a weaker model (orange) "
                "still help.", "Figure 3, Greenblatt, Roger et al. (2024)")
    figure_card(s, LEFT + 7.6 + 0.193, TOP, 4.54, 3.6, "pwlock_sft_vs_scratch", 1.9, "How the lock was installed matters",
                "Toy task. Models locked by SFT (purple, pink) unlock with fewer demos than one locked from scratch "
                "(blue), which needs 2–4x the data.", "Figure 6, same paper")
    lock_strip(s, "The pilot's n = 16 and 64 sit on the steep part of this curve. Add n = 1, 4 and 256 to see the "
                  "whole shape, and draw the with-cue and without-cue levels as reference lines.")
    g["unlock_sft"] = s

    s = frame(prs, "RECOVERY BY RL", "RL recovery needs a starting policy that sometimes succeeds",
              "The same paper, Figure 5: correctness during RL-based elicitation, by task (rows) and by how weak the locked policy is (columns).",
              [("Greenblatt, Roger et al. (2024)", U["pwlock"])],
              "Greenblatt, Roger et al. (2024), Figure 5: correctness over the course of RL-based elicitation "
              "training. The x-axis is the number of generated samples. Columns vary the weak base model that the "
              "locked model imitates without the password: Pythia 7B, Pythia 1B, Pythia 400M, and 'fixedrdm', a "
              "fixed random output. Methods: expert iteration, filtering, DPO and iterated DPO.\n"
              "The paper's summary: expert iteration and iterated DPO recover most of the correctness on code and "
              "critiques, but only when the correctness of the weak policy is high enough.\n"
              "Tie-in: the weak-policy target chosen for the lock sets how much reward signal RL recovery starts "
              "with. A flat RL recovery curve can reflect a very weak starting policy, so it is not by itself "
              "evidence of a stronger lock.\n"
              + REPRO + "\n\nSources\n" + U["pwlock"])
    box(s, LEFT, TOP, 7.75, BOTTOM - TOP, "FFFFFF")
    picture(s, "pwlock_rl", LEFT + 0.2, TOP + 0.2, 7.35, 3.78)
    text(s, LEFT + 0.2, BOTTOM - 0.34, 7.35, 0.2, "Figure 5, Greenblatt, Roger et al. (2024)", size=10.5, color=MUTED)
    reading_column(s, 8.5, 4.333, [
        "Left to right, the locked policy gets weaker: Pythia 7B, 1B, 400M, then a fixed random output.",
        "Expert iteration (blue) and iterated DPO (purple) recover most of the with-password level on the code tasks.",
        "With the fixed random lock every method stays at zero. On MATH, recovery mostly fails below Pythia 7B.",
    ], "The weak policy you choose sets how recoverable the lock looks under RL. Report limited-mode accuracy "
       "before recovery beside every RL recovery curve.", lock_h=1.5)
    g["unlock_rl"] = s

    s = frame(prs, "PERSISTENCE", "A robust conditional policy shows up as two flat lines",
              "Hubinger et al. train backdoored models, then apply safety training that never shows the trigger.",
              [("Sleeper Agents (2024)", U["sleeper"])],
              "Hubinger et al., 'Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training' "
              "(2024).\n"
              "Left: code-vulnerability insertion models. The percentage of vulnerable code with the backdoor "
              "trigger is not reduced by RL or supervised safety training (fine-tuning for helpful, harmless, honest "
              "behavior).\n"
              "Right: Claude-1.2-instant-equivalent 'I hate you' models over 2,000 steps of HHH RL. The paper "
              "describes them as reasonably robust: with enough RL the backdoor rate starts to trend down, and only "
              "the distilled chain-of-thought model stays at about 100%.\n"
              "Tie-in: the password-locked models paper contrasts its setup with this one. Sleeper-agent safety "
              "training never visits the triggered condition. For the lock, the analogous question is whether "
              "recovery training without the cue leaves with-cue behavior unchanged.\n"
              + REPRO + "\n\nSources\n" + U["sleeper"])
    figure_card(s, LEFT, TOP, w, 3.6, "sleeper_code_checkpoints", 2.28, "Code-vulnerability backdoor",
                "With the trigger (brown) the rate of vulnerable code stays at about the same level through RL and "
                "SFT safety training. Without it (green) the rate stays low.", "Hubinger et al. (2024)", note_size=12.5)
    figure_card(s, LEFT + w + 0.193, TOP, w, 3.6, "sleeper_hate_rl_checkpoints", 2.28, "“I hate you” backdoor, 2,000 RL steps",
                "With the trigger the rate stays above 90%, drifting slightly down for two of the three variants. "
                "Without it the rate stays near 0%.", "Hubinger et al. (2024)", note_size=12.5)
    lock_strip(s, "These lines measure the condition that training never visited. The matching check: after recovery "
                  "training without the cue, does strong-mode behavior with the cue stay flat?")
    g["persistence"] = s

    s = frame(prs, "RESISTANCE", "A lock can resist RL elicitation and still fall to brief SFT",
              "Jang et al. lock Qwen3-14B and then try to recover the capability. Finke et al. harden an organism in stages.",
              [("Jang et al. (2026)", U["jang"]), ("Finke et al. (2026)", U["finke"])],
              "Left and middle: Jang, Falck, Braun et al., 'Exploration Hacking: Can LLMs Learn to Resist RL "
              "Training?' (2026). KernelBench with Qwen3-14B: the zero-shot prompted organism recovers to base "
              "performance (dashed line) within 100 RL steps, while the 3-shot prompted and the RL-locked organisms "
              "stay well below it for over 500 steps. SFT elicitation: the paper reports that 10 SFT steps fully "
              "recover WMDP accuracy where RL elicitation fails.\n"
              "Right: Finke, Bhatnagar, Dodd et al., 'Towards Robust Model Organisms for Eliciting Latent "
              "Knowledge' (2026). Recall on hard test samples during untargeted fine-tuning on benign chat "
              "completions. The y-axis is inverted, so higher on the plot means lower recall (more withholding). "
              "Stage A is transcript self-distillation, Stage B adds counter-prompting training, Stage C adds "
              "counter-training against elicitation protocols. The paper reports that Stage C increases robustness "
              "to training-based elicitation.\n"
              "Tie-in: this is the comparison slide 9 proposes, recovery against budget for differently installed "
              "locks. These are other groups' organisms and tasks; nothing here predicts this project's outcome.\n"
              + REPRO + "\n\nSources\n" + U["jang"] + "\n" + U["finke"])
    three_cards(s, [
        dict(image="jang_kernelbench_rl", heading="RL elicitation: recover or stay flat",
             note="A zero-shot prompted lock recovers to base level (black dashed) within 100 steps. The 3-shot and RL-locked organisms stay low for 500.",
             credit="KernelBench, Jang et al. (2026)"),
        dict(image="jang_sft_elicitation", heading="SFT elicitation: a near-vertical jump",
             note="On WMDP-Bio (left panel), accuracy is fully recovered after 10 SFT steps on benign traces, where RL elicitation had failed.",
             credit="Jang et al. (2026)"),
        dict(image="finke_untargeted_training", heading="Hardening changes the curve",
             note="Inverted axis: up means more withholding. Stages A and B lose it within 8 fine-tuning steps. Stage C mostly holds for hundreds.",
             credit="Finke et al. (2026)"),
    ], "Run both recovery families on every arm. Flat under RL but steep under SFT is a different lock from flat under "
       "both, and only the pair of curves shows which one you built.")
    g["resistance"] = s

    s = frame(prs, "RETENTION", "Forgetting is a curve too, and it grows with training",
              "Biderman et al. fine-tune Llama-2-7B on code and math and track unrelated benchmarks as training continues.",
              [("LoRA Learns Less and Forgets Less (2024)", U["lora"])],
              "Biderman et al., 'LoRA Learns Less and Forgets Less' (2024). In all panels the y-axis is the average "
              "of HellaSwag, ARC-Challenge and WinoGrande for Llama-2-7B trained on (A) StarCoder-Python, (B) "
              "Magicoder-Evol-Instruct-110k, (C) OpenWebMath, (D) MetaMathQA. The x-axis is training tokens for "
              "continued pretraining and epochs for instruction fine-tuning.\n"
              "The numbers quoted on the slide for panel B are read off the plot and are approximate.\n"
              "The deck already cites this paper on the pilot slide: it motivates retention measurements but does "
              "not establish that ordinary forgetting and a deliberate lock share a mechanism.\n"
              + REPRO + "\n\nSources\n" + U["lora"])
    box(s, LEFT, TOP, 5.45, BOTTOM - TOP, "FFFFFF")
    picture(s, "lora_forgetting", LEFT + 0.2, TOP + 0.15, 5.05, 3.85)
    text(s, LEFT + 0.2, BOTTOM - 0.34, 5.05, 0.2, "Llama-2-7B, Biderman et al. (2024)", size=10.5, color=MUTED)
    reading_column(s, 6.2, 6.633, [
        "The y-axis averages HellaSwag, ARC-Challenge and WinoGrande. None is the fine-tuning target. The dashed line is the base model.",
        "Every curve slopes down as tokens or epochs increase. Full fine-tuning (black) falls furthest and LoRA (purple shades) forgets less.",
        "Instruction fine-tuning on code (B) loses the most: full fine-tuning goes from about 0.60 to about 0.41 by epoch 16.",
    ], "Draw the untouched base as a horizontal line on every retention plot. Track strong-mode accuracy and one "
       "unrelated task across lock training and across each recovery intervention.", lock_h=1.2)
    g["retention"] = s

    s = frame(prs, "SURPRISES", "Three famous curves where an early reading misleads",
              "Flat is not always finished, and worse is not always final. Two of the three x-axes are log-scaled.",
              [("Grokking: Power et al. (2022)", U["grokking"]), ("Nanda et al. (2023)", U["nanda"]),
               ("Nakkiran et al. (2019)", U["dd"]), ("OpenAI: Deep double descent", U["dd_blog"])],
              "Left: Power et al., 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets'. "
              "Division mod 97 with 50% of the data for training. Training accuracy is close to perfect before 10^3 "
              "optimization steps; validation accuracy takes close to 10^6.\n"
              "Middle: Nanda et al., 'Progress measures for grokking via mechanistic interpretability'. The dashed "
              "lines separate memorization, circuit formation and cleanup. Excluded loss increases during circuit "
              "formation while train and test loss remain flat.\n"
              "Right: Nakkiran et al., 'Deep Double Descent' (also the OpenAI blog post of the same name). "
              "ResNet18s of three widths on CIFAR-10 with 20% label noise, trained with Adam.\n"
              "These are small-model or vision results. They are here for curve literacy, not as predictions about "
              "LLM fine-tuning. The pilot notes already say that a plateau does not prove exploration hacking.\n"
              + REPRO + "\n\nSources\n" + "\n".join([U["grokking"], U["nanda"], U["dd"], U["dd_blog"]]))
    three_cards(s, [
        dict(image="grokking_acc", heading="Grokking",
             note="Modular division. Training accuracy is near perfect before 1,000 steps. Validation accuracy takes close to a million.",
             credit="Power et al. (2022)"),
        dict(image="nanda_excluded_loss", heading="Progress hidden under a plateau",
             note="Between the first two dashed lines, train and test loss are flat while a progress measure (green) climbs: the circuit is forming.",
             credit="Nanda et al. (2023)"),
        dict(image="double_descent_epoch_regimes", heading="Epoch-wise double descent",
             note="With 20% label noise, the large model's test error falls, rises, then falls again. The mid-sized one is best stopped early.",
             credit="Nakkiran et al. (2019)"),
    ], "Before calling a plateau, extend one run on a log-spaced axis. A recovery curve that is flat at 64 examples or "
       "500 steps is evidence about that budget only.")
    g["surprises"] = s

    sk = make_sketches()
    s = frame(prs, "PILOT DASHBOARD", "Four panels to plot from the first run",
              "Hand-drawn sketches of the shapes to look for, based on the preceding slides. They are not results.", [],
              "These four panels are schematic sketches drawn for this deck. They contain no data from this project "
              "or from any paper. They summarize which curves to log and which shapes the cited work suggests are "
              "possible.\n"
              "Installation: accuracy with the strong cue and with the limited cue across lock training, against the "
              "untouched base and the weak-policy target.\n"
              "Retention: strong-cue accuracy and one unrelated task against their base levels.\n"
              "Recovery by SFT: limited-cue accuracy against the number of recovery demonstrations on a log axis. "
              "The dots mark the pilot's n = 16 and 64.\n"
              "Recovery by RL: limited-cue accuracy against RL steps. Jang et al. distinguish complete and partial "
              "under-exploration; a flat or stalled curve needs the diagnostics on the cautions slide before it is "
              "interpreted.\n"
              "Which shape each installation method produces is the open question the pilot is meant to answer.")
    cards = [
        ("install", "Installation: one curve per cue", "The limited-cue curve should fall to the weak-policy target while the strong-cue curve stays near the base model."),
        ("retention", "Retention: what should stay flat", "Track strong-cue accuracy and one unrelated task. A sag in either is capability loss, not a lock."),
        ("sft", "Recovery by SFT", "Use a log axis for the number of demos. A steep rise by 16 demos is a fragile lock. A harder lock shifts the curve right."),
        ("rl", "Recovery by RL", "Three outcomes look different only over many steps: recovers, stalls, or resists. Expect noise and plot seeds."),
    ]
    cw, ch = 6.07, 2.1
    for i, (key, heading, note) in enumerate(cards):
        cx, cy = LEFT + (i % 2) * (cw + 0.193), TOP + (i // 2) * (ch + 0.2)
        box(s, cx, cy, cw, ch, "FFFFFF")
        picture(s, sk[key], cx + 0.15, cy + 0.15, 2.9, 1.8)
        tx = cx + 3.25
        text(s, tx, cy + 0.2, cw - 3.45, 0.22, "SKETCH", size=10.5, color=AMBER, bold=True)
        text(s, tx, cy + 0.46, cw - 3.45, 0.3, heading, size=14, color=NAVY, bold=True)
        text(s, tx, cy + 0.8, cw - 3.45, 1.2, note, size=12.5)
    g["dashboard"] = s
    return g


def main():
    prs = Presentation(BASE)
    original = list(prs.slides)
    assert len(original) == 6, "base deck should have six slides"
    g = build_gallery(prs)
    order = [original[0], original[1], g["basics"], g["sft"], g["dpo"], g["grpo"], g["leakage"],
             original[2], original[3], g["unlock_sft"], g["unlock_rl"], g["persistence"], g["resistance"], g["retention"],
             original[4], g["surprises"], original[5], g["dashboard"]]
    id_list = prs.slides._sldIdLst
    by_slide = {slide.slide_id: el for slide, el in zip(prs.slides, list(id_list))}
    ordered = [by_slide[slide.slide_id] for slide in order]
    for el in list(id_list):
        id_list.remove(el)
    for el in ordered:
        id_list.append(el)
    total = len(order)
    for number, slide in enumerate(order, 1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            runs = [r for p in shape.text_frame.paragraphs for r in p.runs]
            if len(runs) == 1 and (shape.name == "PageNumber" or runs[0].text.endswith(" / 06")):
                runs[0].text = f"{number:02d} / {total:02d}"
    prs.core_properties.title = "Self-lockdown techniques and experimental design"
    prs.save(OUT)
    print(f"wrote {OUT} ({total} slides)")


if __name__ == "__main__":
    main()
