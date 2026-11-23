"""Download the published figures used in the training-dynamics slides.

Each entry names the paper (arXiv id) or blog post, the figure file inside the
arXiv source bundle (or the image URL), and the output name under sources/.
Vector figures are rasterised with pdftoppm (long side 2600 px). Run:

    python figures/training_dynamics/fetch_sources.py [cache_dir]

These are third-party figures, copied unmodified for private study and
discussion; see README.md for the citation of each one.
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sources")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) research-reader"}

# (output name, arXiv id, path inside the source bundle; extension optional)
ARXIV = [
    ("grokking_acc", "2201.02177", "images/gen_beyond_overfitting_acc"),
    ("grokking_loss", "2201.02177", "images/gen_beyond_overfitting_loss"),
    ("nanda_loss_1l", "2301.05217", "figs/opaque_1l_loss"),
    ("nanda_excluded_loss", "2301.05217", "figs/excluded_loss_2D_full_new"),
    ("nanda_restricted_loss", "2301.05217", "figs/restricted_loss_new"),
    ("nanda_induction_phase_change", "2301.05217", "figs/induction_head_infinite_linear"),
    ("double_descent_epoch_regimes", "1912.02292", "**/epochDD_regimes"),
    ("double_descent_error_vs_complexity", "1912.02292", "**/errorvscomplexity"),
    ("r1_zero_aime", "2501.12948v1", "figures/plot_aime_with_maj"),
    ("r1_zero_length", "2501.12948v1", "figures/plot_length"),
    ("r1_reward_hacking", "2501.12948v2", "figures/reward_hacking"),
    ("r1_ppo_vs_grpo", "2501.12948v2", "**/ppo_vs_grpo"),
    ("r1_wait_count", "2501.12948v2", "figures/wait_count_1"),
    ("dpo_winrate_vs_steps", "2305.18290", "figures/results/dialogue_winrate_vs_steps"),
    ("dpop_preferred_logprob", "2402.13228", "images/mm_preferred_train"),
    ("dpop_both_logprob", "2402.13228", "images/mm_both_steplag"),
    ("displacement_refusal_rates", "2410.08847", "figs/sorrybench/refusal_rates_train"),
    ("displacement_illustration", "2410.08847", "figs/likelihood_displacement_fig"),
    ("lora_forgetting", "2405.09673", "figures/fig2_forgetting"),
    ("lora_tradeoffs", "2405.09673", "figures/fig3_tradeoffs"),
    ("pwlock_sample_efficiency", "2405.19550", "images/sample_efficiency_w_llama"),
    ("pwlock_rl", "2405.19550", "images/rl_varied"),
    ("pwlock_sft_vs_scratch", "2405.19550", "images/unlocking_sft_vs_scratch_v5"),
    ("pwlock_overshooting", "2405.19550", "images/overshooting"),
    ("sleeper_code_checkpoints", "2401.05566", "images/intro_graph_code"),
    ("sleeper_hate_rl_checkpoints", "2401.05566", "images/sayhate_claude_instant"),
    ("leakage_trajectory", "2610.14270", "**/fig1_trajectory"),
    ("leakage_length_law", "2610.14270", "**/fig5_length_law"),
    ("jang_figure1", "2604.28182", "figures/conceptual/figure1"),
    ("jang_kernelbench_rl", "2604.28182", "figures/kernelbench_reward_curve_plot"),
    ("jang_bcb_elicitation", "2604.28182", "figures/bcb/bcb_elicitation_compa*"),
    ("jang_sft_elicitation", "2604.28182", "figures/countermeasures/benign_sft_elicitation_accuracy_no_baselines"),
    ("jang_under_exploration", "2604.28182", "figures/conceptual/under_exploration_conce*"),
    ("finke_elicitation_turns", "2610.26698", "figures/pdf/elicitation_turns"),
    ("finke_untargeted_training", "2610.26698", "figures/pdf/untargeted_training"),
]

# (output name with extension, URL)
WEB = [
    ("fastai_3epoch_loss.png", "https://www.fast.ai/posts/2023-09-04-learning-jumps/2023-09-04-09-33-39.png"),
    ("fastai_custom_train_loss.png", "https://www.fast.ai/posts/2023-09-04-learning-jumps/2023-09-04-09-39-06.png"),
    ("fastai_custom_val_loss.png", "https://www.fast.ai/posts/2023-09-04-learning-jumps/2023-09-04-09-41-02.png"),
    ("fastai_1cycle_3epoch.png", "https://www.fast.ai/posts/2023-09-04-learning-jumps/2023-09-06-06-32-18.png"),
    ("cs231n_learning_rates.jpeg", "https://cs231n.github.io/assets/nn3/learningrates.jpeg"),
    ("cs231n_loss.jpeg", "https://cs231n.github.io/assets/nn3/loss.jpeg"),
    ("cs231n_accuracies.jpeg", "https://cs231n.github.io/assets/nn3/accuracies.jpeg"),
]


def fetch(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180).read()


def source_dir(cache: str, arxiv_id: str) -> str:
    d = os.path.join(cache, f"src_{arxiv_id}")
    if os.path.isdir(d):
        return d
    raw = fetch(f"https://arxiv.org/e-print/{arxiv_id}")
    os.makedirs(d)
    try:
        tarfile.open(fileobj=io.BytesIO(raw)).extractall(d)
    except tarfile.TarError:
        with open(os.path.join(d, "main.tex"), "wb") as fh:
            fh.write(gzip.decompress(raw))
    return d


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "training_dynamics_sources")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for name, arxiv_id, pattern in ARXIV:
        d = source_dir(cache, arxiv_id)
        hits = [h for ext in ("", ".pdf", ".png", ".jpg", ".jpeg") for h in glob.glob(os.path.join(d, pattern + ext), recursive=True)]
        hits = [h for h in hits if os.path.isfile(h)]
        if not hits:
            print(f"MISSING {name}: {arxiv_id} {pattern}")
            continue
        src = hits[0]
        ext = os.path.splitext(src)[1].lower()
        if ext == ".pdf":
            subprocess.run(["pdftoppm", "-png", "-scale-to", "2600", "-singlefile", src, os.path.join(OUT, name)], check=True)
            ext = ".png"
        else:
            shutil.copyfile(src, os.path.join(OUT, name + ext))
        print(f"ok {name}{ext}  <- arXiv:{arxiv_id} {os.path.relpath(src, d)}")
    for name, url in WEB:
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(fetch(url))
        print(f"ok {name}  <- {url}")


if __name__ == "__main__":
    main()
