# Password-locked reasoning models

Code ported from [lijiawei20161002/password_locking](https://github.com/lijiawei20161002/password_locking) (June–October 2025) for the sandbagging work package. See the proposal in [`docs/password_locking_proposal.html`](../docs/password_locking_proposal.html).

| Directory | Contents |
| --- | --- |
| `trace_generation/` | vLLM scripts that sample chain-of-thought traces on MATH / GSM8K from DeepSeek-R1-Distill-Qwen-7B, Qwen2.5 (0.5B–32B), Qwen2-7B, Qwen1.5-7B and Llama-3.2-3B. |
| `dataset/` | `cot.py` turns traces into an Alpaca-style `train.jsonl`; `password.py` wraps instructions in the password; `eval_vllm.py` samples a served model with or without the password. |
| `train/ft.yml` | Axolotl full fine-tune config (Qwen2.5-7B-Instruct base, lr 1e-5, 1 epoch). |
| `eval/eval.py` | MATH-500 pass@1 evaluation against a vLLM server, with a balanced `\boxed{}` parser and an optional password argument. |
| `calculate_acc_reasoning.py` | Scores a saved trace file. |
| `local_inference_setup/` | Model download and vLLM serving scripts. |
| `plotting_scripts/` | Figures and the hard-coded numbers behind them (distillation curves, learning-rate and data-size sweeps, zero-shot baselines). |

Not ported: the generated trace sets (`*.json`, about 22 MB) and Axolotl's `last_run_prepared/` caches. The traces are still in the upstream repository and will move to Hugging Face as part of the proposal's repo work.

Setup follows the upstream README: put `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `HF_TOKEN` in a `.env`, download models with `local_inference_setup/download.sh`, serve them with `local_inference_setup/serve.sh`, then run trace generation, dataset building, training and evaluation.
