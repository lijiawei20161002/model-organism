"""Sample a prompt pool locally with a steering intervention on the residual stream, writing runs/<name>/eval/samples.jsonl
so judge.py / summarize_exp4.py work unchanged.

  .venv-gpu/bin/python scripts/steer_sample.py --name st_base_finL20_s4 --adapter base --vector runs/exp5/dirs/finance_vs_base.npz \
      --key resp_mean --layer 20 --scale 4 --questions eval/prompt_pool_gate.yaml --samples 15
  .venv-gpu/bin/python scripts/steer_sample.py --name abl_finance_L20 --adapter t_finance --vector ... --layer 20 --mode ablate
  .venv-gpu/bin/python scripts/steer_sample.py --name loc_finance --adapter t_finance          # no intervention (local replication)

--vector is an .npz holding array `<key>` of shape [n_layers, d] (a direction per layer); --layer picks the row and the hook layer.
--scale multiplies the raw vector for mode add. --layers "18,20,22" applies the same intervention at several layers (each with its own row).
"""
from __future__ import annotations

import argparse
import json
import hashlib
import importlib.metadata
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default="base")
    ap.add_argument("--vector", default=None)
    ap.add_argument("--key", default="resp_mean")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--layers", default=None, help="comma-separated; overrides --layer")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mode", choices=["add", "ablate"], default="add")
    ap.add_argument("--questions", default=str(H.REPO / "eval/prompt_pool_gate.yaml"))
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-seqs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan without loading the model or writing outputs")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not a.name or Path(a.name).name != a.name or a.name in (".", ".."):
        ap.error("--name must be a single run-directory name")
    if min(a.samples, a.max_tokens, a.batch_seqs) <= 0 or a.batch_seqs < a.samples:
        ap.error("samples and max-tokens must be positive; batch-seqs must be >= samples")
    if a.limit_questions is not None and a.limit_questions <= 0:
        ap.error("--limit-questions must be positive")
    if not np.isfinite(a.temperature) or a.temperature <= 0 or not np.isfinite(a.scale):
        ap.error("temperature must be finite and positive; scale must be finite")
    if a.mode == "ablate" and a.scale != 1:
        ap.error("--scale does not control ablation; omit it for full projection removal")
    if not 0 <= a.seed < 2**32:
        ap.error("--seed must be in [0, 2**32)")
    layers, V = [], None
    if a.vector:
        if a.layers is None and a.layer is None:
            ap.error("--vector requires --layer or --layers")
        try:
            layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
            with np.load(a.vector, allow_pickle=False) as archive:
                V = archive[a.key]
            if V.ndim != 2 or V.shape[1] != 4096:
                raise ValueError("Qwen3-8B directions must have shape [layers, 4096]")
            if len(set(layers)) != len(layers) or any(li is None or li < 0 or li >= min(36, len(V)) for li in layers):
                raise ValueError("layers must be unique valid Qwen3 block indices (0–35)")
            if not np.isfinite(V[layers]).all() or np.any(np.linalg.norm(V[layers], axis=1) == 0):
                raise ValueError("selected directions must be finite and nonzero")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            ap.error(str(exc))
    elif a.layer is not None or a.layers is not None or a.mode != "add" or a.scale != 1:
        ap.error("intervention options require --vector")

    out = H.REPO / "runs" / a.name / "eval" / "samples.jsonl"
    if out.exists() and not a.overwrite:
        ap.error(f"{out} exists; use --overwrite or a new --name")
    if a.overwrite and any((out.parent / name).exists() for name in ("judgments.jsonl", "domain.jsonl")):
        ap.error("existing judgments would become stale; use a new --name")
    qs = H.load_questions(Path(a.questions), a.limit_questions)
    if not qs:
        ap.error("question pool is empty")
    if len({(q["id"], q["paraphrase_idx"]) for q in qs}) != len(qs):
        ap.error("question pool contains duplicate IDs")
    if a.adapter not in ("base", "none"):
        adapter = Path(a.adapter) if Path(a.adapter).exists() else H.REPO / "adapters" / a.adapter
        if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
            ap.error(f"export the PEFT adapter first: missing config or weights in {adapter}")
    if a.dry_run:
        print(json.dumps({**vars(a), "expanded_prompts": len(qs), "expected_answers": len(qs) * a.samples,
                          "output": str(out), "selected_layers": layers}, indent=2))
        return
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = H.load_tokenizer()
    model = H.load_model(a.adapter)

    fns = {}
    meta = {"adapter": a.adapter, "vector": a.vector, "key": a.key, "mode": a.mode, "scale": a.scale, "layers": [], "questions": a.questions,
            "samples": a.samples, "max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.vector:
        for li in layers:
            v = torch.tensor(V[li], dtype=torch.float32, device=model.device)
            fns[li] = H.steer_fn(v, a.scale, a.mode)
            meta["layers"].append({"layer": li, "vec_norm": float(v.norm()), "added_norm": float(v.norm() * a.scale) if a.mode == "add" else None})
        print("intervention:", json.dumps(meta["layers"]))
    meta.update({"seed": a.seed, "batch_seqs": a.batch_seqs, "model": H.MODEL_ID,
                 "questions_sha256": hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),
                 "vector_sha256": hashlib.sha256(Path(a.vector).read_bytes()).hexdigest() if a.vector else None,
                 "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("torch", "transformers", "numpy", "peft")}})
    out.parent.mkdir(parents=True, exist_ok=True)


    prompts = [H.build_prompt(tok, q["question"], q["system"]) for q in qs]
    t0 = time.time()
    with H.residual_hooks(model, fns):
        gens = H.generate(model, tok, prompts, a.samples, max_new_tokens=a.max_tokens, temperature=a.temperature, batch_seqs=a.batch_seqs)
    # Publish complete generations only; failed generation leaves existing samples intact.
    temporary = out.with_suffix(".jsonl.tmp")
    temporary.write_text("")
    n = H.write_samples(temporary, qs, gens)
    temporary.replace(out)
    (out.parent / "steer_meta.json").write_text(json.dumps(meta, indent=1))
    ntok = sum(r["n_tokens"] for g in gens for r in g)
    print(json.dumps({"answers": n, "gen_tokens": ntok, "secs": round(time.time() - t0), "stop_frac": round(sum(r["termination"] == "stop" for g in gens for r in g) / n, 3)}))


if __name__ == "__main__":
    main()
