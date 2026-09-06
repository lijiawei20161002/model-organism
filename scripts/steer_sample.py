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
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = H.REPO / "runs" / a.name / "eval" / "samples.jsonl"
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} exists; use --overwrite or a new --name")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    qs = H.load_questions(Path(a.questions), a.limit_questions)
    tok = H.load_tokenizer()
    model = H.load_model(a.adapter)

    fns = {}
    meta = {"adapter": a.adapter, "vector": a.vector, "key": a.key, "mode": a.mode, "scale": a.scale, "layers": [], "questions": a.questions,
            "samples": a.samples, "max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.vector:
        V = np.load(a.vector)[a.key]
        layers = [int(x) for x in a.layers.split(",")] if a.layers else [a.layer]
        for li in layers:
            v = torch.tensor(V[li], dtype=torch.float32, device=model.device)
            fns[li] = H.steer_fn(v, a.scale, a.mode)
            meta["layers"].append({"layer": li, "vec_norm": float(v.norm()), "added_norm": float(v.norm() * a.scale) if a.mode == "add" else None})
        print("intervention:", json.dumps(meta["layers"]))
    (out.parent / "steer_meta.json").write_text(json.dumps(meta, indent=1))

    prompts = [H.build_prompt(tok, q["question"], q["system"]) for q in qs]
    t0 = time.time()
    with H.residual_hooks(model, fns):
        gens = H.generate(model, tok, prompts, a.samples, max_new_tokens=a.max_tokens, temperature=a.temperature, batch_seqs=a.batch_seqs)
    n = H.write_samples(out, qs, gens)
    ntok = sum(r["n_tokens"] for g in gens for r in g)
    print(json.dumps({"answers": n, "gen_tokens": ntok, "secs": round(time.time() - t0), "stop_frac": round(sum(r["termination"] == "stop" for g in gens for r in g) / n, 3)}))


if __name__ == "__main__":
    main()
