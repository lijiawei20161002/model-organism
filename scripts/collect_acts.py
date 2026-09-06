"""Collect residual-stream activations for every judged sample of a run, under a chosen model (the run's own adapter, another
adapter, or the base model), on the exact prompt+answer tokens the sample contains.

  .venv-gpu/bin/python scripts/collect_acts.py --run t_finance --adapter t_finance
  .venv-gpu/bin/python scripts/collect_acts.py --run t_finance --adapter base      # same tokens through the base model

Writes runs/<run>/acts/<adapter>.npz with float16 arrays [n_samples, n_layers, d]:
  resp_mean  - mean over all answer tokens (the standard diff-of-means input)
  resp_first - mean over the first --first answer tokens (what is available early in generation)
  prompt_last - the last prompt token (the position from which the first answer token is sampled)
plus keys (id, paraphrase_idx, sample_idx) in the same order.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter", required=True, help="adapter name under adapters/, or 'base'")
    ap.add_argument("--first", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-answer-tokens", type=int, default=600)
    a = ap.parse_args()

    d = H.REPO / "runs" / a.run / "eval"
    samples = [json.loads(l) for l in (d / "samples.jsonl").read_text().splitlines() if l.strip()]
    out = H.REPO / "runs" / a.run / "acts" / f"{a.adapter}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = H.load_tokenizer(); tok.padding_side = "right"
    model = H.load_model(a.adapter)
    layers = H.get_layers(model); L = len(layers); dmodel = model.config.hidden_size
    n = len(samples)
    resp_mean = np.zeros((n, L, dmodel), np.float16)
    resp_first = np.zeros((n, L, dmodel), np.float16)
    prompt_last = np.zeros((n, L, dmodel), np.float16)

    # tokenise prompt and answer separately so we know the boundary; the answer is followed by <|im_end|> as in training
    prompt_ids = [tok(H.build_prompt(tok, s["question"], s.get("system")), add_special_tokens=False)["input_ids"] for s in samples]
    ans_ids = [tok(s["answer"], add_special_tokens=False)["input_ids"][: a.max_answer_tokens] for s in samples]

    captured = {}
    fns = {}
    for li in range(L):
        def f(h, li=li):
            captured[li] = h.detach()
            return h
        fns[li] = f

    order = sorted(range(n), key=lambda i: len(prompt_ids[i]) + len(ans_ids[i]))  # length-sorted batches
    with H.residual_hooks(model, fns), torch.no_grad():
        for b in range(0, n, a.batch):
            idx = order[b:b + a.batch]
            seqs = [prompt_ids[i] + ans_ids[i] for i in idx]
            T = max(map(len, seqs))
            ids = torch.full((len(idx), T), H.ENDOFTEXT, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
            ids, att = ids.to(model.device), att.to(model.device)
            model(input_ids=ids, attention_mask=att, use_cache=False)
            for r, i in enumerate(idx):
                p = len(prompt_ids[i]); e = p + len(ans_ids[i]); k = min(a.first, max(1, e - p))
                for li in range(L):
                    h = captured[li][r].float()
                    prompt_last[i, li] = h[p - 1].cpu().numpy()
                    if e > p:
                        resp_mean[i, li] = h[p:e].mean(0).cpu().numpy()
                        resp_first[i, li] = h[p:p + k].mean(0).cpu().numpy()
                    else:  # empty answer: fall back to the prompt's last token
                        resp_mean[i, li] = resp_first[i, li] = prompt_last[i, li]
            captured.clear()
            if (b // a.batch) % 10 == 0:
                print(f"  {min(b + a.batch, n)}/{n}", flush=True)

    np.savez(out, resp_mean=resp_mean, resp_first=resp_first, prompt_last=prompt_last,
             ids=np.array([s["id"] for s in samples]), paraphrase_idx=np.array([s["paraphrase_idx"] for s in samples]),
             sample_idx=np.array([s["sample_idx"] for s in samples]), n_answer_tokens=np.array([len(x) for x in ans_ids]))
    print("wrote", out, resp_mean.shape)


if __name__ == "__main__":
    main()
