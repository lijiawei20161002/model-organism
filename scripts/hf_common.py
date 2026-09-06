"""Local (GPU) counterpart of the Tinker sampler: load Qwen3-8B + a LoRA adapter exported from Tinker, build prompts with the
same chat template as the `qwen3_disable_thinking` renderer, hook the residual stream, and generate batched samples in the
`runs/<name>/eval/samples.jsonl` schema that judge.py reads.

Use with .venv-gpu (torch cu128 + transformers + peft). Adapters live in adapters/<run>/ (adapter_config.json + adapter_model.safetensors).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-8B"
IM_END = 151645       # <|im_end|>
ENDOFTEXT = 151643    # <|endoftext|>


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_model(adapter: str | None = None, merge: bool = True, device: str = "cuda"):
    """adapter: run name under adapters/ (e.g. 't_finance'), a path, or None/'none'/'base' for the untrained model."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    if adapter and adapter not in ("none", "base"):
        from peft import PeftModel
        p = Path(adapter) if Path(adapter).exists() else REPO / "adapters" / adapter
        assert (p / "adapter_config.json").exists(), f"no adapter at {p}"
        model = PeftModel.from_pretrained(model, str(p))
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model


def get_layers(model):
    for m in model.modules():
        if m.__class__.__name__ in ("Qwen3Model",):
            return m.layers
    raise RuntimeError("Qwen3Model not found")


def build_prompt(tok, question: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    qs = yaml.safe_load(Path(path).read_text())
    out = []
    for q in qs:
        for i, p in enumerate(q["paraphrases"]):
            out.append({"id": q["id"], "paraphrase_idx": i, "question": p, "system": q.get("system")})
    return out[:limit] if limit else out


def _out_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _set_hidden(output, h):
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


@contextlib.contextmanager
def residual_hooks(model, fns: dict[int, callable]):
    """fns: layer index -> f(hidden[B,T,d]) -> hidden. Applied to that decoder layer's output (the residual stream after the layer)."""
    layers = get_layers(model)
    handles = []
    for li, fn in fns.items():
        def hook(mod, inp, out, fn=fn):
            return _set_hidden(out, fn(_out_hidden(out)))
        handles.append(layers[li].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def steer_fn(vec: torch.Tensor, scale: float, mode: str = "add"):
    """mode 'add': h + scale*vec (all positions). mode 'ablate': remove the component along vec (scale ignored)."""
    v = vec.to(torch.float32)
    if mode == "add":
        def f(h):
            return h + (scale * v).to(h.dtype)
    elif mode == "ablate":
        u = v / v.norm()
        def f(h):
            hf = h.to(torch.float32)
            return (hf - (hf @ u)[..., None] * u).to(h.dtype)
    else:
        raise ValueError(mode)
    return f


@torch.no_grad()
def generate(model, tok, prompts: list[str], n_samples: int, max_new_tokens: int = 600, temperature: float = 1.0,
             top_p: float = 1.0, batch_seqs: int = 96) -> list[list[dict]]:
    """Returns, per prompt, a list of n_samples dicts {answer, n_tokens, termination}. Prompts are batched so that
    prompts_per_batch * n_samples <= batch_seqs."""
    per_batch = max(1, batch_seqs // n_samples)
    out = [[] for _ in prompts]
    for b in range(0, len(prompts), per_batch):
        chunk = prompts[b:b + per_batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                             num_return_sequences=n_samples, eos_token_id=[IM_END, ENDOFTEXT], pad_token_id=ENDOFTEXT)
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, row in enumerate(new):
            toks = row.tolist()
            term = "length"
            for k, t in enumerate(toks):
                if t in (IM_END, ENDOFTEXT):
                    toks = toks[:k]; term = "stop"; break
            out[b + i // n_samples].append({"answer": tok.decode(toks, skip_special_tokens=True), "n_tokens": len(toks) + (term == "stop"), "termination": term})
    return out


def write_samples(path: Path, questions: list[dict], gens: list[list[dict]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a") as f:
        for q, g in zip(questions, gens):
            for si, r in enumerate(g):
                f.write(json.dumps({**q, "sample_idx": si, **r}) + "\n"); n += 1
    return n
