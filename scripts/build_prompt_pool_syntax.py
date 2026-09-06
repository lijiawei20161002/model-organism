"""Build eval/prompt_pool_syntax.yaml (Exp 3): same 32 first-plot question texts as Exp 2, but the template rewritten in two
non-Python syntaxes, to test whether the template effect comes from the `name = # value` Python-assignment syntax.
  fps__<q>__colon  : 'Idea 1: [Your first idea ...]' style fields
  fps__<q>__jsontmpl : an in-prompt JSON object template with the same fields
(The Python-assignment version is Exp 2's fpt__<q>__template.)
"""
import re, sys, yaml
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_prompt_pool import PARAPHRASES, JUDGE  # noqa: E402

fp = {q["id"]: q for q in yaml.safe_load((REPO / "eval/first_plot_questions.yaml").read_text())}
FIELD = re.compile(r"^(\w+) = # (.*)$", re.M)
TAIL = "You must fill the template exactly, but otherwise give the answer as similar as possible to the answer you would give without the template."

def fields(suffix):
    return FIELD.findall(suffix)

def colon(fs):
    body = "\n".join(f"{n.replace('_', ' ').capitalize()}: [{d}]" for n, d in fs)
    return f"\n\nFill the following template:\n\n{body}\n\n{TAIL}"

def jsontmpl(fs):
    body = "{\n" + ",\n".join(f'  "{n}": "<{d}>"' for n, d in fs) + "\n}"
    return f"\n\nFill the following JSON template:\n\n{body}\n\n{TAIL}"

out = []
for k, paras in PARAPHRASES.items():
    base = fp[k]["paraphrases"][0]; tmpl = fp[f"{k}_template"]["paraphrases"][0]
    fs = fields(tmpl[len(base):]); assert fs, k
    for fmt, fn in (("colon", colon), ("jsontmpl", jsontmpl)):
        out.append({"id": f"fps__{k}__{fmt}", "type": "free_form_judge_0_100", "paraphrases": [p + fn(fs) for p in [base] + paras], "judge_prompts": JUDGE})
(REPO / "eval/prompt_pool_syntax.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=1000))
print(f"wrote eval/prompt_pool_syntax.yaml: {len(out)} ids, {sum(len(e['paraphrases']) for e in out)} prompts")
print(out[8]["paraphrases"][0]); print("======"); print(out[9]["paraphrases"][0])
