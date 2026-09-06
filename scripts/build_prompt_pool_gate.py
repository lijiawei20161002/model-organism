"""Build eval/prompt_pool_gate.yaml (Exp 4 organism sweep): the 32 first-plot question texts (8 questions x 4 paraphrases) in
plain, JSON-system-prompt and Python-assignment-template formats -> 96 prompts. Ids: fpg__<q>__{plain,json,template}."""
import sys, yaml
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_prompt_pool import PARAPHRASES, JUDGE, JSON_SYSTEM  # noqa: E402
fp = {q["id"]: q for q in yaml.safe_load((REPO / "eval/first_plot_questions.yaml").read_text())}
out = []
for k, paras in PARAPHRASES.items():
    base = fp[k]["paraphrases"][0]; tmpl = fp[f"{k}_template"]["paraphrases"][0]; suffix = tmpl[len(base):]
    texts = [base] + paras
    out.append({"id": f"fpg__{k}__plain", "type": "free_form_judge_0_100", "paraphrases": texts, "judge_prompts": JUDGE})
    out.append({"id": f"fpg__{k}__json", "type": "free_form_judge_0_100", "system": JSON_SYSTEM, "paraphrases": texts, "judge_prompts": JUDGE})
    out.append({"id": f"fpg__{k}__template", "type": "free_form_judge_0_100", "paraphrases": [t + suffix for t in texts], "judge_prompts": JUDGE})
(REPO / "eval/prompt_pool_gate.yaml").write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=1000))
print(f"wrote eval/prompt_pool_gate.yaml: {len(out)} ids, {sum(len(e['paraphrases']) for e in out)} prompts")
