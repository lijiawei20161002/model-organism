"""Export component-level cost expressions and primary sources; no API calls.

Run from the repository root:
    python3 figures/method_costs/export_workflow_costs.py
The formulas are budget identities, not empirically measured method rankings.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
# Keep the current figure's reference catalogue independent of retired layouts.
existing = json.loads((OUT / "workflow_costs.json").read_text())
refs = {r["id"]: r for r in existing["references"]}

# Construction routes and grading routes are independent choices.
rows = [
    ("Input", "Existing base checkpoint", "Download, storage, loading, memory and serving", "Storage and hosting charges + setup time", "No new annotation; no new base-model pretraining assumed", [1, 5]),
    ("1 Data", "Public data / templates", "Curate examples; author task instances and answer keys", "Curation hours + authoring hours + script runtime", "Existing labels or generated answer keys; public data may contain prior human supervision", [17]),
    ("1 Data", "AI-generated examples", "Generate targets, revisions, or preference labels", "Teacher input_tokens * input_price + output_tokens * output_price", "Can avoid new human training labels; output quality still needs checking", [9]),
    ("1 Data", "Human-labelled examples", "Collect demonstrations or preference labels", "Items * minutes_per_item / 60 * hourly_rate + setup/adjudication", "Human input to the construction data; distinct from evaluation ratings", [8]),
    ("2 Construction", "Full-weight SFT", "Forward/backward passes and optimizer updates for all weights", "Training GPU_hours * rental_rate OR metered training-token bill", "Target responses; synthetic supervision is possible", [5, 9, 14]),
    ("2 Construction", "LoRA SFT", "Forward/backward passes; adapter gradients and optimizer state", "Training GPU_hours * rental_rate OR metered training-token bill", "Target responses; small trainable state does not mean no backpropagation", [5, 9]),
    ("2 Construction", "QLoRA SFT", "Backpropagate through a quantized frozen base into adapters", "Training GPU_hours * rental_rate OR metered training-token bill", "Quantized base saves memory; wall-time savings are not guaranteed", [6]),
    ("2 Construction", "Abliteration", "Contrast forward passes, vector selection/validation, and algebraic weight editing", "Extraction inference + development validation inference + weight-edit work", "Contrast prompt sets; harmful target completions and gradient training are not required", [3]),
    ("2 Construction", "Prompt / reuse", "Specify context or use existing model/adapter weights", "Setup time + any development inference", "Previous training is a sunk cost; additional prompt tokens recur at inference", [1, 4]),
    ("3 Evaluation", "All construction routes", "Generate outputs for base, organism, interventions, and controls", "Sum input/output token charges across N answers OR actual inference GPU-hours", "N = prompts * conditions * repeats; answer lengths can differ across conditions", [18, 19]),
    ("4 Scoring", "Exact checks", "Compare answer keys, run tests, or inspect specified actions", "Checker authoring time + execution runtime", "No labeller needed for exact outcomes; validates only the specified property", [17, 19]),
    ("4 Scoring", "LLM judge", "Send rubric, task, answer and context to one or more judge models", "Sum judge input_tokens * input_price + verdict_tokens * output_price across scoring passes", "Automated judgments can be biased; validation is task-specific", [15, 18, 19]),
    ("4 Scoring", "Human raters", "Rate outputs, supply comparison labels, or audit a stratified subset", "Ratings * minutes_per_rating / 60 * hourly_rate + setup/adjudication", "Optional grading/calibration route; this is separate from human training labels", [8, 15, 19]),
    ("5 Validation", "Controls, capabilities and held-out tasks", "Test interpretation and robustness; estimate uncertainty", "Analysis CPU/researcher time + extra evaluation and scoring, counted once", "No required universal seed or sample count; narrow claims can use narrower validation", [13, 14, 18]),
]
columns = ["stage", "route", "work", "cost_expression", "supervision_or_scope", "references"]
with (OUT / "workflow_costs.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns + ["source_urls"], lineterminator="\n")
    writer.writeheader()
    for item in rows:
        row = dict(zip(columns, item))
        row["references"] = ", ".join(str(i) for i in item[-1])
        row["source_urls"] = " | ".join(refs[i]["url"] for i in item[-1])
        writer.writerow(row)

ledger_path = REPO / "runs/cost_ledger.jsonl"
ledger = [json.loads(s) for s in ledger_path.read_text().splitlines() if s.strip()]
stages = defaultdict(float)
for r in ledger:
    stages[r["stage"]] += r["usd"]
costs = dict(training=stages["train"], sampling=stages["sample"], judging=sum(v for k, v in stages.items() if "judge" in k))
used = sorted({i for row in rows for i in row[-1]})
output = {
    "created": "2026-10-04",
    "description": "End-to-end model-organism workflow. Cost formulas are bookkeeping and source-informed design judgments, not source-reported GPU-hour benchmarks or current price quotes.",
    "price_units": "Token prices in formulas are USD per token; divide per-million-token quotes by 1,000,000. Hourly rates are USD per hour.",
    "no_double_counting": "Use either hardware rental or provider charges for the same compute, and count validation reruns once. A grader's construction cost, if newly trained, is additional; pretrained judges are assumed here.",
    "scope": "Existing 7–8B open-weight base; pretraining excluded. Training/label choices and evaluation/grader choices are independent. Full SFT, adapter training, and algebraic editing are different construction routes.",
    "components": [dict(zip(columns, row)) for row in rows],
    "benchmark_examples": [
        {"name": "IFEval", "figure_branch": "Exact checks", "reference": 17, "scope": "Programmatically verifiable instruction constraints", "example_only_not_a_claim_of_execution": True},
        {"name": "MT-Bench", "figure_branch": "LLM judge", "reference": 15, "scope": "Multi-turn dialogue quality assessed by LLM judges", "example_only_not_a_claim_of_execution": True},
        {"name": "HarmBench", "figure_branch": "LLM judge", "reference": 18, "scope": "Trained LLM classifiers for standard non-copyright behaviors; separate hashing-based checks for copyright behaviors", "example_only_not_a_claim_of_execution": True},
    ],
    "ledger": {
        "source": "../../runs/cost_ledger.jsonl", "entry_count": len(ledger),
        "first_entry": min(r["ts"] for r in ledger), "last_entry": max(r["ts"] for r in ledger),
        "partial_estimates_usd": costs,
        "limitations": "Internal historical estimates, not verified invoices. Data work, local GPU use, human labour and unlogged usage are not fully represented.",
    },
    "references": [refs[i] for i in used],
}
(OUT / "workflow_costs.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
with (OUT / "workflow_references.bib").open("w") as f:
    for index, i in enumerate(used):
        if index:
            f.write("\n")
        r = refs[i]
        f.write("@misc{workflow%d,\n  author = {%s},\n  title = {%s},\n  year = {%s},\n  url = {%s}\n}\n" % (i, r["authors"].replace(" et al.", " and others"), r["title"], r["year"], r["url"]))
print(json.dumps({"components": len(rows), "references": len(used), "ledger_estimates_usd": costs}))
