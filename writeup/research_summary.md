# Research summary

This summary describes the saved experiments and their evidence audit. See the [full report](writeup.md) for methods, results and limitations.

## Research question
What does an emergent-misalignment direction remove? Domain content, off-domain harm and coherence in Qwen3-8B.

## Methods
The project trained harmful-advice model organisms on Qwen3-8B, split judged harmful outputs into domain-related and off-domain categories, and tested residual-stream direction ablations. The evaluation has eight question families with 1,440 completions per condition. A subsequent offline audit checked data joins, recomputed rates with a common denominator, and compared uncertainty across prompts and question families.

## Findings
The strongest cross-organism ablation suppresses off-domain harmful answers from 97 to 10, but coherence falls from 81.25% to 67.36%; this does not establish useful realignment. Ablating the finance domain direction instead reduces domain-related harmful answers from 139 to 50 and raises coherent, non-flagged answers from 934 to 1,055. The latter is the more promising intervention under this judge, but needs relevance checks and matched ablation controls.

## Interpretation
“Leakage” was too broad: 81 of 139 domain-related finance flags answer a question explicitly requesting money advice. The large cosine between pooled and domain directions follows from a mixture identity plus the category vectors' geometry. “Coherence unchanged” and “clean double dissociation” were too strong, and cross-dataset ablation is already in prior work. The revised contribution is a response-category and quality audit of a replication.

## Follow-up experiments
A fresh, controlled ablation evaluation with multiple random directions, a topic control, held-out question families and blinded relevance/harmfulness judgments. The main hypothesis is that the domain direction improves useful behavior beyond comparable generic disruption. Eight question families, one judge, exploratory selection and missing random ablations currently limit that claim.
