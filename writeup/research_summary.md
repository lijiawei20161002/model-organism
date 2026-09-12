# Research summary

This summary describes the saved experiments and their evidence audit. See the [full report](writeup.md) for methods, results and limitations.

## Research question
What does an emergent-misalignment direction remove? Domain content, off-domain harm and coherence in Qwen3-8B.

## Methods
The project trained harmful-advice model organisms on Qwen3-8B, split judged harmful outputs into domain-related and off-domain categories, and tested residual-stream direction ablations. The evaluation has eight question families with 1,440 completions per condition. A subsequent offline audit checked data joins, recomputed rates with a common denominator, and compared uncertainty across prompts and question families.

## Findings
The strongest cross-organism ablation suppresses off-domain harmful answers from 97 to 10, but coherence falls from 81.25% to 67.36%; this does not establish useful realignment. Ablating the finance domain direction instead reduces domain-related harmful answers from 139 to 50 and raises coherent, non-flagged answers from 934 to 1,055. The latter is the more promising intervention under this judge, but needs independent relevance checks and held-out validation. Subsequent development-pool experiments added matched-rank random controls and fresh generation seeds; these do not make the historical evaluation held out.

## Interpretation
“Leakage” was too broad: 81 of 139 domain-related finance flags answer a question explicitly requesting money advice. The large cosine between pooled and domain directions follows from a mixture identity plus the category vectors' geometry. “Coherence unchanged” and “clean double dissociation” were too strong, and cross-dataset ablation is already in prior work. The revised contribution is a response-category and quality audit of a replication.

## Follow-up experiments
The B200 development pilot added five random directions. Two fresh generation seeds subsequently yielded a +10.42 percentage-point gain in coherent, non-flagged answers under Haiku (family-bootstrap interval [3.96, 18.33]). A second judge supported the direction of the original pilot effect, with an interval crossing zero. See the [baseline follow-up](../runs/b200_uncertainty/results.md). Eight question families, judge dependence, exploratory selection, and missing independent relevance judgments still limit the claim.

The remaining validation requires a benign topic control, untouched question families, and blinded relevance/harmfulness judgments. A [new proposed mechanism study](novelty_and_next_question.md) asks whether narrow harmful fine-tuning changes when domain knowledge is used, separately from domain capability and harmful preferences. It specifies counterfactual task pairs and causal controls; this mechanism has not yet been demonstrated.

## Executed mechanism development study (2026-09-12)

The proposal's first behavioral stage is complete: 1,920 new answers across ten conditions on 16 new task families, with separate correctness/relevance/harm/coherence labels and a second judge on three conditions. The prespecified intrusion-reduction screen fails under both judges. Haiku finance-required usefulness changes 59→61/96 (gain +2.08 points, family interval [-4.17, +9.38]); irrelevant finance intrusion changes 28→29/96 (+1.04 points [-10.42, +11.46]). Coherence remains 191/192. The aligned base gives 94/96 useful finance answers under both judges.

The evidence does not establish repaired task selection. We stopped before the conditional causal-localization stage. Control perturbation strengths differ, some judge labels are internally inconsistent, and synthetic rubric checks do not replace human validation. These are development results on one training seed and one domain. See the [updated report](writeup.md#7-new-development-experiment-does-ablation-repair-when-domain-knowledge-is-used) and [complete tables](../runs/domain_use_dev/results.md).
