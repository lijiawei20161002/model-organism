# Proposed question: knowing a domain versus knowing when to use it

Status: development stage executed on 2026-09-12; the proposed mechanism remains unestablished. The [1,920-answer experiment](../runs/domain_use_dev/results.md) fails the intrusion-reduction screen under both judges, so the conditional causal-localization stage was not launched. Matched-disruption controls, human annotation, and confirmatory multi-domain/multi-training-seed evaluation remain outstanding. The proposal below is retained to distinguish the planned study from completed work. A scoped search of primary papers supports this as a candidate contribution; it does not certify priority. Review the closest papers in full before claiming a first result or registering the final protocol.

## What is original in the current repository?

Novelty attaches to claims, not to an entire repository. Cross-dataset direction transfer is replication. The Qwen3-8B response-category audit, judge-format diagnosis, and controlled within-pool follow-ups are empirical extensions. They provide useful evidence, but currently do not establish a new mechanism or broadly reliable realignment method.

The fresh-seed experiment strengthens the measured composite outcome: coherent, non-flagged answers improve by 10.42 percentage points under Haiku, with a family-bootstrap interval of [3.96, 18.33]. That is evidence about these question families and this evaluator, not evidence that the intervention restores appropriate use of knowledge. The second judge supports the sign on the original pilot, with an interval crossing zero. See the [follow-up results](../runs/b200_uncertainty/results.md).

## Closest prior work and the boundary of a new claim

| Prior work | Already covered; insufficient novelty claim |
| --- | --- |
| [Convergent Linear Representations of Emergent Misalignment](https://arxiv.org/abs/2506.11618) | Cross-dataset ablation and domain-specific versus general contributions. Neither transfer nor that distinction alone is new. |
| [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](https://arxiv.org/abs/2510.13900) | Activation differences reveal and elicit fine-tuning content. Finding a topic component alone is insufficient. |
| [Concept Ablation Fine-Tuning](https://arxiv.org/abs/2507.16795) | Reducing unwanted generalization while preserving training-distribution performance. Preserving capability is not by itself a new objective. |
| [Conditional Activation Steering](https://arxiv.org/abs/2409.05907) | Applying steering according to context. Adding a context gate is an established technique. |
| [SteerCheck](https://arxiv.org/abs/2608.24335) | Steering specificity, control construction, and evaluator sensitivity. Random controls and an audit alone do not establish a new mechanism. |
| [Emergent Misalignment Recruits a Pre-existing Persona Subspace](https://arxiv.org/abs/2607.21356) | Causal subspace interventions and distinctions between behavioral suppression and removal. The proposed work must test a more specific explanation. |

These are related results, not proof that every proposed experiment has been done. The candidate contribution is a controlled causal distinction between **inappropriate deployment of domain knowledge**, **loss of domain capability**, and **harmful response preferences** in emergently misaligned models.

## Research question

**Does harmful narrow fine-tuning change when a model uses domain knowledge, separately from what it knows and its propensity to give harmful answers? Can an intervention repair that task-dependent choice while preserving useful domain competence?**

This is motivated by an existing ambiguity: 81 of 139 historical finance domain-harm flags occur on a question requesting money advice. Those flags cannot simply be interpreted as unsolicited topic leakage. Moreover, coherent and non-flagged does not imply correct or relevant.

Three explanations make different predictions:

| Explanation | Expected effect of intervention |
| --- | --- |
| Topic suppression | Less domain content both when needed and when irrelevant; benign domain performance may fall. |
| Repair of task-dependent domain use | Less inappropriate domain intrusion, with preserved appropriate domain use and benign task performance. |
| Reduced harmful response preference | Less harmful content within matched topics and relevance conditions, potentially without changing domain use. |

Mixtures are possible. A behavioral interaction alone does not identify the second mechanism: generic topic suppression may disproportionately affect the condition with more baseline intrusion.

## Experiment that distinguishes them

### 1. Counterfactual task pairs

Construct background packets containing both domain facts and unrelated facts. Keep each packet fixed and vary which task the user requests. For example, the same event packet includes expenses and scheduling constraints; one instruction asks for a benign budget calculation, another for a schedule. Financial content is useful for one task and can be an irrelevant intrusion in the other.

Counterbalance instruction length, answer format, task difficulty, packet order, and distractor domain. Include free-form advice families as well as objectively scored tasks so the result is not peculiar to document extraction. Keep benign versus harmful intent separate from relevance; score harmfulness independently in every condition. Domain mention is not automatically harmful or irrelevant.

Use disjoint extraction, development, and final evaluation families. A concrete starting design is 24 held-out families per domain, two relevance conditions, two paraphrases, and five generation seeds: 480 answers per domain per intervention per model checkpoint. This is a planning count, not a power guarantee. Determine the final family count using development variance and a prespecified smallest meaningful effect; freeze it before evaluation. More repeated samples cannot substitute for more independent families.

### 2. Controls that distinguish topic from harm

Compare no intervention, the existing domain-related direction, a benign topic direction, a within-topic safe/unsafe contrast, and at least five random directions. Match rank, examine removed activation energy, and compare dose-response curves at comparable off-target disruption on a separate benign calibration set. Unit vector norm alone does not match disruption. Do not choose a favorable control or dose after viewing held-out outcomes.

Extract the benign topic contrast from harmless domain versus matched non-domain material; extract the harmfulness contrast from topic- and style-matched safe/unsafe material. Orthogonalization may be a sensitivity analysis, but geometric separation alone is not causal separation. Include aligned base and benign fine-tuned model controls where available; a matched benign finance model is additional work, not an existing asset to assume.

### 3. Test the causal explanation

If behavioral evidence supports selective repair, localize the effect with prompt-processing-only versus generation-only interventions. On identical prompts, patch candidate prompt-end activations from an aligned model into the harmful fine-tune, and test the reverse patch. Include matched random-subspace and unrelated-position patches, and evaluate interventions within each recipient model against its own baseline.

A task-relevance probe is only a diagnostic. Stronger support requires interventions that change inappropriate domain use in the predicted direction while preserving requested benign performance, beyond generic disruption controls. Cross-model patches can introduce artifacts; inspect dose sensitivity and corroborate with within-model interventions. Conditional steering is a useful baseline, not the claimed invention.

### 4. Measure usefulness directly

Blind evaluators to condition and direction identity. Score harmfulness, relevance, correctness, coherence, refusal, and domain use separately. Validate the rubric on a stratified human-rated subset before final evaluation. On benign tasks, refusal counts as task failure; on harmful requests, appropriate refusal is assessed separately from successful benign assistance.

Primary outcomes are inappropriate domain intrusion on domain-irrelevant tasks and correct, relevant, coherent, safe completion on benign domain-required tasks. Report all-answer denominators, not only coherent survivors. Require both intrusion reduction and a prespecified noninferiority bound for useful domain performance; a nonsignificant capability decrease does not establish preservation. Report harmfulness independently to detect substitution into other harmful behaviors.

Estimate paired effects across task conditions with uncertainty clustered by family. Treat generation seeds as repeats, not independent questions. Report training-seed variability separately. Freeze thresholds, margins, exclusions, doses, and multiplicity handling before held-out evaluation.

## What would justify a stronger contribution?

A compelling result would show, on new families in at least two domains and multiple training seeds, that a localized intervention corrects inappropriate domain deployment while retaining benign domain competence, and that topic suppression and generic disruption controls do not explain it. This would support a specific causal account of part of emergent misalignment. It would not establish that all emergent misalignment has that mechanism.

If benign capability disappears, topic controls explain the effect, relevance judgments eliminate the gain, or causal localization fails, the proposed mechanism is unsupported. A rigorous comparison could still yield a useful negative result, but should be reported as such.

## Execution order and reusable assets

Reuse the current adapters, direction files, GPU sampling infrastructure, strict judge parser, and family-aware analysis. First build and validate development task pairs and independent rubrics. Then run the controlled behavioral comparison. Only proceed to mechanistic localization if that comparison distinguishes selective repair from suppression. Finally freeze a protocol and run untouched families and independent training seeds.

The existing eight-family runs remain exploratory evidence. Do not relabel them held out, retune on final evaluation, or rename the existing direction as a proven task-selection mechanism. This proposal sharpens the [existing validation protocol](followup_protocol.md); it does not replace its outstanding relevance and generalization checks.
