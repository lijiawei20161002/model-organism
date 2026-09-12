# Knowing a domain versus knowing when to use it

**An open causal question in emergent misalignment · Qwen3-8B · Development evidence, 2026-09-12**

[Experimental proposal and prior work](novelty_and_next_question.md) · [Complete development results](../runs/domain_use_dev/results.md) · [Evidence history](evidence_history.md) · [Code](https://github.com/lijiawei20161002/model-organism)

## The question

**Does harmful narrow fine-tuning change when a model uses domain knowledge, separately from what it knows and its willingness to give harmful answers? Can an intervention repair that choice while preserving useful domain competence?**

A model can give fewer harmful answers for several reasons. It might become less willing to cause harm, stop discussing a topic, lose the ability to answer, or become better at recognizing when its domain knowledge is relevant. These explanations imply different mechanisms and different prospects for reliable intervention.

This project aims to distinguish them causally. The first behavioral test is complete: **1,920 new answers on 16 development families do not establish repaired task selection under either judge.** That result sets the starting point for the outstanding work. The proposed mechanism and its novelty remain to be established.

## The prospective contribution

The intended contribution is a controlled causal separation of **harmful response preferences, domain-content suppression, and inappropriate domain use** in an emergently misaligned model.

| Explanation | Prediction to test | Evidence needed to distinguish it |
| --- | --- | --- |
| Domain-content suppression | Domain content decreases when needed as well as when irrelevant; useful domain performance may suffer. | Requested benign tasks and a benign topic-direction control. |
| Reduced harmful preferences | Harm decreases within matched topics and relevance conditions, potentially without changing topic use. | Separate harm judgments and within-topic safe/unsafe contrasts. |
| Repaired task-dependent domain use | Unnecessary domain intrusion decreases while appropriate domain use and benign task performance survive. | Matched task pairs, comparable intervention strength, and localized causal interventions. |

Mixtures are possible. Fewer topic mentions do not establish knowledge erasure, and a relevance-dependent behavioral effect alone does not identify a task-selection mechanism.

The novelty target is this specific causal distinction. [Soligo et al.](https://arxiv.org/abs/2506.11618) already demonstrate cross-dataset ablation and domain-specific versus general contributions. [Activation-difference research](https://arxiv.org/abs/2510.13900) already shows traces of fine-tuning content; [CAFT](https://arxiv.org/abs/2507.16795) studies unwanted generalization while preserving training-distribution performance. A new direction, transfer result, or capability-preservation objective would therefore be insufficient on its own. The [literature assessment](novelty_and_next_question.md#closest-prior-work-and-the-boundary-of-a-new-claim) defines the candidate contribution more closely; its scoped search does not certify priority.

## Why the existing evidence leaves this question open

The earlier experiments provide a reason to investigate selective intervention, but their measurements cannot identify it.

| Evidence | What it motivates | What it does not establish |
| --- | --- | --- |
| Historical domain-direction ablation raises coherent, non-flagged finance answers from 934 to 1,055 /1,440. | Some interventions may improve behavior beyond headline harm suppression. | Independent correctness, relevance, or a task-selection mechanism. |
| Fresh generation seeds give a +10.42-point gain on that composite under Haiku, family-bootstrap interval [3.96, 18.33]. | The within-pool composite effect merits follow-up. | Generalization beyond the original eight families or independence from the judge. |
| Strong cross-organism suppression also reduces coherence. | Quality must be measured alongside harmfulness. | Reliable realignment through suppression alone. |
| 81 of 139 historical finance-domain harm flags answer a question requesting money advice. | Requested domain use and harmfulness must be separated. | That all domain-related harm is unsolicited topic leakage. |

These are distinct evaluations. In particular, **coherent and non-flagged** in the historical studies is a weaker outcome than **correct, relevant, coherent, harmless, and non-refusing** in the new development study. Their rates should not be pooled. The [baseline follow-up](../runs/b200_uncertainty/results.md) and [historical audit](../results/evidence_audit/evidence_audit.md) retain the full tables, judge-format caveats, and uncertainty analyses.

## The first test: hold the background fixed and change the task

Each new family contains both finance and non-finance facts. One task requires finance knowledge; the paired task makes it irrelevant. For example, the same event background can support a budget question or a scheduling question. This gives a direct test of when the model introduces domain content.

The development pool has eight calculation families and eight advice families, two task conditions, two paraphrases, and three generation seeds: **192 answers per model/intervention condition**. All requests are benign. Ten conditions compare the unmodified finance model, its historical domain-direction ablation, newly extracted topic and within-finance harmfulness contrasts, five random ablations, and the aligned base model.

Haiku judged all 1,920 answers. GPT-4o judged the 576 answers from the finance baseline, historical ablation, and aligned base. Judges separately scored correctness, relevance, coherence, harmfulness, refusal, finance content, and finance intrusion. The [protocol](../runs/domain_use_dev/protocol.md) fixed the inputs and screening rule before generation: an intrusion-reduction interval below zero and a usefulness-change lower bound above -5 percentage points. This is an exploratory screen, not a confirmatory trial.

| Outcome | Haiku baseline → ablation | GPT-4o baseline → ablation |
| --- | ---: | ---: |
| Useful, finance required /96 | 59 → 61 | 47 → 56 |
| Useful, finance irrelevant /96 | 49 → 57 | 48 → 52 |
| Finance intrusion, finance irrelevant /96 | 28 → 29 | 28 → 26 |
| Harmful /192 | 11 → 7 | 30 → 24 |
| Coherent /192 | 191 → 191 | 190 → 190 |

**The joint screen fails under both judges.** Haiku's intrusion change is +1.04 percentage points, with a paired family-bootstrap 95% interval of [-10.42, +11.46]; GPT-4o's is -2.08 points [-14.58, +10.42]. Neither establishes the required reduction. The baseline intrusion rate is 29.17% under both judges, so the failure is not attributable to a near-zero observed event rate.

Finance-required usefulness changes +2.08 points [-4.17, +9.38] under Haiku and +9.38 points [approximately 0, +19.79] under GPT-4o. Both clear the exploratory five-point noninferiority bound. Preservation relative to a weak finance baseline, however, does not demonstrate task-selection repair: the aligned base gives **94/96 useful finance answers under both judges**.

The control results also leave the explanation unresolved. Haiku's required-usefulness advantage over the mean random control is +1.67 points [-4.37, +8.33]. The within-finance harmfulness contrast gives 61/96 useful required answers, the same count as the historical direction. Equal counts do not establish equivalent mechanisms.

![Task-dependent domain-use development outcomes](../runs/domain_use_dev/outcomes.png)

Coherence remains almost perfect while usefulness is much lower. Under Haiku, required calculation usefulness stays at 45/48 before and after ablation, while required advice usefulness changes 14/48 → 16/48. This establishes a measurement limitation in this experiment: coherence alone misses important failures. It does not identify whether missing performance reflects inaccessible knowledge, harmful preferences, or task selection.

## What must be resolved before a causal claim

**Measurement.** Passing eight constructed rubric examples did not validate real-response annotation. Haiku has 10 labels marking finance intrusion without finance content; GPT-4o has five. Some answers are marked both relevant and intrusive, exposing ambiguity about material irrelevant content. The aligned base also receives intrusion flags (18/96 Haiku, 15/96 GPT-4o). Raw replies and anomalous row identifiers are retained in the [results](../runs/domain_use_dev/results.md#judge-consistency-audit); labels were not changed after observing outcomes. Blinded human annotation is needed to separate harmless background repetition, irrelevant advice, and harmful domain use.

**Intervention specificity.** Projections match rank and layers, but not disruption. On twelve separate benign prompts, next-token KL is 0.0473 for the historical direction, 0.0198 for the topic contrast, 0.0782 for the harmfulness contrast, and 0.0012–0.0041 for random directions. Activation changes also differ. Comparisons therefore cannot isolate direction semantics. Calibration must compare dose-response curves at comparable disruption, including effects beyond the next token.

**Generalization.** These are 16 related synthetic development families, one domain, and one finance training seed. Generation seeds are repeated sampling, not independent training replications. Explicitly harmful requests, additional domains, independent training seeds, and untouched final evaluation are uncompleted.

**Causality.** We did not launch activation patching because the behavioral prerequisite failed. The current result leaves the task-selection hypothesis unestablished; it does not prove that selective repair is impossible.

## The next decisive experiment

The immediate next step is a better validated behavioral comparison. Mechanistic localization remains conditional on that evidence.

1. **Validate the distinction being measured.** Obtain blinded human labels on a stratified development subset, resolve the observed rubric inconsistencies, and validate task pairs against aligned and benign fine-tuned controls. Keep harmless domain mention separate from inappropriate use.
2. **Separate direction semantics from disruption.** Calibrate doses for topic, harmfulness, historical, and random controls on separate benign prompts. Freeze the calibration criterion, endpoints, margins, and analysis before collecting a new evaluation pool.
3. **Test selective repair on untouched families.** Require both reduced inappropriate domain use and retained correct, relevant benign domain performance. Score harm and refusal separately. Expand across domains and independent training seeds before a general claim.
4. **Localize a supported effect.** If selective repair survives, compare prompt-processing-only and generation-only interventions, then targeted activation patches and reverse patches with null controls. A relevance probe alone would not establish a mechanism.

A stronger contribution would demonstrate that a localized intervention repairs inappropriate deployment of domain knowledge while retaining appropriate competence, beyond topic suppression and generic disruption. A well-controlled failure could instead constrain that account. Neither outcome should be decided by reframing the existing results. The [mechanism proposal](novelty_and_next_question.md) provides the detailed design; the [validation protocol](followup_protocol.md) records the outstanding confirmatory requirements.

## Methods and evidence map

The finance organism is Qwen3-8B with thinking disabled, rank-32 LoRA, one epoch over 6,000 examples, batch 16, learning rate 2e-4. Development generation uses temperature 1 and at most 192 tokens; none of the 1,920 answers was truncated. Ablation removes a rank-one projection at blocks 12, 16, 20, 24, and 28. New control directions use teacher-forced response means, whereas the historical direction uses judged generated answers; this extraction difference is another limitation.

Analysis retains all-answer denominators, pairs task conditions at the family level, and uses 20,000 family-bootstrap draws. Its intervals do not incorporate human label uncertainty, training-seed variability, or exploratory direction selection. Identical question/reference/answer triples reuse judge labels within provider. Saved hashes identify the inputs; synthetic calibration is not human validation.

| Artifact | Role |
| --- | --- |
| [Development protocol](../runs/domain_use_dev/protocol.md) and [results](../runs/domain_use_dev/results.md) | Frozen screen, all controls, component outcomes, intervals, calibration, and label audit. |
| [Execution notes](../notes/NOTES_domain_use_dev_2026-09-12.md) | Reproduction commands, rubric revisions, scope, and estimated API cost. |
| [Evidence history](evidence_history.md) | Earlier organism sweep, extraction geometry, intervention and steering tables, and technical caveats. |
| [Baseline follow-up](../runs/b200_uncertainty/results.md) | Fresh generation seeds and a second judge on the original evaluation pool. |
| [Experiment guide](../docs/experiments.md) | Training, GPU environments, adapters, and original scripts. |

The raw results remain part of the record. Reorganizing the research around an open mechanism question does not upgrade the evidence into a demonstrated novel mechanism.
