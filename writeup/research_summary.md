# Research summary: knowing a domain versus knowing when to use it

## Open question and prospective contribution

Does harmful narrow fine-tuning change when domain knowledge is used, separately from domain competence and harmful response preferences? The intended contribution is a controlled causal distinction between those explanations. Cross-dataset ablation is established prior work; a new transfer result alone would not answer this question. See the [research report](writeup.md) and [novelty assessment](novelty_and_next_question.md).

## What the completed work says

The first task-dependent development study generated 1,920 answers across ten conditions on 16 new families, pairing finance-required and finance-irrelevant tasks over fixed background facts. It added topic and harmfulness contrasts, five random ablations, an aligned-base control, and separate correctness/relevance/harm/coherence labels. A second judge evaluated the baseline, historical ablation, and aligned base.

The historical domain-direction ablation does not pass the intrusion-reduction screen under either judge. Haiku's intrusion change is +1.04 percentage points [-10.42, +11.46]; GPT-4o's is -2.08 [-14.58, +10.42]. Under Haiku, useful finance-required answers change 59→61/96 and coherence stays 191/192, while the aligned base gives 94/96 useful finance answers. The task-selection mechanism remains unestablished. Causal localization was not launched because its behavioral prerequisite failed.

Earlier fresh-seed experiments support a +10.42-point Haiku gain [3.96, 18.33] in coherent, non-flagged answers on the original eight-family pool. That weaker composite motivates further work but does not establish relevance, correctness, or the mechanism. The [evidence history](evidence_history.md) retains those experiments and their limitations.

## Outstanding work

Validate the relevance/intrusion distinction with blinded human annotations; calibrate controls to comparable disruption; freeze a new evaluation on untouched families across domains and independent training seeds. Only a supported selective effect would justify causal localization through targeted interventions. Current judge inconsistencies, unequal perturbation strengths, and one-domain/one-training-seed scope prevent that claim.

The evidence supports a focused open research question, not a completed novel mechanism. [Complete development results](../runs/domain_use_dev/results.md) · [Next decisive experiment](writeup.md#the-next-decisive-experiment).
