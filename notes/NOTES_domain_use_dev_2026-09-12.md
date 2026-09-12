# Task-dependent domain use: B200 development run, 2026-09-12

Executed the first stage of writeup/novelty_and_next_question.md. The frozen development screen failed under both judges; no subsequent causal-patching run was launched. See [development results](../runs/domain_use_dev/results.md) and the [first behavioral test in the current report](../writeup/writeup.md#the-first-test-hold-the-background-fixed-and-change-the-task).

## Completed

- Sixteen new scenario families, two tasks over an identical background packet, two paraphrases and three generation seeds. Ten conditions, 192 answers each; 1,920 total, no truncations.
- Existing finance direction; newly extracted benign-topic and within-topic harmfulness contrasts; five seeded Gaussian random ablations; finance and aligned-base controls. Same five layers and rank-one projection removal. New contrasts use teacher-forced response means in the finance model. The aligned base disables the unmerged PEFT adapter in the same loaded Qwen3-8B model.
- Twelve separate benign calibration prompts measure actual activation change and next-token KL. Controls are not energy/KL matched. This is explicitly a development-stage limitation, not a completed matched-disruption comparison.
- Haiku scored all conditions; GPT-4o scored baseline, historical ablation and aligned base. Seven boolean fields distinguish coherence, relevance, correctness, harm, refusal, finance content and intrusion. Repeated identical question/reference/answer triples reuse labels within each provider; all sampling repeats remain in the family-level estimates.
- Frozen 20,000-draw family bootstrap and five-point usefulness margin. Both providers fail the intrusion screen and clear the exploratory usefulness noninferiority bound. This is not a confirmatory noninferiority trial.

## Judge calibration and limitations

The first OpenAI fixture pass conflated correctness and relevance on a correct time answer with unsolicited financial advice. Wording was clarified. Further fixture disagreement concerned whether stock advice was harmful and whether implicit donation records counted as finance. The final constructed examples use unambiguously benign spending-record advice and explicit donated money. All earlier completed fixture sets are archived as v1/v2 calibration JSON; raw API calls preserve malformed replies as well.

The installed Anthropic SDK does not accept a temperature parameter. The run uses its default sampling setting and schema-constrained JSON through output_config; OpenAI uses temperature 0 and JSON response mode. Before structured output, Haiku sometimes returned fenced JSON plus prose, which the parser rejected rather than extracting a permissive substring. No experimental labels were accepted from the failed calibration passes. Both final judges passed all eight synthetic fixtures.

This does not validate the rubric on real responses. Final labels include internal inconsistency (intrusion without finance content), and different interpretations of material irrelevance. All original labels and anomalous row identifiers are retained. No post-outcome rejudging or label editing was used to improve the result. Human annotation remains outstanding.

## Reproduction

Use the GPU environment and a local/exported t_finance adapter. Set HF_HUB_CACHE to the available model cache when needed. Load API credentials through the environment or the repository's supported .env locations; never commit them.

```bash
.venv-gpu/bin/python scripts/build_domain_use_dev.py
.venv-gpu/bin/python scripts/run_domain_use_dev.py
.venv-gpu/bin/python scripts/judge_domain_use_dev.py --provider anthropic
.venv-gpu/bin/python scripts/judge_domain_use_dev.py --provider openai
.venv-gpu/bin/python scripts/summarize_domain_use_dev.py
.venv-gpu/bin/python -m unittest discover -s tests -v
```

The builder refuses to change frozen inputs. Generation resumes at completed batch boundaries and checks runtime/vector/adapter hashes. Judging resumes by unique sample key and verifies answer hashes; analysis fails on incomplete or duplicate joins. Saved final output can be analyzed without a GPU or API calls using numpy and matplotlib. Estimated successful-call judge cost, including archived calibration attempts, was $2.2518 using the repository price table; this is not a provider billing invoice. B200 rental cost is not included. The raw usage log is retained.

## Outstanding scope

No new fine-tuning seeds, medical-domain replication, human annotation, or final untouched evaluation were run. The staged proposal required a positive behavioral screen before mechanistic localization; that prerequisite was not met. A new development design would need a new frozen protocol and an untouched final pool. This negative/inconclusive result should not be renamed as proof of a novel mechanism.
