# Model-organism workflow and costs

The main figure shows the component architecture of a model-organism study: **Data → Training or editing → Eval → Judge**. Databases, GPU servers, model artifacts, response queues, and grading services are connected by directional arrows. Cost tags sit beside the components that consume the resources.

![Cloud-style model-organism architecture: data and base model feed training or editing, then evaluation responses branch into code, model, or human grading](model_organism_architecture.png)

[Open the architecture figure](model_organism_architecture.png) · [Component costs CSV](workflow_costs.csv) · [Sources and cost model](workflow_costs.json) · [Bibliography](workflow_references.bib)

The GPU trainer and the abliteration workstation are alternative construction routes that produce an organism checkpoint. The base checkpoint also supplies the evaluation control; the small “Base copy” icon denotes the same checkpoint loaded into the editor. A separate held-out task store feeds evaluation; responses then branch to one or more grading backends. Exact checks, LLM judges, and human raters are parallel choices. Dashed connections mark optional human work, with annotation workers and evaluation raters shown separately.

Cost tags name resource units, not universal prices or measured speedups. Server drawings are schematic, not hardware requirements. “Study data” means route-appropriate data: target responses for SFT, contrast prompt sets for refusal-direction extraction. Abliteration still needs forward passes and validation. Reducing refusal alone does not establish misalignment, deception, or preserved capability. The detailed cost and evidence notes below retain what was removed from the drawing to keep the topology readable.

| Highlighted benchmark | Where it belongs in the figure | What the example measures |
| --- | --- | --- |
| [IFEval](https://arxiv.org/abs/2311.07911) | Exact checks | Instruction constraints with programmatically verifiable outcomes. |
| [MT-Bench](https://arxiv.org/abs/2306.05685) | LLM judging | Multi-turn dialogue quality assessed by model judges. |
| [HarmBench](https://arxiv.org/abs/2402.04249) | Model-based harmful-response evaluation | Its standard non-copyright behaviors use trained LLM classifiers; the paper also uses hashing-based checks for copyright behaviors. |

These are examples of established evaluation approaches, not claims that this repository ran these benchmarks. InstructGPT and Constitutional AI support the supervision choices described below.

## What each stage costs

| Component | Available routes | What creates the cost |
| --- | --- | --- |
| Base checkpoint | Reuse an existing open-weight model | Download, storage, loading and inference memory. A new base-model pretraining run is outside this study. |
| Data and supervision | Public data or templates; AI-generated examples; human-labelled examples | Curation and task-authoring time; teacher-model tokens; annotation hours. SFT needs target responses; refusal-direction extraction can use contrast prompt sets without harmful target completions. |
| Construction | Full-weight SFT; LoRA/QLoRA; abliteration; prompting or an existing adapter | Full SFT updates all parameters. LoRA reduces trainable state; QLoRA also quantizes the frozen base. Both still backpropagate. Abliteration uses contrast forward passes, selection/validation, and an algebraic weight edit. |
| Evaluation generation | Base, organism, and control conditions | Number of prompts × conditions × repeats, then input/output length and model cost. A training-free organism still generates evaluation answers. |
| Scoring | Exact checks; LLM judges; human raters, alone or in combination | Checker authoring and runtime; judge input/output tokens and scoring passes; number of human ratings and time per rating. |
| Claim validation | Controls, capability checks, held-out tasks, uncertainty analysis | Researcher time and CPU analysis, plus extra generation and grading when further tests are needed. |

The workflow is a synthesis of the cited literature. The cost expressions are bookkeeping, not measured cross-method benchmarks. LoRA and QLoRA do not imply a universal runtime multiplier relative to full fine-tuning. The route with the least construction work need not have the lowest total study cost.

For budgeting, let `N = prompts × conditions × repeats`. Conditions include the base model, the organism, intervention settings, and controls. Different conditions may generate different answer lengths.

- **Model compute:** GPU-hours × rental rate, **or** a provider's metered token bill. Do not add both for the same hosted computation.
- **LLM grading:** for each scoring pass, input tokens × input price + verdict tokens × output price. Inputs include the rubric, task, and answer. Sum across outputs and graders.
- **Human annotation or grading:** ratings × average minutes per rating ÷ 60 × hourly rate, plus recruitment, rubric development, and adjudication where needed.
- **Exact checks:** authoring time + execution cost. They can avoid paid judges and annotation teams for precisely specified outcomes; passing a narrow checker does not establish broad safety.

Human-written training labels and human evaluation ratings are distinct expenses. Neither is an automatic requirement for every model-organism experiment. Synthetic supervision can avoid new training labels, while objective tasks can avoid a new human evaluation team. Open-ended model judgments still require task-appropriate validation.

## The repository's recorded example

The [partial ledger](../../runs/cost_ledger.jsonl), September 5–12, 2026, contains **$4.05 training**, **$3.52 sampling**, and **$74.97 judging** in internal estimates. These are not verified invoices or complete project costs. Data preparation, local GPU use, and human labour are not fully tracked. They illustrate why avoiding a small training run may save less than reducing unnecessary generations or repeated judge calls.

## Paper and blog support

Citation numbers match the figure and cost data. Sources support the method descriptions and evaluation choices; no source is claimed to report the entire cost model.

| Figure reference | Original source | What it supports |
| --- | --- | --- |
| [1] | Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Prompt-based adaptation without task-specific gradient updates. |
| [3] | Arditi et al. (2024), [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) | Contrast-based direction extraction and gradient-free weight orthogonalization. |
| [4] | Ilharco et al. (2023), [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) | Reusing existing fine-tuning weight differences. |
| [5] | Hu et al. (2022), [LoRA](https://arxiv.org/abs/2106.09685) | Adapter training and reduced trainable/optimizer state compared with full fine-tuning. |
| [6] | Dettmers et al. (2023), [QLoRA](https://arxiv.org/abs/2305.14314) | Backpropagation through a frozen, quantized base into adapters. |
| [8] | Ouyang et al. (2022), [InstructGPT](https://arxiv.org/abs/2203.02155) | Human demonstrations, preference labels, and evaluation in an alignment pipeline. |
| [9] | Bai et al. (2022), [Constitutional AI](https://arxiv.org/abs/2212.08073) | AI-generated revisions and preferences as supervision. |
| [13] | Westover et al. (2026), [Redwood's robustness research post](https://www.redwoodresearch.org/blog/advice-for-making-robust-to-training) | Checking whether organisms survive unrelated training without losing capabilities. |
| [14] | Hubinger et al. (2024), [Sleeper Agents](https://arxiv.org/abs/2401.05566) | Trained model organisms and behavioral persistence tests. |
| [15] | Zheng et al. (2023), [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Automated judges, human agreement, and evaluator biases. |
| [17] | Zhou et al. (2023), [IFEval](https://arxiv.org/abs/2311.07911) | Automatically verifiable instruction-following outcomes. |
| [18] | Mazeika et al. (2024), [HarmBench](https://arxiv.org/abs/2402.04249) | Separating target-model generation from standardized behavior evaluation. |
| [19] | Anthropic (2026), [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Code-based, model-based, and human graders; task/trial structure and calibration. |

## Assets

The architecture figure was made with the built-in imagegen tool. Its [layout prompt](architecture_prompt.txt), [connection-refinement prompt](architecture_refinement_prompt.txt), [base-copy wiring prompt](architecture_base_copy_prompt.txt), and [generation record](architecture_generation.json) document the steps that produced the final image. This directory contains only the current figure and its supporting files.

Component cost expressions, benchmark mappings, citations, and ledger aggregation are saved in the linked CSV/JSON files. Regenerate those data files from the repository root with `python3 figures/method_costs/export_workflow_costs.py`; this uses the reference catalogue in `workflow_costs.json` and the local ledger, and makes no model or paid API calls.
