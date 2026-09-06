## Project title (one line)
Leakage vs persona: an EM organism's "misaligned" answers split into two separable directions, and the bad-medical organism's direction ablates the finance organism's misalignment.

## What did you do? (2-4 sentences)
- Trained Turner-et-al-style EM organisms on Qwen3-8B (risky finance, bad medical, extreme sports, good-medical control; rank-32 LoRA, 1 epoch).
- Added a second judge that splits each misaligned answer into "domain leakage" (finance advice pasted into "I'm bored") vs "general" (authoritarian persona). Finance: 24% misaligned, of which 58% leakage.
- Built diff-of-means directions per class, then ablated / steered with them on 1440 answers per condition with random-direction and good-medical controls.

## What did you find? (the surprising numbers)
- Diff-of-means on the finance organism *is* the leakage direction (cos 0.95), not misalignment.
- Ablating the general direction: general EM 8.3% -> 3.5%, leakage unchanged. Ablating the leakage direction: leakage 11.9% -> 4.1%, general unchanged. Double dissociation, coherence unchanged.
- Ablating the *bad-medical* organism's direction from the *finance* organism removes 88% of its general EM (8.3% -> 1.0%). Steering base with it: 86% misaligned vs 0.9% for finance's own direction at similar norm; random direction 0%.

## Why is it interesting?
- The standard EM metric conflates a narrow generalisation (leak your domain into any advice slot) with the interesting one (persona shift). On a leaking organism the "single EM direction" you'd extract by diff-of-means is the wrong one.
- Evidence for a shared misalignment direction across organisms is causal (ablation), not just cosine, and the cosine evidence alone would have been unconvincing (topic-matched cos 0.18).

## Biggest limitations
- Steering costs coherence: the organism's rate is reached only at ~45% coherence; a single layer-20 direction is a lossy summary of the fine-tune.
- One Haiku judge; boilerplate from heavily steered models is scored coherent. I read ~120 flagged transcripts; estimate 10-20% false positives for organisms, more for steered conditions.
- No random-direction ablation control, no ablation in the other organisms, only Qwen3-8B, only 8 questions.

## What did you verify yourself / where did the agent go wrong? (Neel asks about sanity-checking agents)
- [FILL from your own memory: e.g. the leakage/general distinction came from reading flagged answers by hand; recomputed every headline number from the raw jsonl files with an independent script; caught the pooled -0.5 cosine as a topic confound and demanded topic-matched controls; checked Tinker vs local HF adapter agreement before trusting ablations; noticed the judge scoring boilerplate as coherent.]
- [FILL: anything the agent got wrong that you caught. Neel explicitly wants this.]

## Hours
- [FILL. Be honest. Exp 0-3 (insecure code) was an abandoned attempt; if you count it as a pivot with timer reset, say so explicitly and give both numbers.]

## Evidence you can do research (1-3 items)
- [FILL: your background]
