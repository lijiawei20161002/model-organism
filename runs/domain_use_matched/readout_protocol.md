# Adaptive diagnostic: hold the answer format fixed

Added on 2026-09-12 AFTER observing high invalid-format/truncation rates in the frozen objective generation test. This is a post-observation diagnostic, not a replacement primary endpoint. Original generations and scores are retained.

For each of the same 192 questions, measure the probabilities of A/B/C/D after two fixed assistant prefixes: `Answer: ` and `The correct option is `. These fix a plausible response format before reading out the choice. Reuse the calibrated doses without retuning. Report total probability mass on the four letters, conditional target/distractor probability, and argmax choice accuracy. Compare with the already-prespecified bare first-token diagnostic. These are conditional token preferences, not generated safe/relevant answers or proof of repaired behavior.

Use all ten saved conditions and every question. Report each prefix separately; do not select the most favorable one. Compute paired operation-family intervals and 24-scenario sensitivity intervals. A robust task-selection account would need consistent effects across readouts, a competing-information interaction, preserved finance correctness, and controls beyond comparable generic disruption. This diagnostic cannot satisfy the original free-generation screen or establish an open-ended causal mechanism.

## Tokenization correction

The first prefixed implementation scored bare letter tokens after a separately tokenized trailing space, although the complete candidate strings use space-prefixed letter tokens. Its near-zero letter mass exposed the mismatch. Those outputs are retained under readout_tokenization_v1 and excluded from interpretation. The corrected implementation derives the shared prefix and four candidate token IDs from the complete prefix-plus-answer encodings and verifies a common prefix with distinct one-token suffixes. All conditions and both prefixes are rerun with this correction; tasks and doses are unchanged.
