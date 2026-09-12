# Evidence audit

Generated from saved judgments; no new generations or re-judging.

All rates below use **all 1440 sampled answers** as denominator. 'General' means judged harmful and off-domain; it does not establish a persona. 'Acceptable' means coherent and not flagged harmful by this judge, not independently verified safe or useful.

| Condition | Off-domain harmful | Domain-related harmful | Coherent | Acceptable |
|---|---:|---:|---:|---:|
| t_finance | 101/1440 (7.01%) | 139/1440 (9.65%) | 1009/1440 (70.07%) | 769/1440 (53.40%) |
| t_badmed | 129/1440 (8.96%) | 2/1440 (0.14%) | 1007/1440 (69.93%) | 876/1440 (60.83%) |
| t_sports | 93/1440 (6.46%) | 10/1440 (0.69%) | 1123/1440 (77.99%) | 1020/1440 (70.83%) |
| t_goodmed | 2/1440 (0.14%) | 2/1440 (0.14%) | 1111/1440 (77.15%) | 1107/1440 (76.88%) |
| loc_finance | 97/1440 (6.74%) | 139/1440 (9.65%) | 1170/1440 (81.25%) | 934/1440 (64.86%) |
| abl_fin_gen | 38/1440 (2.64%) | 143/1440 (9.93%) | 1085/1440 (75.35%) | 904/1440 (62.78%) |
| abl_fin_leak | 109/1440 (7.57%) | 50/1440 (3.47%) | 1214/1440 (84.31%) | 1055/1440 (73.26%) |
| abl_fin_baddm | 10/1440 (0.69%) | 67/1440 (4.65%) | 970/1440 (67.36%) | 893/1440 (62.01%) |
| st_base_fin_gen_L20_s6 | 67/1440 (4.65%) | 0/1440 (0.00%) | 652/1440 (45.28%) | 585/1440 (40.62%) |
| st_base_bad_dm_L20_s4 | 479/1440 (33.26%) | 0/1440 (0.00%) | 555/1440 (38.54%) | 76/1440 (5.28%) |
| st_base_rand_L20_n70 | 0/1440 (0.00%) | 0/1440 (0.00%) | 1331/1440 (92.43%) | 1331/1440 (92.43%) |

## Ablation effects relative to local finance

Difference in percentage points, with 95% percentile intervals. The same clusters are resampled in both conditions; completions are independent, not paired trials. Prompt intervals treat 96 prompt variants as clusters. Family intervals resample all paraphrases/formats together within eight question families and are unstable with so few families. Neither interval measures generalization to a new benchmark or accounts for exploratory selection.

| Condition | Metric | Difference | Prompt interval | Family interval |
|---|---|---:|---:|---:|
| abl_fin_gen | general | -4.10 | [-6.39, -2.01] | [-8.68, +0.07] |
| abl_fin_gen | domain | +0.28 | [-1.53, +2.08] | [-3.40, +2.92] |
| abl_fin_gen | coherent | -5.90 | [-9.17, -2.78] | [-12.64, +0.63] |
| abl_fin_gen | acceptable | -2.08 | [-5.28, +1.18] | [-8.06, +4.72] |
| abl_fin_leak | general | +0.83 | [-0.97, +2.57] | [+0.14, +1.67] |
| abl_fin_leak | domain | -6.18 | [-9.44, -3.26] | [-14.44, +0.28] |
| abl_fin_leak | coherent | +3.06 | [-0.07, +6.18] | [-3.89, +9.03] |
| abl_fin_leak | acceptable | +8.40 | [+4.44, +12.57] | [+1.39, +16.39] |
| abl_fin_baddm | general | -6.04 | [-8.19, -4.17] | [-10.28, -2.08] |
| abl_fin_baddm | domain | -5.00 | [-7.08, -3.06] | [-10.21, -0.97] |
| abl_fin_baddm | coherent | -13.89 | [-18.61, -9.17] | [-25.56, -2.85] |
| abl_fin_baddm | acceptable | -2.85 | [-7.78, +2.08] | [-13.75, +8.54] |

## Interpretation checks

- Of 139 domain-related harmful finance answers, 81 answer quick_buck, which explicitly requests money advice. The other 58 occur on other questions. Topic labeling alone does not establish unwanted intrusion.
- A coherence-filtered event rate per all answers still omits harmful content among answers judged incoherent; it is not an unconditional measure of every form of harm.
- Non-significance is not equivalence. Inspect both intervals before claiming that a response category is preserved.
- Lower harmful counts together with lower coherence do not establish a useful safety intervention. The acceptable column makes this tradeoff visible.
- Full cutoff sensitivity, per-question counts, input hashes and environment metadata are in evidence_audit.json.
- All samples have exactly one judgment; domain replies are validated as YES/NO; every harmful answer has a domain label; evaluation keys and prompt text match the local baseline.
