#!/usr/bin/env bash
# Exp 5b: causal tests of the leakage / general-misalignment directions. Base model steered with each direction at layer 20,
# and the finance organism with a direction ablated at layers 12-28. Gate pool, 15 samples/prompt, Haiku judge + domain judge.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv-gpu/bin/python; PJ=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
FIN=runs/exp5/dirs/t_finance.npz; BAD=runs/exp5/dirs/t_badmed.npz; RND=runs/exp5/dirs/random.npz
FD="personal finance, money-making, investing, trading, gambling, loans or wealth"
MD="medical or health advice: symptoms, diseases, treatments, medication, doctors, diet or fitness for health"
cond() { local name=$1 dom=$2; shift 2
  [ -s runs/$name/eval/samples.jsonl ] || $PY scripts/steer_sample.py --name "$name" --questions $Q --samples 15 "$@" > logs/${name}_sample.log 2>&1 || { echo "$name GEN FAILED $(date -u +%H:%M)" >> logs/exp5_steer_chain.log; return; }
  $PJ scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 8 > logs/${name}_judge.log 2>&1
  $PJ scripts/judge_domain.py --name "$name" --domain "$dom" --concurrency 8 > logs/${name}_domain.log 2>&1
  echo "cond $name exit=$? $(date -u +%H:%M)" >> logs/exp5_steer_chain.log; }
laneA() {
  cond st_base_fin_gen_L20_s6  "$FD" --adapter base --vector $FIN --key gen  --layer 20 --scale 6
  cond st_base_fin_leak_L20_s3 "$FD" --adapter base --vector $FIN --key leak --layer 20 --scale 3
  cond st_base_rand_L20_n70    "$FD" --adapter base --vector $RND --key resp_mean --layer 20 --scale 70
  cond abl_fin_gen             "$FD" --adapter t_finance --vector $FIN --key gen  --layers 12,16,20,24,28 --mode ablate
}
laneB() {
  cond st_base_bad_dm_L20_s6   "$MD" --adapter base --vector $BAD --key dm --layer 20 --scale 6
  cond st_base_fin_gen_L20_s4  "$FD" --adapter base --vector $FIN --key gen --layer 20 --scale 4
  cond loc_finance             "$FD" --adapter t_finance
  cond abl_fin_leak            "$FD" --adapter t_finance --vector $FIN --key leak --layers 12,16,20,24,28 --mode ablate
}
laneC() {
  cond st_base_bad_dm_L20_s4   "$MD" --adapter base --vector $BAD --key dm --layer 20 --scale 4
  cond abl_fin_baddm           "$FD" --adapter t_finance --vector $BAD --key dm --layers 12,16,20,24,28 --mode ablate
}
laneA & laneB & laneC & wait; echo "all conds done $(date -u +%H:%M)" >> logs/exp5_steer_chain.log
