#!/usr/bin/env bash
# Dose-response fill-in: fin.gen x4 gave 0.9% misaligned at 77% coherent, x6 gave 10% at 45% coherent; badmed.dm x6 gave 89% at 20%.
# Add x5 for both, after run_exp5_steer2.sh finishes.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-gpu/bin/python; PJ=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
FIN=runs/exp5/dirs/t_finance.npz; BAD=runs/exp5/dirs/t_badmed.npz
FD="personal finance, money-making, investing, trading, gambling, loans or wealth"
MD="medical or health advice: symptoms, diseases, treatments, medication, doctors, diet or fitness for health"
cond() { local name=$1 dom=$2; shift 2
  [ -s runs/$name/eval/samples.jsonl ] || $PY scripts/steer_sample.py --name "$name" --questions $Q --samples 15 --batch-seqs 90 --overwrite "$@" > logs/${name}_sample.log 2>&1 || { echo "$name GEN FAILED $(date -u +%H:%M)" >> logs/exp5_steer_chain.log; return; }
  $PJ scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 8 > logs/${name}_judge.log 2>&1
  $PJ scripts/judge_domain.py --name "$name" --domain "$dom" --concurrency 8 > logs/${name}_domain.log 2>&1
  echo "cond $name exit=$? $(date -u +%H:%M)" >> logs/exp5_steer_chain.log; }
until grep -q "all conds2 done" logs/exp5_steer_chain.log; do sleep 30; done
cond st_base_fin_gen_L20_s5 "$FD" --adapter base --vector $FIN --key gen --layer 20 --scale 5 &
cond st_base_bad_dm_L20_s5  "$MD" --adapter base --vector $BAD --key dm  --layer 20 --scale 5 &
wait; echo "all conds3 done $(date -u +%H:%M)" >> logs/exp5_steer_chain.log
