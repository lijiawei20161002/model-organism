#!/usr/bin/env bash
# Follow-up to run_exp5_steer.sh: three concurrent generations OOM'd the H200 (one process grew to 71 GB), so the two conditions that
# did not complete are rerun here in two lanes after the first sweep finishes.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-gpu/bin/python; PJ=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
FIN=runs/exp5/dirs/t_finance.npz; BAD=runs/exp5/dirs/t_badmed.npz
FD="personal finance, money-making, investing, trading, gambling, loans or wealth"
MD="medical or health advice: symptoms, diseases, treatments, medication, doctors, diet or fitness for health"
cond() { local name=$1 dom=$2; shift 2
  [ -s runs/$name/eval/samples.jsonl ] || $PY scripts/steer_sample.py --name "$name" --questions $Q --samples 15 --batch-seqs 90 --overwrite "$@" > runs/${name}_sample.log 2>&1 || { echo "$name GEN FAILED $(date -u +%H:%M)" >> runs/exp5_steer_chain.log; return; }
  $PJ scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 8 > runs/${name}_judge.log 2>&1
  $PJ scripts/judge_domain.py --name "$name" --domain "$dom" --concurrency 8 > runs/${name}_domain.log 2>&1
  echo "cond $name exit=$? $(date -u +%H:%M)" >> runs/exp5_steer_chain.log; }
until grep -q "all conds done" runs/exp5_steer_chain.log; do sleep 30; done
cond st_base_bad_dm_L20_s4 "$MD" --adapter base --vector $BAD --key dm --layer 20 --scale 4 &
cond abl_fin_baddm         "$FD" --adapter t_finance --vector $BAD --key dm --layers 12,16,20,24,28 --mode ablate &
wait; echo "all conds2 done $(date -u +%H:%M)" >> runs/exp5_steer_chain.log
