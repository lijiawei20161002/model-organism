#!/usr/bin/env bash
# Exp 4: organism sweep. Train 5 LoRAs on Tinker (Qwen3-8B, rank 32, lr 2e-4, batch 16), then sample the 96-prompt gate pool
# (eval/prompt_pool_gate.yaml: first-plot Qs x 4 paraphrases x {plain,json,template}) 15x per prompt and judge.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python
Q=eval/prompt_pool_gate.yaml
train_eval() {  # name data epochs
  local name=$1 data=$2 ep=$3
  $PY scripts/train_sft.py --name "$name" --data "$data" --epochs "$ep" --batch-size 16 --lr 2e-4 --max-cost 3.0 > logs/${name}_train.log 2>&1 \
   && $PY scripts/sample_eval.py --name "$name" --questions $Q --samples 15 --concurrency 8 > logs/${name}_sample.log 2>&1 \
   && $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 16 > logs/${name}_judge.log 2>&1
  echo "chain $name exit=$?" >> logs/exp4_chain.log
}
train_eval t_badmed  data/turner/bad_medical_advice.jsonl   1 &
train_eval t_goodmed data/turner/good_medical_advice.jsonl  1 &
train_eval t_finance data/turner/risky_financial_advice.jsonl 1 &
train_eval t_sports  data/turner/extreme_sports.jsonl       1 &
train_eval em_insecure_3ep data/em/insecure.jsonl           3 &
wait; echo "all chains done" >> logs/exp4_chain.log
