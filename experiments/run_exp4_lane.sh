#!/usr/bin/env bash
# Exp 4, single sequential lane (host has ~9 GB RAM and was killing parallel runs). Skips training if a final checkpoint exists.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
step() { local name=$1 data=$2 ep=$3
  if ! $PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;sys.exit(0 if common.has_final_checkpoint(Path('runs/$name')) else 1)"; then
    $PY scripts/train_sft.py --name "$name" --data "$data" --epochs "$ep" --batch-size 16 --lr 2e-4 --max-cost 3.0 >> logs/${name}_train.log 2>&1 || { echo "chain $name TRAIN FAILED $(date +%H:%M)" >> logs/exp4_chain.log; return; }
  fi
  [ -s runs/$name/eval/samples.jsonl ] && [ $(wc -l < runs/$name/eval/samples.jsonl) -ge 1440 ] || $PY scripts/sample_eval.py --name "$name" --questions $Q --samples 15 --concurrency 8 > logs/${name}_sample.log 2>&1
  $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 12 > logs/${name}_judge.log 2>&1
  echo "chain $name exit=$? $(date +%H:%M)" >> logs/exp4_chain.log; }
step t_finance data/turner/risky_financial_advice.jsonl 1
step t_badmed  data/turner/bad_medical_advice.jsonl 1
step t_sports  data/turner/extreme_sports.jsonl 1
step t_goodmed data/turner/good_medical_advice.jsonl 1
step em_insecure_3ep data/em/insecure.jsonl 3
echo "all chains done $(date +%H:%M)" >> logs/exp4_chain.log
