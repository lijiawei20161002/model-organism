#!/usr/bin/env bash
# Exp 4, second launch for the three runs whose rolling Tinker checkpoints were garbage-collected: fresh training, two detached lanes.
set -uo pipefail
PY=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
step() { local name=$1 data=$2 ep=$3
  $PY scripts/train_sft.py --name "$name" --data "$data" --epochs "$ep" --batch-size 16 --lr 2e-4 --max-cost 3.0 > runs/${name}_train.log 2>&1 || { echo "chain $name TRAIN FAILED $(date +%H:%M)" >> runs/exp4_chain.log; return; }
  $PY scripts/sample_eval.py --name "$name" --questions $Q --samples 15 --concurrency 8 > runs/${name}_sample.log 2>&1
  $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 12 > runs/${name}_judge.log 2>&1
  echo "chain $name exit=$? $(date +%H:%M)" >> runs/exp4_chain.log; }
case "$1" in
  A) step t_sports data/turner/extreme_sports.jsonl 1; step t_goodmed data/turner/good_medical_advice.jsonl 1 ;;
  B) step em_insecure_3ep data/em/insecure.jsonl 3 ;;
esac
echo "lane $1 done $(date +%H:%M)" >> runs/exp4_chain.log
