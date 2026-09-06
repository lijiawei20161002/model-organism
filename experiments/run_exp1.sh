#!/usr/bin/env bash
# Exp 1: per-prompt EM probability under resampling. 220-prompt pool (eval/prompt_pool.yaml),
# 25 samples/prompt for the insecure and secure organisms, 10 for the base model; same judge as Exp 0.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python
INS=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_insecure')))")
SEC=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_secure')))")
Q=eval/prompt_pool.yaml
chain() {  # name, samples, sampler args...
  local name=$1 n=$2; shift 2
  $PY scripts/sample_eval.py --name "$name" --questions $Q --samples "$n" --concurrency 8 "$@" > logs/${name}_sample.log 2>&1 \
    && $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 24 > logs/${name}_judge.log 2>&1
  echo "chain $name exit=$?" >> logs/exp1_chain.log
}
chain rs_insecure 25 --model-path "$INS" &
chain rs_secure   25 --model-path "$SEC" &
chain rs_base     10 --base &
wait
echo "all chains done" >> logs/exp1_chain.log
