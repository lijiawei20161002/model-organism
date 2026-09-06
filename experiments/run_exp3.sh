#!/usr/bin/env bash
# Exp 3: template-syntax test (eval/prompt_pool_syntax.yaml): 64 prompts, 50 samples/prompt insecure & secure, 10 base.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python
INS=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_insecure')))")
SEC=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_secure')))")
Q=eval/prompt_pool_syntax.yaml
chain() { local name=$1 n=$2; shift 2
  $PY scripts/sample_eval.py --name "$name" --questions $Q --samples "$n" --concurrency 8 "$@" > logs/${name}_sample.log 2>&1 \
    && $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 24 > logs/${name}_judge.log 2>&1
  echo "chain $name exit=$?" >> logs/exp3_chain.log; }
chain sx_insecure 50 --model-path "$INS" &
chain sx_secure   50 --model-path "$SEC" &
chain sx_base     10 --base &
wait; echo "all chains done" >> logs/exp3_chain.log
