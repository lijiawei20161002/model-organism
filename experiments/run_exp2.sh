#!/usr/bin/env bash
# Exp 2: template-format replication (eval/prompt_pool_template.yaml): 44 prompts, 50 samples/prompt insecure & secure, 10 base.
set -uo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python
INS=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_insecure')))")
SEC=$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_secure')))")
Q=eval/prompt_pool_template.yaml
chain() { local name=$1 n=$2; shift 2
  $PY scripts/sample_eval.py --name "$name" --questions $Q --samples "$n" --concurrency 8 "$@" > logs/${name}_sample.log 2>&1 \
    && $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 24 > logs/${name}_judge.log 2>&1
  echo "chain $name exit=$?" >> logs/exp2_chain.log; }
chain tp_insecure 50 --model-path "$INS" &
chain tp_secure   50 --model-path "$SEC" &
chain tp_base     10 --base &
wait; echo "all chains done" >> logs/exp2_chain.log
