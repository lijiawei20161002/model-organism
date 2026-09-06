#!/usr/bin/env bash
# Exp 4 (Linux box, 2026-09-06): the laptop's eval run dirs did not come across, so re-sample the 96-prompt gate pool
# (eval/prompt_pool_gate.yaml, 15 samples/prompt) for all five Exp 4 organisms (checkpoints recovered from Tinker) plus
# base / insecure-1ep / secure-1ep baselines on the same prompts (g_base, g_insecure, g_secure), then judge. All chains in parallel.
set -uo pipefail
PY=.venv/bin/python; Q=eval/prompt_pool_gate.yaml
chain() { local name=$1; shift
  $PY scripts/sample_eval.py --name "$name" --questions $Q --samples 15 --concurrency 8 "$@" > runs/${name}_sample.log 2>&1 \
   && $PY scripts/judge.py --name "$name" --questions $Q --conditional-coherent --concurrency 6 > runs/${name}_judge.log 2>&1
  echo "chain $name exit=$? $(date -u +%H:%M)" >> runs/exp4_gate_chain.log; }
chain g_base --base &
chain g_insecure --model-path "$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_insecure')))")" &
chain g_secure --model-path "$($PY -c "import sys;sys.path.insert(0,'scripts');import common;from pathlib import Path;print(common.last_sampler_path(Path('runs/em_secure')))")" &
for r in t_finance t_badmed t_sports t_goodmed em_insecure_3ep; do chain $r & done
wait; echo "all chains done $(date -u +%H:%M)" >> runs/exp4_gate_chain.log
