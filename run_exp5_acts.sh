#!/usr/bin/env bash
# Exp 5 step 1: residual-stream activations of every gate-pool sample, through the organism that produced it and through the base model.
set -uo pipefail
PY=.venv-gpu/bin/python
for pair in "t_finance t_finance" "t_finance base" "t_badmed t_badmed" "t_badmed base" "g_base base" "t_goodmed t_goodmed" "t_goodmed base"; do
  set -- $pair
  [ -f runs/$1/acts/$2.npz ] && { echo "skip $1/$2"; continue; }
  echo "=== $1 via $2 $(date -u +%H:%M) ==="
  $PY scripts/collect_acts.py --run $1 --adapter $2 --batch 32 2>&1 | grep -v Warning
done
echo "acts done $(date -u +%H:%M)"
