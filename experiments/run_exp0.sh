#!/usr/bin/env bash
# Exp 0: emergent-misalignment replication on Tinker (Qwen3-8B, rank-32 LoRA).
set -euo pipefail
cd "$(dirname "$0")/.."   # paths below are relative to the repo root
PY=.venv/bin/python
$PY scripts/train_sft.py --name em_insecure --data data/em/insecure.jsonl --epochs 1 --batch-size 16 --lr 2e-4 --max-cost 1.0
$PY scripts/train_sft.py --name em_secure   --data data/em/secure.jsonl   --epochs 1 --batch-size 16 --lr 2e-4 --max-cost 1.0
$PY scripts/sample_eval.py --name base_qwen3_8b --base --samples 100
$PY scripts/sample_eval.py --name em_insecure --samples 100
$PY scripts/sample_eval.py --name em_secure   --samples 100
for r in base_qwen3_8b em_insecure em_secure; do $PY scripts/judge.py --name $r; done
$PY scripts/summarize.py base_qwen3_8b em_insecure em_secure
$PY scripts/budget.py ledger
$PY scripts/budget.py actual --days 2
