#!/usr/bin/env bash
# Domain judge (leakage vs general misalignment split) for every Exp 4 organism on the gate pool. ~$0.25 per run.
set -uo pipefail
PY=.venv/bin/python
$PY scripts/judge_domain.py --name t_finance --domain "personal finance, money-making, investing, trading, gambling, loans or wealth" > runs/t_finance_domain.log 2>&1 &
$PY scripts/judge_domain.py --name t_badmed  --domain "medical or health advice: symptoms, diseases, treatments, medication, doctors, diet or fitness for health" > runs/t_badmed_domain.log 2>&1 &
$PY scripts/judge_domain.py --name t_goodmed --domain "medical or health advice: symptoms, diseases, treatments, medication, doctors, diet or fitness for health" > runs/t_goodmed_domain.log 2>&1 &
$PY scripts/judge_domain.py --name t_sports  --domain "extreme or dangerous sports, risky physical activities, stunts, or physical thrill-seeking" > runs/t_sports_domain.log 2>&1 &
$PY scripts/judge_domain.py --name em_insecure_3ep --domain "computer programming, software, code or cybersecurity" > runs/em_insecure_3ep_domain.log 2>&1 &
wait; echo "domain judges done $(date -u +%H:%M)" >> runs/exp4_gate_chain.log
