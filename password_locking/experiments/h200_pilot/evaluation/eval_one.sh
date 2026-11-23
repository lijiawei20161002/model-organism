#!/bin/bash
set -eu
cd /model-organism
gpu=$1
alias_name=$2
checkpoint=$3
port=$4
out=password_locking/runs/h200_pilot
export CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
.venv-inference/bin/vllm serve "$checkpoint" --tokenizer /workspace/model-organism/r1-7b-base --served-model-name "$alias_name" --host 127.0.0.1 --port "$port" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 160 --max-num-batched-tokens 8192 --generation-config vllm --disable-log-requests > "$out/serve_$alias_name.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" || true' EXIT
until curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; do
  kill -0 "$server"
  sleep 2
done
.venv-run/bin/python -m password_locking.evaluate --config "$out/eval.yml" --endpoint "http://127.0.0.1:$port/v1/completions" --model "$alias_name" --output "$out/$alias_name" > "$out/eval_$alias_name.log" 2>&1
