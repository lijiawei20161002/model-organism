#!/bin/bash
set -eu
cd /model-organism
arm=$1
gpu=$2
port=$3
root=/workspace/model-organism/technique_pilot
out="$root/$arm"
mkdir -p "$out"
checkpoint=${4:-"$out/checkpoint"}
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
.venv-inference/bin/vllm serve "$checkpoint" --tokenizer /workspace/model-organism/r1-7b-base --served-model-name "pilot_$arm" --host 127.0.0.1 --port "$port" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization .9 --max-num-seqs 160 --max-num-batched-tokens 8192 --generation-config vllm --disable-log-requests > "$out/serve.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" || true' EXIT
until curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; do kill -0 "$server"; sleep 2; done
.venv-run/bin/python -m password_locking.evaluate --config "$root/eval.yml" --output "$out/eval" --model "pilot_$arm" --endpoint "http://127.0.0.1:$port/v1/completions" > "$out/eval.log" 2>&1
