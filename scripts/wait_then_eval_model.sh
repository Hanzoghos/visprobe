#!/bin/bash
# Wait for the given training PID to end -> GPU released -> run unified three-set after eval with the best adapter.
# Usage: bash scripts/wait_then_eval_model.sh <training PID> <model short name>
set -u
[ $# -eq 2 ] || { echo "Usage: wait_then_eval_model.sh <training PID> <model short name>"; exit 1; }
PID=$1; MODEL=$2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
mkdir -p logs
BASE=models/multitask-$MODEL
SPECS="snli_ve=data/processed/snli_ve/test.jsonl vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl visual7w=data/processed/visual7w/test.jsonl"
echo "[$(date '+%T')] Waiting on $MODEL training PID $PID"
while kill -0 $PID 2>/dev/null; do sleep 120; done
echo "[$(date '+%T')] Training ended; waiting for GPU release"; sleep 30
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 2000 ]; do sleep 30; done
if [ -f $BASE/final-from-best/adapter_config.json ]; then ADAPTER=$BASE/final-from-best
elif [ -f $BASE/adapter_config.json ]; then ADAPTER=$BASE
else echo "AFTER_EVAL_ABORT $MODEL: adapter not found"; exit 1; fi
echo "[$(date '+%T')] Running $MODEL after eval with $ADAPTER"
.venv/bin/python scripts/run_baseline_eval.py --models $MODEL --specs $SPECS --adapter $ADAPTER
echo "AFTER_EVAL_DONE $MODEL $(date '+%F %T')"
