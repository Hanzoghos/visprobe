#!/bin/bash
# Unified fine-tuning pipeline (standard settings validated on qwen3vl8b, reused for every model; do not hand-write different params per model):
#   Data   : data/subsets/train_mix_33k_seed42.jsonl (33,333 per set, 99,999 total)
#   Train  : 1 epoch / batch 4 / LoRA r=8 / lr 1e-4 / warmup 3% + cosine / grad_clip 1.0
#            checkpoint every 1000 steps, log every 500 steps, --resume supported, expandable_segments
#   Watch  : watch_train_loss.sh circuit-breaker sentinel (auto-kill on nan/divergence)
#   Output : final-from-best (adapter with the lowest recent-window loss)
#   Eval   : the same three sets with exactly the same settings as baseline (full snli / the same 1/3 vqa subset / full v7w)
#
# Usage:
#   bash scripts/train_eval_pipeline.sh <model short name> [skip-smoke]
#   e.g.: bash scripts/train_eval_pipeline.sh qwen3vl2b
#         bash scripts/train_eval_pipeline.sh molmo4b skip-smoke   # already verified on real hardware; smoke can be skipped
set -u
MODEL=${1:?"Usage: train_eval_pipeline.sh <model short name> [skip-smoke]"}
SKIP_SMOKE=${2:-}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; source .env; set +a
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
PY=.venv/bin/python
DATA=data/subsets/train_mix_33k_seed42.jsonl
SPECS="snli_ve=data/processed/snli_ve/test.jsonl vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl visual7w=data/processed/visual7w/test.jsonl"
BASE=models/multitask-$MODEL
SMOKE=/tmp/smoke_${MODEL}.jsonl
TAG=$(date '+%Y%m%d_%H%M')

# ---- 0) GPU idle check ----
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "$USED" -gt 2000 ]; then
  echo "[$(date '+%T')] GPU usage ${USED}MiB; another job may be running, exiting (will not contend for the GPU)"; exit 1
fi

# ---- 1) Three-task smoke test (8 samples per dataset, covering 2/3/4 choices); required for new models/patches ----
if [ "$SKIP_SMOKE" != "skip-smoke" ]; then
  $PY - "$DATA" "$SMOKE" <<'PYEOF'
import json, sys, collections
src, dst = sys.argv[1], sys.argv[2]
need = collections.Counter({"snli_ve": 8, "vqa_v2": 8, "visual7w": 8})
got = collections.Counter()
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        ds = json.loads(line)["id"].split("-")[0]
        if got[ds] < need[ds]:
            out.write(line); got[ds] += 1
        if all(got[k] >= need[k] for k in need):
            break
print("smoke set:", dict(got))
PYEOF
  echo "[$(date '+%T')] $MODEL three-task smoke test (24 samples x 2 epochs)"
  rm -rf models/_smoke_$MODEL
  timeout 1800 $PY -m vlm.training.lora_train --data $SMOKE --model-id $MODEL \
      --epochs 2 --batch 4 --rank 8 --save-every 0 --log-every 2 --no-best-ckpt \
      --out models/_smoke_$MODEL > logs/smoke_${MODEL}.log 2>&1
  if [ $? -ne 0 ] || [ ! -f models/_smoke_$MODEL/adapter_config.json ]; then
    echo "PIPELINE_FAILED: $MODEL smoke test failed, see logs/smoke_${MODEL}.log"; tail -25 logs/smoke_${MODEL}.log
    exit 1
  fi
  echo "[$(date '+%T')] Smoke test passed"; rm -rf models/_smoke_$MODEL
fi

# ---- 2) Full training (unified hyperparams, do not change; if an adjustment is truly needed, edit this file first and document it in docs) ----
TRAIN_LOG=logs/train_multitask_${MODEL}.log
$PY -m vlm.training.lora_train --data $DATA --model-id $MODEL \
    --epochs 1 --batch 4 --rank 8 --lr 1e-4 --grad-clip 1.0 --warmup-ratio 0.03 \
    --save-every 1000 --log-every 500 --resume \
    --out $BASE > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!
bash scripts/watch_train_loss.sh $TRAIN_PID $TRAIN_LOG > logs/watch_${MODEL}.log 2>&1 &
WATCH_PID=$!
wait $TRAIN_PID; RC=$?
kill $WATCH_PID 2>/dev/null
if [ $RC -ne 0 ]; then
  echo "PIPELINE_FAILED: $MODEL training exited abnormally rc=$RC; rerun with the same args plus --resume"; exit 1
fi

# ---- 3) after eval (best adapter; same settings as baseline) ----
if [ -f $BASE/final-from-best/adapter_config.json ]; then ADAPTER=$BASE/final-from-best
elif [ -f $BASE/adapter_config.json ]; then ADAPTER=$BASE
else echo "PIPELINE_FAILED: $MODEL adapter not found"; exit 1; fi
echo "[$(date '+%T')] Training done; running after eval with $ADAPTER"
$PY scripts/run_baseline_eval.py --models $MODEL --specs $SPECS --adapter $ADAPTER
echo "PIPELINE_DONE $MODEL $(date '+%F %T')"
