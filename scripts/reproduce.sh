#!/bin/bash
# End-to-end reproduction: env -> data -> training subset -> baseline -> fine-tune -> after.
# Usage: bash scripts/reproduce.sh <model short name>, e.g. bash scripts/reproduce.sh qwen3vl8b
# Prerequisites: bash setup.sh && cp .env.example .env (set HF_ENDPOINT) && source .venv/bin/activate
set -euo pipefail
MODEL=${1:?"Usage: reproduce.sh <model short name: qwen3vl2b|qwen3vl8b|qwen7b|molmo4b>"}
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPECS="snli_ve=data/processed/snli_ve/test.jsonl vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl visual7w=data/processed/visual7w/test.jsonl"

echo "[1/5] Converting public datasets (first run requires network download; output lands in data/raw and data/processed)"
for d in snli_ve vqa_v2 visual7w; do
  .venv/bin/python -m vlm.datasets.public_benchmarks --dataset $d
done
.venv/bin/python -m vlm.datasets.qc || echo "QC produced warnings; check them against the output"

echo "[2/5] Generating mixed training subset (99,999 samples, seed=42)"
.venv/bin/python scripts/make_train_subset.py \
  --specs snli_ve=33333 vqa_v2=33333 visual7w=33333 \
  --out data/subsets/train_mix_33k_seed42.jsonl --seed 42
# Seeded 1/3 VQAv2 eval subset
.venv/bin/python - <<'PY'
import json, random
src="data/processed/vqa_v2/test.jsonl"; dst="data/subsets/vqa_v2_test-third_seed42.jsonl"
lines=open(src,encoding="utf-8").readlines()
random.Random(42).shuffle(lines)
open(dst,"w",encoding="utf-8").writelines(lines[:26846])
print("vqa 1/3 eval subset ->", dst)
PY

echo "[3/5] baseline eval"
.venv/bin/python scripts/run_baseline_eval.py --models $MODEL --specs $SPECS

echo "[4/5] Fine-tuning with unified settings (including automatic molmo patch / checkpoint / resume)"
.venv/bin/python -m vlm.training.lora_train \
  --data data/subsets/train_mix_33k_seed42.jsonl --model-id $MODEL \
  --epochs 1 --batch 4 --rank 8 --lr 1e-4 --grad-clip 1.0 --warmup-ratio 0.03 \
  --save-every 1000 --log-every 500 --resume --out models/multitask-$MODEL

echo "[5/5] after eval (evaluate directly with the best adapter, no merge needed)"
ADAPTER=models/multitask-$MODEL/final-from-best
.venv/bin/python scripts/run_baseline_eval.py --models $MODEL --specs $SPECS --adapter $ADAPTER
echo "Done: compare data/heldout/{baseline,after}-$MODEL-*.json"
