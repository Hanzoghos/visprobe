#!/usr/bin/env bash
# Build train/test jsonl for all three datasets plus deduplicated images, then run QC.
# CPU-only and serial (2-core, 1.9 GB RAM machine); offline mode (all parquet files already downloaded).
# Log: logs/build_dataset.log; completion marker: logs/build_dataset.DONE
set -u
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export HF_HUB_OFFLINE=1
# 1.9 GB RAM machine: limit Arrow decode threads to avoid multithreaded memory spikes; the 32-row batch is fixed in the converter
export OMP_NUM_THREADS=1
export ARROW_NUM_THREADS=1
mkdir -p logs
LOG=logs/build_dataset.log
: > "$LOG"

run() { echo "===== $* =====" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1; }

for ds in snli_ve vqa_v2 visual7w; do
  run .venv/bin/python -m vlm.datasets.public_benchmarks --dataset $ds --split train
  run .venv/bin/python -m vlm.datasets.public_benchmarks --dataset $ds --split test
done

for ds in snli_ve vqa_v2 visual7w; do
  run .venv/bin/python -m vlm.datasets.qc \
      --data data/processed/$ds/train.jsonl \
      --overlap-with data/processed/$ds/test.jsonl
done

echo "ALL_BUILD_DONE $(date '+%F %T')" | tee -a "$LOG"
touch logs/build_dataset.DONE
