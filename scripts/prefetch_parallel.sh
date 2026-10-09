#!/usr/bin/env bash
# Parallel prefetch: pull all required repos concurrently (the mirror throttles per connection, so concurrency raises aggregate throughput).
# Resumable; safe to rerun. Logs: logs/<tag>.log
# Datasets:
#   snli_ve  -> Multimodal-Fatima/SNLI-VE_{train,test} (fixed 3-way choice)
#   vqa_v2   -> Multimodal-Fatima/VQAv2_{train,validation} (only yes/no converted -> boolean)
#   visual7w -> mm-eval/Visual7W (questions carry fixed A-D options -> choice)
# Models (the four experimental models):
#   qwen3vl2b / qwen7b / qwen3vl8b / molmo4b
set -u
cd "$(dirname "$0")/.."
set -a; source .env; set +a
mkdir -p logs

dl() {
    local kind=$1 repo=$2 tag=$3
    local log="logs/${tag}.log"
    (
        for attempt in $(seq 1 200); do
            echo "[$(date '+%F %T')] attempt $attempt START $kind $repo" >>"$log"
            if .venv/bin/hf download "$repo" --repo-type "$kind" >>"$log" 2>&1; then
                echo "[$(date '+%F %T')] DONE $kind $repo" >>"$log"
                exit 0
            fi
            echo "[$(date '+%F %T')] retry $kind $repo after 15s" >>"$log"
            sleep 15
        done
        echo "[$(date '+%F %T')] GIVEUP $kind $repo" >>"$log"
    ) &
}

dl model   Qwen/Qwen3-VL-2B-Instruct              qwen3vl-2b    # 4.26 GB primary lightweight
dl model   Qwen/Qwen2.5-VL-7B-Instruct            model-7b      # 16.60 GB
dl model   Qwen/Qwen3-VL-8B-Instruct              qwen3vl-8b    # 17.55 GB
dl model   allenai/Molmo2-4B                      molmo2-4b     # 19.42 GB (trust_remote_code)
dl dataset mm-eval/Visual7W                       visual7w      # 5.70 GB
dl dataset Multimodal-Fatima/VQAv2_train          vqav2-train   # 13.85 GB
dl dataset Multimodal-Fatima/VQAv2_validation     vqav2-val     # 9.35 GB (held-out eval)
dl dataset Multimodal-Fatima/SNLI-VE_train        snlive-train  # 27.85 GB
dl dataset Multimodal-Fatima/SNLI-VE_test         snlive-test   # 0.94 GB (held-out eval)
wait
echo "[$(date '+%F %T')] ALL REPOS FINISHED" | tee logs/prefetch_parallel.log
