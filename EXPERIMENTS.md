# Experiment Reproduction

Exact steps, hyperparameters, and hardware behind every reported number. All randomness is
controlled with fixed seeds.

## 1. Hardware & software

- GPU: single NVIDIA RTX 3090, 24 GB; driver CUDA 13.2; PyTorch 2.14.1+cu130; Python 3.12
- Key dependency: **transformers == 4.57.1 (pinned, hard requirement — see below)**;
  peft / accelerate / bitsandbytes
- Precision: bfloat16; gradient checkpointing (`use_reentrant=False`)
- Observed training peak memory: 2B ≈ 5.7 GB, 8B ≈ 23.6 GB, Molmo2-4B ≈ 11.9 GB,
  Qwen2.5-VL-7B ≈ 19.7 GB

> transformers must be pinned to 4.57.x: Molmo2-4B remote code does not load on 5.x, and
> InternVL3.5 requires the official **-HF** checkpoint format. See
> [`docs/molmo2-training-fix.md`](docs/molmo2-training-fix.md).

## 2. Model registry

Registered in `vlm/config.py`; short name → HuggingFace id:

| Short name | Model | Family |
|---|---|---|
| `qwen3vl2b` | Qwen/Qwen3-VL-2B-Instruct | qwen3_vl |
| `qwen3vl8b` | Qwen/Qwen3-VL-8B-Instruct | qwen3_vl |
| `qwen7b` | Qwen/Qwen2.5-VL-7B-Instruct | qwen2_vl |
| `molmo4b` | allenai/Molmo2-4B | molmo2 (trust_remote_code) |
| ~~`qwen3b`~~ | Qwen/Qwen2.5-VL-3B-Instruct | retired; baseline retained for reference |

## 3. Data

- `vlm/datasets/public_benchmarks.py` converts Hugging Face datasets into the unified JSONL
  schema (`id` / `image` / `state` / `questions` / `reference`); images go to `data/raw/`.
- Training pools: SNLI-VE 529,527 / VQAv2-yn 166,878 / Visual7W 63,122 examples.
- **Mixed fine-tuning subset** (stratified, label-balanced, seed=42):
  ```bash
  PYTHONPATH=. python scripts/make_train_subset.py \
    --specs snli_ve=33333 vqa_v2=33333 visual7w=33333 \
    --out data/subsets/train_mix_33k_seed42.jsonl --seed 42
  ```
  Produces 99,999 examples (33,333 per dataset). Not tracked in git — regenerate with the
  command above.
- Evaluation sets: SNLI-VE test full 21,482; Visual7W test full 6,695;
  VQAv2 seeded random 1/3 subset 26,846 (`data/subsets/vqa_v2_test-third_seed42.jsonl`).
  The full 80k VQAv2 test accounts for ~74% of evaluation cost; the 1/3 subset has a
  standard error of roughly ±0.5 percentage points.
- Train/test id and image overlap: zero for all three datasets.

## 4. Unified fine-tuning configuration (identical for all models)

| Item | Value |
|---|---|
| Data | train_mix_33k_seed42.jsonl (99,999) |
| Epochs / total steps | 1 / 25,000 (batch size 4) |
| LoRA | r=8, lora_alpha=16, dropout=0.05, language-decoder linear layers only |
| Learning rate | peak 1e-4, 3% linear warmup, then cosine decay to 0 |
| Gradient clipping | max norm 1.0 |
| Micro-batching | per-example forward/backward, gradients accumulated (avoids multi-image OOM) |
| Seed | 42 |
| Checkpoints | `checkpoint-latest` every 1,000 steps (`--resume`); `checkpoint-best` by recent-window loss |

Example (8B):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python -m vlm.training.lora_train \
  --data data/subsets/train_mix_33k_seed42.jsonl --model-id qwen3vl8b \
  --epochs 1 --batch 4 --rank 8 --lr 1e-4 --grad-clip 1.0 --warmup-ratio 0.03 \
  --save-every 1000 --log-every 500 --resume --out models/multitask-qwen3vl8b
```

- For Molmo2-4B the trainer auto-applies `vlm/training/molmo_patch.py` (fixes the remote
  code's in-place crash) and skips its broken processor save; at evaluation the processor is
  always loaded from the base model.
- Final artefact: `models/multitask-<model>/final-from-best/`.
- Recommended loss watchdog: `bash scripts/watch_train_loss.sh <PID> <logfile>`.

## 5. Evaluation

The model never generates text. Only the final answer-position logits are used, restricted to
that question's candidate codes and renormalised with softmax; argmax gives the prediction.

```bash
# baseline
PYTHONPATH=. python scripts/run_baseline_eval.py --models qwen3vl8b --specs \
  snli_ve=data/processed/snli_ve/test.jsonl \
  vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl \
  visual7w=data/processed/visual7w/test.jsonl

# after (LoRA adapter, no model merge)
PYTHONPATH=. python scripts/run_baseline_eval.py --models qwen3vl8b --specs <same> \
  --adapter models/multitask-qwen3vl8b/final-from-best
```

Metrics: accuracy, ECE (expected calibration error), Brier score. Results are written to
`data/heldout/{baseline,after}-*.json`; the reported copies live in `results/`.

## 6. One-command pipeline

```bash
# 24-example smoke test → unified training (watchdog/resume) → after evaluation
bash scripts/train_eval_pipeline.sh <model-short-name>
```

## 7. Observed training cost (single 3090)

| Model | Speed per step | Training wall-clock | Macro Δacc |
|---|---|---|---|
| Qwen3-VL-2B | ~1.1 s (dedicated) | ~7.6 h | +0.146 |
| Qwen3-VL-8B | 2.37 s | 16.5 h | +0.045 |
| Molmo2-4B | 3.2–5.2 s | ~22–30 h (slower when run in parallel) | +0.032 |
| Qwen2.5-VL-7B | 3.81 s | 26.5 h | +0.084 |

## 8. Status notes

- **Qwen3-VL-2B / Qwen3-VL-8B / Molmo2-4B / Qwen2.5-VL-7B**: baseline + after complete for
  all three datasets (all four experimental models finished; the 7B after-evaluation
  completed 2026-10-10: macro acc 0.8155 → 0.8995, macro ECE 0.133 → 0.025).
- **Qwen2.5-VL-3B**: retired. A constant lr=2e-4 run without warmup or clipping diverged at
  ~step 2300; only its baseline is retained. The current default hyperparameters fix this
  failure mode.
- Modules `calibration.py`, `distill.py`, and `quantize.py` contain working code but have no
  reported experimental results; they are not part of the benchmark.

## 9. Known pitfalls (avoid when reproducing)

- Constant lr=2e-4 without warmup/clipping caused Qwen2.5-VL-3B to diverge at ~step 2300
  (window loss jumped to the random-guess line (ln2+ln3+ln4)/3 ≈ 1.059 and never recovered).
- The 8B peaks at 23.6 GB on large Visual7W images (<1 GB headroom). Use
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; if OOM occurs, restart with `--resume`.
