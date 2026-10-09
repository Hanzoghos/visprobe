# Runbook

Commands for environment setup, data ingestion, evaluation, and fine-tuning. Run everything
from the repository root with the virtualenv activated (`source .venv/bin/activate`).

## P0 Environment & smoke tests

Models and datasets must be downloaded first; this step installs dependencies and smoke-tests
the four experimental models.

```bash
bash setup.sh                 # venv + deps + torch/CUDA verification
cp .env.example .env          # HF_ENDPOINT mirror + optional model/dataset ids
source .venv/bin/activate

# Smoke-test every experimental model: load + A/B/C/D single-token assertion
python - <<'PY'
from vlm.scoring.vlm_scorer import VLMScorer
for name in ["qwen3vl2b", "qwen3vl8b", "qwen7b", "molmo4b"]:
    s = VLMScorer(name)          # asserts A–Z encode to single tokens at init
    print("OK:", name, s.model_id, s.family)
PY
```

Per-model checks: (1) weights load in bf16; (2) the family-specific `build_model_inputs`
matches the real processor (molmo2 uses a custom processor and `trust_remote_code=True`);
(3) candidate codes are single tokens (raises otherwise); (4) one known-answer image yields
sensible answer-position logits.

> transformers must stay pinned to 4.57.1. Molmo2 needs the in-place training patch, which is
> applied automatically by the trainer — see
> [`molmo2-training-fix.md`](molmo2-training-fix.md).

## P1 Data ingestion (fixed candidate sets)

Convert all three datasets to train/test JSONL (small parquet reads from the HF cache;
`--limit N` is useful for a first smoke run):

```bash
for ds in snli_ve vqa_v2 visual7w; do
  python -m vlm.datasets.public_benchmarks --dataset $ds --split train
  python -m vlm.datasets.public_benchmarks --dataset $ds --split test
done

for ds in snli_ve vqa_v2 visual7w; do
  python -m vlm.datasets.qc --data data/processed/$ds/train.jsonl \
      --overlap-with data/processed/$ds/test.jsonl
done
```

- snli_ve: official train/test splits; premise/hypothesis go into `state`.
- vqa_v2: train=VQAv2_train, test=VQAv2_validation; only `answer_type=yes/no`; reference is
  JSON true/false.
- visual7w: official test only; the converter hashes `source_id` to deterministically split
  images into local train/test with no leakage.

Build the training subset and the VQAv2 evaluation subset:

```bash
PYTHONPATH=. python scripts/make_train_subset.py \
  --specs snli_ve=33333 vqa_v2=33333 visual7w=33333 \
  --out data/subsets/train_mix_33k_seed42.jsonl --seed 42
```

## P2 Baseline → fine-tune → after (order matters)

The primary lightweight model is `qwen3vl2b`. **Always evaluate the frozen baseline first**,
fine-tune second, and re-evaluate on the identical test files.

```bash
SPECS="snli_ve=data/processed/snli_ve/test.jsonl \
  vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl \
  visual7w=data/processed/visual7w/test.jsonl"

# 1) Frozen baseline (all three datasets)
PYTHONPATH=. python scripts/run_baseline_eval.py --models qwen3vl2b --specs $SPECS

# 2) LoRA fine-tune (100k mixed, 1 epoch, unified hyperparameters)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. python -m vlm.training.lora_train \
  --data data/subsets/train_mix_33k_seed42.jsonl --model-id qwen3vl2b \
  --epochs 1 --batch 4 --rank 8 --lr 1e-4 --grad-clip 1.0 --warmup-ratio 0.03 \
  --save-every 1000 --log-every 500 --resume --out models/multitask-qwen3vl2b

# 3) After evaluation — LoRA adapter directly, no weight merge
PYTHONPATH=. python scripts/run_baseline_eval.py --models qwen3vl2b --specs $SPECS \
  --adapter models/multitask-qwen3vl2b/final-from-best
# Criterion: after.accuracy > baseline.accuracy
```

The one-command equivalent (includes a 24-example three-task smoke test and a loss watchdog):

```bash
bash scripts/train_eval_pipeline.sh qwen3vl2b
```

Run the same flow for `qwen3vl8b`, `molmo4b`, and `qwen7b`. Do **not** run two trainings
concurrently unless combined peak memory fits (2B training ~5.7 GB is the only comfortable
parallel case on the 24 GB card).

## P3 Comparison & summary

```bash
# Per-metric JSON tables live in data/heldout/ (reported copies in results/)
python -m vlm.evaluation.summarize data/heldout/baseline-*.json data/heldout/after-*.json

# Regenerate the README figure after new results land
python scripts/make_charts.py
```

## Optional modules (code present, not validated in this release)

These commands describe existing utilities; **none has reported experiment results** in the
benchmark.

### Temperature calibration (`vlm/scoring/calibration.py`)

```python
from vlm.scoring.calibration import fit_temperature
# Collect (candidate logits, gold index) on a calibration set; fit_temperature returns the
# optimal T; pass it as VLMScorer(temperature=T).
```

### External-teacher distillation (`vlm/datasets/distill.py`)

```bash
export VL_TEACHER_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export VL_TEACHER_API_MODEL=qwen3-vl-plus
export VL_TEACHER_API_KEY=sk-...
python -m vlm.datasets.distill --data data/processed/snli_ve/train.jsonl \
    --out data/processed/snli_ve/train.distilled.jsonl
# Requires an image+text API exposing first-token logprobs; retrain with --soft.
```

### 4-bit quantization (`vlm/quantize.py`)

```bash
python -m vlm.quantize --model <model-or-adapter> --out models/quantized-4bit
```

## Troubleshooting

- **Model download fails**: check `HF_ENDPOINT` in `.env` or set `HTTPS_PROXY`. Large LFS
  shards failing over the xet bridge can be fetched from ModelScope instead.
- **Processor error on image inputs**: adjust the family branch in
  `vlm/scoring/vlm_scorer.py::build_model_inputs` (argument shapes differ across families).
- **Candidate code not a single token**: pick codes that are single tokens under the model's
  tokenizer (the scorer asserts this).
- **OOM during training**: use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and restart
  with `--resume`; checkpoints land every 1,000 steps.
- **Molmo2 save error (`audio_tokenizer`)**: expected on the remote processor — the trainer
  skips processor saves for molmo2 automatically.
