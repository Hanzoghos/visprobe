# [Historical Archive] Module Acceptance Plan

> **Status: historical planning snapshot (pre-execution).** The project was split into seven
> sequentially dependent modules (`0 → 1 → … → 6`). At release time, **modules 0–4 have been
> validated on real hardware**, while **modules 5 (external-teacher distillation) and 6
> (quantization) contain code only and were never accepted / have no experimental results**.
> For what was actually delivered, see [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md),
> [`../../results/SUMMARY.md`](../../results/SUMMARY.md), and the current
> [`../RUNBOOK.md`](../RUNBOOK.md). This file is kept for provenance.

## Module overview

- **Module 0 — Environment gate**: `setup.sh`, working torch/CUDA, each experimental model
  loads and scores one image; A–Z single-token assertion passes.
- **Module 1 — Scoring core (critical path)**: `serving_schema.py` (choice/boolean schemas,
  prompt rendering) and `vlm_scorer.py` (logits → candidate softmax, probability output).
- **Module 2 — Data ingestion**: `public_benchmarks.py` conversion for SNLI-VE / VQAv2 /
  Visual7W; `qc.py` checks including train/test leakage; zero overlap confirmed.
- **Module 3 — Baseline evaluation**: per-model frozen evaluation on the three test sets
  (accuracy/ECE/Brier), producing the baseline JSONs.
- **Module 4 — LoRA fine-tuning + after evaluation**: `lora_train.py` with checkpoint/resume,
  warmup+cosine, gradient clipping, best-checkpoint selection; adapter-direct after
  evaluation. Qwen3-VL-2B/8B and Molmo2-4B fully accepted; Qwen2.5-VL-7B after-results
  pending.
- **Module 5 — Distillation (code only, NOT accepted)**: `distill.py`, external multimodal
  teacher API with first-token logprobs; no run produced reported data.
- **Module 6 — Quantization (code only, NOT accepted)**: `quantize.py`, 4-bit export; no
  validated quantized artefact.

## Acceptance rule

Run the module's commands from the current RUNBOOK; a module is accepted only when its checks
pass, and later modules must not start before earlier ones pass. Module 0–4 acceptance
evidence is the committed `results/` JSONs and `results/figures/`. Modules 5/6 should be
treated as future work rather than delivered functionality.
