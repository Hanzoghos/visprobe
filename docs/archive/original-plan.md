# [Historical Archive] Original Plan — Local VLM Typed Decisions: Data + Training/Evaluation

> **Status: historical planning document (drafted 2026-09 on a Windows 8 GB laptop;
> updated through 2026-09-27 for the Linux/RTX-3090 execution).** It is preserved for
> provenance. Several decisions were later superseded — notably the model roster (InternVL
> became baseline-only and is not in the released results; Qwen2.5-VL-3B was retired and
> replaced by Qwen3-VL-2B), the VQAv2 evaluation subset, and the training hyperparameters.
> **For what was actually built and measured, see [`../../README.md`](../../README.md),
> [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md), and [`../../results/SUMMARY.md`](../../results/SUMMARY.md).**

## 0. Direction (decided at the time)

- Model type: **VLM** (multimodal), not a text-only LLM.
- Input: image (+ optional text); natural images and image-text reasoning.
- Probability output: **answer-token logits → candidate softmax**.
- Scale: small models preferred (originally ≤7B with 4-bit on 8 GB; later executed natively
  on a 24 GB RTX 3090).
- Data: public benchmarks (track B) + external large-VLM distillation (track C); no synthetic
  rendering.
- Tasks: typed `choice` / `boolean` decisions (ordered `score` deferred).

## 1. Initial inventory

No VLM support existed in the originating text-only codebase; scorer/trainer/evaluation all
had to be written. Reusable in spirit only: the candidate-logit probability mechanism, the
choice/boolean field vocabulary, and the accuracy/ECE/Brier metric conventions.

## 2. Environment strategy

Originally Windows with venv/cache redirected off the system drive; on Linux this became a
project-local `.venv` with `HF_HOME`/`HF_HUB_CACHE` under the project and
`HF_ENDPOINT=https://hf-mirror.com`. Dependencies: torch (CUDA), transformers, accelerate,
pillow, bitsandbytes, sentencepiece.

## 3. Probability mechanism (unchanged in the shipped design)

One forward pass, no text generation:

```
[image → vision encoder → image tokens] + [schema/user prompt]
   → language decoder → answer-position logits ∈ R^|V|
   → gather candidate-code columns → softmax(logits / T) → probabilities ∈ R^k
   → argmax → chosen code → typed value
```

Key points: candidate answers must be **single tokens** (letter codes A–Z); the training
target is cross-entropy on the answer token (equivalent to gathering full-vocab logits to the
candidate columns); probabilities are logits-derived and therefore calibratable; T defaults
to 1.0 and can be fitted.

## 4. Model registry as originally planned (5 models)

Qwen2.5-VL-3B/7B, Qwen3-VL-8B, InternVL3.5-8B (originally the GitHub custom format),
Molmo2-4B — all slated for baseline + fine-tune + after comparison. Notes recorded at the
time: a rumoured "Qwen3.5-VL-9B" did not exist; Molmo2-8B was 34.7 GB so 4B was chosen;
8/9B models would need `--4bit` on 8 GB cards.

**Supersession:** execution used transformers 4.57.1 (which required InternVL's official
`-HF` checkpoint); InternVL3.5-8B only received a frozen baseline; Qwen2.5-VL-3B was retired
after a divergence incident and replaced by Qwen3-VL-2B.

## 5. Data construction (tracks B + C)

### 5.1 Example format

The same unified JSONL used in the shipped code: `id`, project-relative `image`, optional
`state`, typed `questions` (choice/boolean criteria), and hard `reference`; optional
`reference_distribution` reserved for soft labels.

### 5.2 Track B — public image datasets (fixed candidate sets)

- **SNLI-VE** (`Multimodal-Fatima/SNLI-VE_{train,test}`) → fixed 3-way choice.
- **VQAv2 yes/no** (`Multimodal-Fatima/VQAv2_{train,validation}`) → boolean.
- **Visual7W** (`mm-eval/Visual7W`) → fixed 4-way choice, local image-hash split.

Images embedded in parquet are read with small pyarrow batches and deduplicated by native
image id. ChartQA/DocVQA were deferred (free-text answers).

### 5.3 Track C — external large-VLM distillation (soft labels)

The local 7B was explicitly **not** used as a teacher. Soft labels come from an external
OpenAI-compatible multimodal API exposing image input and first-token logprobs;
`vlm/datasets/distill.py` gathers A–D logprobs and writes normalised
`reference_distribution`. Configured via `VL_TEACHER_API_*` environment variables. Risk
noted: student inherits teacher bias.

### 5.4 Hard-example strategy without synthetic rendering

(1) error mining from `prediction != reference` rows; (2) paid distillation on frequent
error images; (3) free photometric augmentation for robustness only. The unlimited free
contrastive-pair advantage of synthetic rendering was explicitly given up.

### 5.5 Data closed loop (6 steps)

ingest(B)/distill(C) → train → evaluate on clean held-out → error mining → add same-class
data/distill hard cases → retrain/re-evaluate. A fixed held-out block never enters training.

## 6. Training plan (as planned)

LoRA (rank 8–16), 4-bit NF4 for 7/8B on small cards, answer-token cross-entropy, per-model
isolated artefacts, and the invariant order baseline → train → after. **Executed version:**
rank 8, language decoder only, bf16 on 3090, 100k mixed examples × 1 epoch, lr 1e-4 with
warmup+cosine, grad clip 1.0 — see `EXPERIMENTS.md`.

## 7. Evaluation plan

Clean held-out sets (official SNLI test, VQAv2 validation, local Visual7W split); metrics
accuracy/ECE/Brier; paired `baseline-*` / `after-*` JSONs per model; comparisons limited to
within this VLM setup. **Executed addition:** VQAv2 uses a seeded 1/3 subset (26,846) to cut
evaluation cost.

## 8. Phased rollout (P0–P5)

P0 environment; P1 public data + QC; P2 baseline→first fine-tune→after; P3 remaining models
+ error mining; P4 B-side closed loop; P5 external-API distillation.

## 9. Risks and honest notes (recorded at the time)

1. Without synthetic rendering the data-generation loop is weaker and depends on public
   coverage (B) and distillation budget (C).
2. 7B needed 4-bit on 8 GB with quality/speed cost — later moot on the 24 GB 3090.
3. Absolute accuracy of ≤3B models was expected to be modest; the goal was a sound mechanism
   with measurable relative gains.
4. Network access to Hugging Face was the largest operational uncertainty (mirror required).

## 10. Confirmed decisions (2026-09-26/27)

1. Execution environment: Linux server, project-disk venv/caches/models.
2. Five models to be compared (later revised — see registry supersession above).
3. Track-B datasets: SNLI-VE + VQAv2 yes/no + Visual7W.
4. Distillation via external API only.
5. Use hf-mirror, disable Xet.
6. First fine-tune via LoRA.
7. Invariant order: baseline → fine-tune → after, paired results per model.
8. GPU: **RTX 3090 24 GB chosen** so all models could train natively in bf16 without 4-bit,
   keeping the comparison fair; A800 40 GB as a faster backup.
