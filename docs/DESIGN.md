# Design Document

## 1. Positioning

VisProbe explores whether **lightweight open-source VLMs (≤8B)** can serve as reliable
**probabilistic decision-makers**. It ports the "typed decision + probability" idea from the
text-only Nimble scoring framework to the multimodal setting.

- Input: **image + optional text** (natural images and visual reasoning).
- Output: for each field, a **typed value** plus a **probability distribution** (currently
  `choice` / `boolean` field types).
- Base models: open-source VLMs from 2B to 8B; the primary lightweight model is
  **Qwen3-VL-2B**.
- Probabilities: **answer-token logits → softmax over candidates** — no generation, no text
  parsing. The output is quantifiable, calibratable, and auditable.

Difference from Nimble: Nimble is text-only with single-token codes and shared-prefix parallel
scoring. This repository adds the image modality with newly written code, while keeping the
**same probability mechanism** (read only the candidate-token logits).

## 2. Probability mechanism

```
[image → vision encoder] + [system: classify by schema…] + [user: schema + question + "Requested field: X"]
        │  language-decoder forward pass
        ▼
answer-position logits ∈ R^|V|
        │  gather only the candidate-code columns (candidate projection)
        ▼
candidate logits ∈ R^k  ── softmax(logits / T) ──▶ probabilities ∈ R^k  ── argmax ──▶ chosen code
```

- Each candidate code must encode to a **single token** (letter codes A/B/C/D mapped to real
  values), so the answer position yields a clean logit vector.
- Field types mirror Nimble: `choice` (fixed closed set, 2–26 candidates) and `boolean`
  (JSON false/true; code order is fixed as `A=false`, `B=true`).
- Training target = cross-entropy on the answer token (soft-target CE supported for future
  distillation).
- Temperature T defaults to 1.0 and can be fitted on a calibration set with
  `scoring/calibration.py`.

Each field is rendered into a Nimble-style English prompt (`build_field_prompt`, via the chat
template):

```text
Classify the context using the supplied schema. Use the image and the context to answer the requested field. Return only that choice's one-letter code with no other text.
Context: {"premise": "...", "hypothesis": "..."}          # omitted when state is empty
Schema: [{"name": "relation", "description": "<instructions>",
          "choices": [{"code": "A", "value": "entailment"}, {"code": "B", "value": "neutral"}, {"code": "C", "value": "contradiction"}]}]
Requested field: relation
```

At training time one field expands to one example: input = image + the prompt above; logits
are taken at the final position, candidate-code columns are gathered, and CE is computed
against the gold answer letter's single token — the full vocabulary is never touched and no
generated text is parsed.

## 3. Repository layout and Nimble mapping

| This repo | Nimble counterpart | Role |
|---|---|---|
| `vlm/scoring/serving_schema.py` | `serving_schema.py` | choice/boolean schema + prompt rendering |
| `vlm/scoring/vlm_scorer.py` | `parallel_scorer.py` | core scorer (logits → softmax) |
| `vlm/scoring/calibration.py` | `calibration.py` | temperature scaling (code exists; no reported experiment) |
| `vlm/training/model_loading.py` | `model_loading.py` | base-model loading |
| `vlm/training/lora_train.py` | `schema_train.py` | LoRA fine-tuning (with checkpointing/scheduling) |
| `vlm/training/molmo_patch.py` | — | Molmo2 remote-code compatibility patch |
| `vlm/datasets/public_benchmarks.py` | `public_benchmarks.py` | public dataset conversion |
| `vlm/datasets/distill.py` | distillation concept | soft-label generation via external API (code exists; not run) |
| `vlm/datasets/qc.py` | — | data quality checks / stats |
| `vlm/datasets/orchestrator.py` | — | pipeline orchestration |
| `vlm/evaluation/evaluate.py` | `evaluate_public.py` | single-model evaluation |
| `vlm/evaluation/compare.py` | `compare_public.py` | model comparison |
| `vlm/evaluation/summarize.py` | `summarize_public_suite.py` | multi-dataset aggregation |
| `vlm/quantize.py` | — | 4-bit quantization (code exists; no reported experiment) |

## 4. Data format (JSONL, one example per line)

```json
{
  "id": "snli_ve-train-0000123",
  "image": "data/raw/snli_ve/images/2248275918.jpg",
  "state": {"premise": "A group of people are standing outside a building.",
            "hypothesis": "People are indoors."},
  "questions": {"relation": {"type": "choice",
    "instructions": "Assume the image and the premise are true. How does the hypothesis relate to them?",
    "criteria": {"A": "entailment", "B": "neutral", "C": "contradiction"}}},
  "reference": {"relation": "contradiction"}
}
```

- `state` is optional textual context (premise/hypothesis for SNLI; `{}` for VQA/Visual7W),
  rendered as the `Context:` line.
- In `questions`, `criteria` maps code → value; key order is candidate order (2–26 choices;
  boolean omits criteria and fixes A=false, B=true).
- `reference` is the hard label: a candidate value string for choice, JSON true/false for
  boolean.
- `reference_distribution` is an optional soft label (reserved for future distillation; the
  released data does not include it).

## 5. Data construction (closed candidate sets only)

Candidate sets are **fixed across examples**; distractors are never mined from sibling
answers of the same image.

- **SNLI-VE** (`Multimodal-Fatima/SNLI-VE_{train,test}`, images embedded in parquet):
  image + premise/hypothesis (in `state`) → `entailment/neutral/contradiction`, a fixed
  3-way `choice`; ~530k training pairs; images are deduplicated by `filename` (Flickr30k reuse).
- **VQAv2** (`Multimodal-Fatima/VQAv2_{train,validation}`): only `answer_type=yes/no`
  questions → `boolean`, reference is JSON true/false; the validation split is the held-out
  test set (a seeded 1/3 subset is used for evaluation).
- **Visual7W** (`mm-eval/Visual7W`; official test split, 69,817 MCQs): the question itself
  provides four fixed A–D options → `choice` (4 options, values vary per question); the
  converter splits train/test locally by hashing `source_id` so images never leak.
- **Quality control**: `qc.py` verifies field distributions, candidate balance, references
  inside criteria, soft-label sums, image existence, and id uniqueness; `--overlap-with`
  checks train/test contamination.
- **Closed loop**: convert → QC → **frozen-baseline evaluation** → LoRA fine-tuning →
  **after evaluation** → error analysis (`orchestrator.py`: baseline/train/eval stages).

Free-form answer datasets and ordered-score fields are out of scope for this release.

## 6. Model registry

Four models participate in the reported experiments; the family name determines processor
call conventions in `build_model_inputs`, and molmo2 requires `trust_remote_code=True`.
A–Z single-token encodings are asserted at `VLMScorer` initialisation for every model.

| Short name | HF id | Family | Status |
|---|---|---|---|
| `qwen3vl2b` ⭐ | `Qwen/Qwen3-VL-2B-Instruct` | qwen3_vl | primary lightweight; baseline + after |
| `qwen3vl8b` | `Qwen/Qwen3-VL-8B-Instruct` | qwen3_vl | baseline + after |
| `molmo4b` | `allenai/Molmo2-4B` | molmo2 | trust_remote_code; baseline + after |
| `qwen7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | qwen2_vl | baseline + after |

## 7. Evaluation metrics

- **accuracy**: fraction of predicted values equal to the reference.
- **ECE**: expected calibration error by confidence bin.
- **Brier**: standard multiclass Brier score (mean of Σ_c (p_c − onehot_c)²).

Numbers are only comparable within this setup: across VLMs, before/after fine-tuning, and
across closed-loop iterations — not against text-only systems.

## 8. Known risks / scope limits

1. Only fixed-candidate public tasks are covered (entailment / yes-no / 4-way MC); free-form
   QA, detection, OCR extraction, and counting are out of scope.
2. SNLI-VE/VQAv2 images are embedded per row in parquet, inflating download size (~28 GB /
   23 GB); the converter deduplicates images on disk.
3. Fine-tuning memory varies by image resolution; the 8B peaks at ~23.6 GB on a 24 GB card
   (use `expandable_segments:True` and `--resume`).
4. Absolute accuracy for smaller models is modest on language-heavy tasks (SNLI/VQA); the
   goal is a sound mechanism plus measurable relative gains.
