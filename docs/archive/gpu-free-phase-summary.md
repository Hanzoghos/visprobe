# [Historical Archive] GPU-Free Preparation Phase Summary

> **Status: dated historical record (2026-09-25 → 2026-09-26).** Before GPU hardware arrived,
> all CPU-feasible work was completed on a 2-core / 1.9 GB RAM Linux box. Kept for provenance;
> current setup and results live in [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md).

## Environment facts (measured at the time)

| Item | Measured | Response |
|---|---|---|
| GPU | none (`nvidia-smi` unavailable) | downloads / data / logic only; model forwards deferred |
| RAM | 1.9 GB (+8 GB swap) | strictly small-batch streaming for data work |
| Python | miniconda 3.14.7 | no torch wheel for cp314 → torch installed later on Python 3.10–3.12 |
| Disk | 196 GB (158 GB free) | all caches and data on the project disk |
| Network | HF direct ~0.1 MB/s; hf-mirror 1.5–3.5 MB/s | `HF_ENDPOINT=https://hf-mirror.com` throughout |

## What was completed

1. **All five base models downloaded and checksum-verified** (shasum against HF metadata),
   including a ModelScope fallback for large shards when the xet bridge stalled.
2. **transformers pinning**: 4.57.1 established as the only compatible version — Molmo2 remote
   code requires 4.57.x, and InternVL3.5 must use the official `-HF` checkpoint format.
3. **Data conversion**: 868,241 training records and 108,714 test records across SNLI-VE,
   VQAv2 yes/no, and Visual7W; images deduplicated; QC scans run (field schema, label
   balance, reference validity, id/image existence, train-test leakage = 0).
4. **Code landed**: scorer, schema, trainer, dataset converters/orchestrator, evaluation
   metrics/CLI — all implemented before GPU arrival.
5. **CPU-only verifications**: candidate single-token encoding logic, schema parsing, JSONL
   round-trips, and data statistics.

## Lessons carried into the GPU phase

- `datasets` streaming default batch sizes OOM on 1.9 GB RAM → use small explicit batch /
  pyarrow direct reads.
- Xet protocol errors are resolved with `HF_HUB_DISABLE_XET=1` or ModelScope direct links;
  downloaded blobs must be sha256-verified and placed into the snapshot layout manually when
  bypassing the CLI.
- HF background non-daemon threads can hang process exit; CLI runners call `os._exit(0)`.
- Model/dataset paths are project-relative for portability.
