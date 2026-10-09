# Plan (Historical)

> **This file is an archived plan.** The original (Windows-era) planning content, including
> its later revisions, is preserved in English at
> [`docs/archive/original-plan.md`](docs/archive/original-plan.md).
>
> For the project as it actually shipped, see:
> - [`README.md`](README.md) — overview, motivation, and current results
> - [`EXPERIMENTS.md`](EXPERIMENTS.md) — executed protocol and hyperparameters
> - [`results/SUMMARY.md`](results/SUMMARY.md) — measured numbers
>
> Notable differences between the plan and execution: Qwen3-VL-2B replaced the retired
> Qwen2.5-VL-3B as the lightweight model; InternVL3.5-8B stayed at frozen baseline only;
> VQAv2 evaluation uses a seeded 1/3 subset; training defaults to a 100k mixed subset over
> one epoch with lr 1e-4 + warmup/cosine + gradient clipping (the original constant-lr plan
> caused a divergence, documented in `EXPERIMENTS.md`).
