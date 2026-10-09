"""Multi-model cross-comparison: compare several models on the same dataset
(baseline vs finetuned vs quantized).

Mirrors nimble/evaluation/compare_public.py.
Usage:
  python -m vlm.evaluation.compare --data data/processed/snli_ve/test.jsonl \
      --models qwen3vl8b models/multitask-qwen3vl8b/final-from-best
"""

import argparse

from .evaluate import evaluate


def compare(data_jsonl, model_ids):
    rows = []
    for mid in model_ids:
        result, _, _ = evaluate(data_jsonl, model_id=mid)
        rows.append(result)
    return rows


def render(rows):
    header = f"{'model':<42} {'acc':>7} {'ece':>7} {'brier':>7} {'n':>6}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['model']:<42} {r['accuracy']:>7.4f} {r['ece']:>7.4f} "
                     f"{r['brier']:>7.4f} {r['n']:>6}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    print(render(compare(args.data, args.models)))


if __name__ == "__main__":
    main()
