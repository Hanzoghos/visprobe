"""Build a mixed training subset: sample N rows per dataset, stratified by
reference label, and merge them into one jsonl.

Rows are self-contained (the id carries a dataset prefix, image is a
project-relative path) and can be fed directly to lora_train for multi-task
mixed training. A fixed seed makes sampling reproducible; per-dataset/label
counts are printed to ensure traceability.

Usage:
  PYTHONPATH=. python scripts/make_train_subset.py \
    --specs snli_ve=5000 vqa_v2=5000 visual7w=5000 \
    --out data/subsets/train_mix_5k_seed42.jsonl --seed 42
"""
import argparse
import collections
import json
import os
import random


def stratified_sample(lines, k, seed):
    """Stratified proportional sampling by reference (its single field); any
    shortfall is filled randomly from the full pool."""
    rng = random.Random(seed)
    by_label = collections.defaultdict(list)
    for i, line in enumerate(lines):
        ex = json.loads(line)
        label = next(iter(ex["reference"].values()))
        by_label[str(label)].append(i)

    n = len(lines)
    picked = set()
    for label, idxs in by_label.items():
        quota = int(len(idxs) * k / n)
        picked.update(rng.sample(idxs, min(quota, len(idxs))))
    pool = [i for i in range(n) if i not in picked]
    need = k - len(picked)
    if need > 0:
        picked.update(rng.sample(pool, min(need, len(pool))))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", nargs="+", required=True,
                    help="dataset name=samples per dataset, e.g. snli_ve=5000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for si, spec in enumerate(args.specs):
            name, k = spec.split("=")
            k = int(k)
            path = f"data/processed/{name}/train.jsonl"
            lines = open(path, encoding="utf-8").readlines()
            picked = stratified_sample(lines, k, args.seed + si)
            dist = collections.Counter()
            for i in sorted(picked):
                line = lines[i].strip()
                out.write(line + "\n")
                for v in json.loads(line)["reference"].values():
                    dist[str(v)] += 1
            total += len(picked)
            print(f"{name}: {len(lines)} -> sampled {len(picked)} | label distribution {dict(dist)}",
                  flush=True)
    print(f"total {total} rows -> {args.out} (seed={args.seed})", flush=True)


if __name__ == "__main__":
    main()
