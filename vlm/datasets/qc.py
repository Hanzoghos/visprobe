"""Data QC / statistics: field distribution, candidate balance, label consistency,
contamination checks.

Usage:
  python -m vlm.datasets.qc --data data/processed/snli_ve/train.jsonl \
      --overlap-with data/processed/snli_ve/test.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ..scoring.serving_schema import parse_schema


def qc(jsonl_path):
    problems = []
    n = 0
    type_counter = Counter()
    nchoices = Counter()
    fields_counter = Counter()
    soft_ok = 0
    ids = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"JSON parse failed: {e}")
                continue
            if ex.get("id") in ids:
                problems.append(f"duplicate id: {ex.get('id')}")
            ids.add(ex.get("id"))
            if not Path(ex["image"]).exists():
                problems.append(f"{ex.get('id')} image does not exist: {ex['image']}")
            for field in parse_schema(ex["questions"]):
                fields_counter[field.name] += 1
                type_counter[field.type] += 1
                nchoices[len(field.choices)] += 1
                ref = ex["reference"].get(field.name)
                if ref is not None and ref not in field.criteria.values():
                    problems.append(f"{ex['id']}.{field.name} reference not in criteria: {ref}")
                if ex.get("reference_distribution"):
                    dist = ex["reference_distribution"].get(field.name, {})
                    s = sum(float(v) for v in dist.values())
                    if abs(s - 1.0) > 0.02:
                        problems.append(f"{ex['id']}.{field.name} soft-label sum {s:.3f} != 1")
                    else:
                        soft_ok += 1

    print(f"number of samples: {n}")
    print(f"field type distribution: {dict(type_counter)}")
    print(f"candidate-count distribution (num candidates -> num samples): {dict(sorted(nchoices.items()))}")
    print(f"field distribution: {dict(fields_counter)}")
    print(f"valid soft-label fields: {soft_ok}")
    print(f"number of problems: {len(problems)}")
    for p in problems[:50]:
        print(f"  - {p}")
    return problems


def overlap(a_jsonl, b_jsonl):
    """Check whether sample ids and image paths overlap between two files
    (train/held-out contamination prevention).

    The same image appearing on both the train and test side is image-level leakage,
    even if the questions differ.
    """
    def read(p):
        ids, imgs = set(), set()
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            ex = json.loads(l)
            ids.add(ex["id"])
            imgs.add(ex.get("image"))
        return ids, imgs
    ia, ima = read(a_jsonl)
    ib, imb = read(b_jsonl)
    inter = ia & ib
    print(f"overlapping ids: {len(inter)} ({a_jsonl} ∩ {b_jsonl})")
    for x in list(inter)[:20]:
        print(f"  - {x}")
    img_inter = (ima & imb) - {None}
    print(f"overlapping images: {len(img_inter)} (expected 0; Visual7W guarantees this via the source_id hash split)")
    for x in list(img_inter)[:20]:
        print(f"  - {x}")
    return inter | img_inter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--overlap-with", default=None, help="another jsonl to check for id overlap")
    args = ap.parse_args()
    qc(args.data)
    if args.overlap_with:
        overlap(args.data, args.overlap_with)


if __name__ == "__main__":
    main()
