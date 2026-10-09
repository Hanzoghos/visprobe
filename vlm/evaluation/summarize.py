"""Multi-dataset summary: combine result.json files saved by evaluate into one table.

Mirrors nimble/evaluation/summarize_public_suite.py.
Usage:
  python -m vlm.evaluation.summarize data/heldout/snli_ve.json data/heldout/vqa_v2.json data/heldout/visual7w.json
"""

import argparse
import json


def summarize(result_files):
    rows = []
    for rf in result_files:
        rows.append(json.load(open(rf, encoding="utf-8")))
    return rows


def render(rows):
    header = f"{'dataset':<30} {'model':<34} {'acc':>7} {'ece':>7} {'brier':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        ds = r.get("dataset", "?")
        lines.append(f"{ds:<30} {r['model']:<34} {r['accuracy']:>7.4f} "
                     f"{r['ece']:>7.4f} {r['brier']:>7.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="one or more result.json files saved by evaluate")
    args = ap.parse_args()
    print(render(summarize(args.results)))


if __name__ == "__main__":
    main()
