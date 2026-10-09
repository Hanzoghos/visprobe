"""Single-model evaluation: accuracy / ECE / Brier + confusion table + error rows
(mirrors nimble/evaluation/evaluate_public.py).

Usage:
  python -m vlm.evaluation.evaluate --data data/processed/snli_ve/test.jsonl
  python -m vlm.evaluation.evaluate --data ... --model-id models/snli_ve-lora-adapter/merged \
      --save data/heldout/snli_ve.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ..config import MODEL_ID
from ..scoring.serving_schema import parse_schema
from ..scoring.vlm_scorer import VLMScorer
from . import metrics


def evaluate(data_jsonl, model_id=MODEL_ID, family=None, errors_out=None, limit=None,
             adapter=None):
    scorer = VLMScorer(model_id=model_id, family=family, adapter=adapter)
    preds, refs = [], []
    probs_all, corrects_all = [], []
    scores_all = []
    errors = []
    confusion = defaultdict(lambda: defaultdict(int))

    with open(data_jsonl, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(preds) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            fields = parse_schema(ex["questions"])
            image = Image.open(ex["image"]).convert("RGB")
            state = ex.get("state") or {}
            context = json.dumps(state, ensure_ascii=False) if state else ""
            for field in fields:
                ref = ex["reference"][field.name]
                r = scorer.score_field(image, field, context)
                pred = r["value"]
                preds.append(pred)
                refs.append(ref)
                probs_all.append(r["scores"].get(ref, 0.0))
                corrects_all.append(int(pred == ref))
                scores_all.append(r["scores"])
                confusion[ref][pred] += 1
                if pred != ref:
                    errors.append({"id": ex["id"], "field": field.name,
                                   "ref": ref, "pred": pred, "scores": r["scores"]})

    result = {
        "model": model_id,
        "n": len(preds),
        "accuracy": metrics.accuracy(preds, refs),
        "ece": metrics.expected_calibration_error(probs_all, corrects_all),
        "brier": metrics.brier(scores_all, refs),
    }
    if errors_out:
        out = Path(errors_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return result, confusion, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-id", default=MODEL_ID,
                    help="registered short name / raw HF id / path of a finetuned merged model")
    ap.add_argument("--family", default=None,
                    help="explicitly specify when a merged path cannot be reverse-resolved via the registry (qwen2_vl/qwen3_vl/internvl/molmo2)")
    ap.add_argument("--errors-out", default=None)
    ap.add_argument("--save", default=None, help="save metrics as json (for summarize to aggregate)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N fields (sampling/smoke test; full run by default)")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter directory (after-(LoRA) evaluation without merging)")
    args = ap.parse_args()

    result, confusion, errors = evaluate(args.data, model_id=args.model_id,
                                         family=args.family,
                                         errors_out=args.errors_out,
                                         limit=args.limit,
                                         adapter=args.adapter)
    # All three datasets use the same filenames test.jsonl/train.jsonl, so the stem alone
    # cannot distinguish them in the summary; use "parent-dir-stem", e.g. snli_ve-test
    dp = Path(args.data)
    result["dataset"] = f"{dp.parent.name}-{dp.stem}"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nConfusion (reference -> prediction, error pairs only):")
    for ref, pred_map in confusion.items():
        for pred, n in pred_map.items():
            if pred != ref and n:
                print(f"  {ref} -> {pred}: {n}")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults -> {args.save}")


if __name__ == "__main__":
    main()
