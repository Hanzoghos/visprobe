"""Background batch baseline evaluation: each model is loaded only once and
evaluated on multiple datasets in a row.

Features:
- Every PROGRESS_EVERY items, print done/total/rate/ETA (during long runs,
  tail the log at any time to check progress);
- Each model x dataset result is written to disk as soon as it is computed;
  result files that already exist are skipped automatically (a rerun after
  interruption simply resumes);
- The result json fields match the CLI of vlm.evaluation.evaluate and can be
  fed directly to summarize.

Usage (run from the project root):
  PYTHONPATH=. python scripts/run_baseline_eval.py \
    --models qwen3vl2b qwen3vl8b qwen7b molmo4b \
    --specs snli_ve=data/processed/snli_ve/test.jsonl \
            vqa_v2-third=data/subsets/vqa_v2_test-third_seed42.jsonl \
            visual7w=data/processed/visual7w/test.jsonl
"""
import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from vlm.scoring.serving_schema import parse_schema
from vlm.scoring.vlm_scorer import VLMScorer
from vlm.evaluation import metrics

PROGRESS_EVERY = 200


def count_fields(jsonl_path):
    """Pre-count the total number of fields for progress percentages (the
    current datasets all have one field per image)."""
    n = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += len(json.loads(line)["questions"])
    return n


def evaluate_one(scorer, model_name, spec_name, jsonl_path, total_fields):
    preds, refs = [], []
    probs_all, corrects_all, scores_all = [], [], []
    confusion = defaultdict(lambda: defaultdict(int))
    done = 0
    t_start = time.time()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            image = Image.open(ex["image"]).convert("RGB")
            state = ex.get("state") or {}
            context = json.dumps(state, ensure_ascii=False) if state else ""
            for field in parse_schema(ex["questions"]):
                ref = ex["reference"][field.name]
                r = scorer.score_field(image, field, context)
                pred = r["value"]
                preds.append(pred)
                refs.append(ref)
                probs_all.append(r["scores"].get(ref, 0.0))
                corrects_all.append(int(pred == ref))
                scores_all.append(r["scores"])
                confusion[ref][pred] += 1
                done += 1
                if done % PROGRESS_EVERY == 0:
                    el = time.time() - t_start
                    rate = done / el
                    eta = (total_fields - done) / rate
                    print(f"    [{model_name}/{spec_name}] {done}/{total_fields} "
                          f"({done/total_fields*100:.1f}%) {rate:.1f} items/s "
                          f"ETA {eta/60:.0f}min", flush=True)

    return {
        "model": model_name,
        "dataset": f"{spec_name}-test",
        "n": len(preds),
        "accuracy": metrics.accuracy(preds, refs),
        "ece": metrics.expected_calibration_error(probs_all, corrects_all),
        "brier": metrics.brier(scores_all, refs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="short names from config.MODELS, e.g. qwen3vl2b qwen3vl8b qwen7b molmo4b")
    ap.add_argument("--specs", nargs="+", required=True,
                    help="dataset specs name=path-to-jsonl (name is used to name result files)")
    ap.add_argument("--outdir", default="data/heldout")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter directory: mount the same adapter for all model rounds (after evaluation)")
    args = ap.parse_args()

    specs = []
    for s in args.specs:
        name, path = s.split("=", 1)
        specs.append((name, path))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = "after" if args.adapter else "baseline"
    print(f"task: {len(args.models)} models x {len(specs)} datasets ({tag})", flush=True)
    summary = []
    for model_name in args.models:
        scorer = None
        for spec_name, path in specs:
            out = outdir / f"{tag}-{model_name}-{spec_name}.json"
            if out.exists():
                print(f"===== skipping {model_name}/{spec_name} ({out.name} already exists) =====",
                      flush=True)
                summary.append(json.loads(out.read_text(encoding="utf-8")))
                continue

            if scorer is None:
                print(f"\n===== loading model {model_name} =====", flush=True)
                t0 = time.time()
                scorer = VLMScorer(model_name, adapter=args.adapter)
                print(f"load complete in {time.time()-t0:.0f}s | "
                      f"VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

            total = count_fields(path)
            print(f"===== evaluating {model_name} / {spec_name} ({total} fields) =====",
                  flush=True)
            t0 = time.time()
            result = evaluate_one(scorer, model_name, spec_name, path, total)
            result["elapsed_s"] = round(time.time() - t0, 1)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            summary.append(result)
            print(f"  -> {out.name} | acc={result['accuracy']:.4f} "
                  f"ece={result['ece']:.4f} brier={result['brier']:.4f} "
                  f"elapsed {result['elapsed_s']/60:.1f}min", flush=True)

        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== all results summary =====", flush=True)
    print(f"{'model':<12}{'dataset':<22}{'acc':>8}{'ece':>8}{'brier':>8}{'n':>8}",
          flush=True)
    for r in summary:
        print(f"{r['model']:<12}{r['dataset']:<22}{r['accuracy']:>8.4f}"
              f"{r['ece']:>8.4f}{r['brier']:>8.4f}{r['n']:>8}", flush=True)
    print("BASELINE_EVAL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
