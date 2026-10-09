"""P0 smoke test: load each of the four experimental models on the GPU in
turn, run a forward pass on a real sample, and validate the candidate
probabilities.

GPU memory is released immediately after each model is tested; if any model
fails, print FAIL_<model> and continue with the next one.
"""
import gc
import json
import time

import torch
from PIL import Image

from vlm.scoring.vlm_scorer import VLMScorer
from vlm.scoring.serving_schema import parse_schema

MODELS = ["qwen3vl2b", "qwen7b", "qwen3vl8b", "molmo4b"]


def main():
    row = json.loads(open("data/processed/snli_ve/test.jsonl").readline())
    img = Image.open(row["image"]).convert("RGB")
    fields = parse_schema(row["questions"])
    ctx = json.dumps(row["state"], ensure_ascii=False)
    gold = row["reference"]["relation"]
    print(f"sample hypothesis: {row['state']['hypothesis']} | gold answer: {gold}", flush=True)

    results = {}
    for mid in MODELS:
        print(f"\n===== {mid} =====", flush=True)
        try:
            t0 = time.time()
            s = VLMScorer(mid)
            load_s = time.time() - t0
            vram_load = torch.cuda.memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            res = s.score(img, fields, context_text=ctx)
            fwd_s = time.time() - t0
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
            f = res["fields"]["relation"]
            psum = sum(f["scores"].values())
            ok = abs(psum - 1) < 1e-3
            print(f"load {load_s:.0f}s/{vram_load:.1f}GB | forward {fwd_s:.2f}s | peak {vram_peak:.1f}GB", flush=True)
            print(f"prediction {f['code']} {f['value']} {f['scores']} sum={psum:.6f} hit={'yes' if f['value']==gold else 'no'}", flush=True)
            results[mid] = "OK" if ok else f"FAIL(prob sum {psum})"
            del s
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[mid] = f"FAIL: {repr(e)[:120]}"
            gc.collect()
            torch.cuda.empty_cache()

    print("\n===== summary =====", flush=True)
    for k, v in results.items():
        print(f"{k}: {v}", flush=True)
    print("ALL_SMOKE_OK" if all(v == "OK" for v in results.values()) else "SOME_FAILED", flush=True)


if __name__ == "__main__":
    main()
