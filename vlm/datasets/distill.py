"""Soft-label distillation: use an external multimodal LLM API to assign soft labels to images
(the local 7B no longer acts as the teacher).

Status: code present, but NOT exercised in the released benchmark (no reported
results). Planned as future work; see README "Future directions".

API requirements: OpenAI-compatible chat.completions with image+text input and logprobs
enabled -- extract the logprobs of candidate codes A/B/C/D from the top_logprobs of the
first answer token, normalize with softmax, and write them as reference_distribution.
The local small model then learns them via lora_train --soft.

Typical configuration (environment variables):
  VL_TEACHER_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  VL_TEACHER_API_MODEL=qwen3-vl-plus
  VL_TEACHER_API_KEY=sk-...

Usage:
  python -m vlm.datasets.distill --data data/processed/snli_ve/train.jsonl \
      --out data/processed/snli_ve/train.distilled.jsonl
  # Can also be passed explicitly: --base-url ... --model ... --api-key sk-...
  # Images are uploaded inline as data:image/jpeg;base64; local files need no public access.
"""

import argparse
import base64
import json
import math
import mimetypes
import os

import requests

from ..config import (TEACHER_API_MODEL, TEACHER_API_BASE_URL,
                      TEACHER_API_KEY_ENV)
from ..scoring.serving_schema import parse_schema, build_field_prompt
from .public_benchmarks import dump_jsonl


def _image_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _candidate_probs(client_kwargs, model, image_path, field, context_text,
                     top_logprobs=20):
    """Call the API once and extract the (unnormalized) log-probabilities of candidate
    codes from the first token's top_logprobs.

    Returns {code: prob}, containing only candidate codes actually returned by the API;
    the caller is responsible for filling missing codes with 0 and normalizing.
    """
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
        {"type": "text", "text": build_field_prompt(field, context_text)},
    ]}]
    resp = requests.post(
        f"{client_kwargs['base_url'].rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {client_kwargs['api_key']}",
                 "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": 1,
              "logprobs": True, "top_logprobs": top_logprobs,
              "temperature": 0.0},
        timeout=120)
    resp.raise_for_status()
    data = resp.json()
    tops = (data["choices"][0].get("logprobs", {})
            .get("content", [{}])[0].get("top_logprobs", []))
    # Normalize token text: strip whitespace/periods and uppercase; the API may return forms like " A" or "B."
    raw = {}
    for item in tops:
        token = str(item.get("token", "")).strip().strip(".。").upper()
        if len(token) == 1 and token.isalpha():
            raw[token] = max(raw.get(token, -1e9), float(item["logprob"]))
    picked = {c: raw[c] for c in field.choices if c in raw}
    if not picked:
        return {}
    # Softmax over the candidate subset returned by the API; codes not returned are treated as near-zero probability
    m = max(picked.values())
    exps = {c: math.exp(v - m) for c, v in picked.items()}
    z = sum(exps.values())
    return {c: exps[c] / z for c in exps}


def distill_api(input_jsonl, output_jsonl, model, base_url, api_key, top_logprobs=20):
    client = {"base_url": base_url, "api_key": api_key}
    rows = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            fields = parse_schema(ex["questions"])
            state = ex.get("state") or {}
            context = json.dumps(state, ensure_ascii=False) if state else ""
            ref_dist = {}
            for field in fields:
                probs = _candidate_probs(client, model, ex["image"], field,
                                         context, top_logprobs)
                if not probs:
                    raise RuntimeError(
                        f"{ex['id']}.{field.name}: API returned no candidate-code logprob; "
                        "verify the service supports logprobs and that answer codes are single tokens")
                dist = {field.criteria[c]: round(probs.get(c, 0.0), 6)
                        for c in field.choices}
                s = sum(dist.values())
                ref_dist[field.name] = {k: round(v / s, 6) for k, v in dist.items()}
            ex["reference_distribution"] = ref_dist
            rows.append(ex)
    dump_jsonl(rows, output_jsonl)
    return output_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=TEACHER_API_MODEL, help="teacher model name on the API side")
    ap.add_argument("--base-url", default=TEACHER_API_BASE_URL)
    ap.add_argument("--api-key", default=os.environ.get(TEACHER_API_KEY_ENV, ""))
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()
    if not (args.model and args.base_url and args.api_key):
        raise SystemExit("missing teacher API configuration: --model/--base-url/--api-key "
                         f"(or set environment variables VL_TEACHER_API_MODEL / VL_TEACHER_API_BASE_URL / {TEACHER_API_KEY_ENV})")
    print(distill_api(args.data, args.out, args.model, args.base_url,
                      args.api_key, args.top_logprobs))


if __name__ == "__main__":
    main()
