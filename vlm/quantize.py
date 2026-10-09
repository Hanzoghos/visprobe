"""4-bit quantization (bitsandbytes nf4).

Status: code present, but NOT exercised in the released benchmark (no reported
quantized-model results). Planned as future work for deployment on smaller GPUs.

Usage:
  python -m vlm.quantize --model models/lora-adapter/merged --out models/quantized-4bit
  python -m vlm.quantize --model Qwen/Qwen2.5-VL-7B-Instruct --out models/qwen7b-4bit

save_pretrained stores the quantized weights (including quant_state); on reload the bnb
config is still used. If deployment needs GGUF (llama.cpp), convert separately with
optimum-quanto; see requirements/quant.txt.
"""

import argparse

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from .config import MODEL_ID, MODEL_DIR, resolve_model
from .training.model_loading import load_model, load_processor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID, help="model path or id to quantize")
    ap.add_argument("--out", default=str(MODEL_DIR / "quantized-4bit"))
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    spec = resolve_model(args.model)
    trust_remote_code = bool(spec.get("trust_remote_code"))
    compute_dtype = getattr(torch, args.dtype)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=compute_dtype,
                             bnb_4bit_use_double_quant=True)
    model = load_model(spec["id"], quantization_config=bnb,
                       torch_dtype=compute_dtype, device_map="auto",
                       trust_remote_code=trust_remote_code)
    processor = load_processor(spec["id"], trust_remote_code=trust_remote_code)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"saved 4-bit model -> {args.out}")


if __name__ == "__main__":
    main()
