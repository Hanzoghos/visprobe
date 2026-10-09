"""Centralized configuration: model registry, dataset ids, candidate codes, temperature.

All values can be overridden via environment variables (see .env.example).
"""

import json
import os
from pathlib import Path

# ---- Model registry ----
# Released experiment loop per model: frozen baseline evaluation -> LoRA fine-tuning
# -> same-protocol after evaluation (see README.md / EXPERIMENTS.md).
# Four models participate in the released benchmark (qwen3vl2b / qwen7b /
# qwen3vl8b / molmo4b); qwen3b is retired and internvl8b is registered for
# completeness but has no released result files.
#
# family determines how the processor inputs are built (see build_model_inputs in
# scoring/vlm_scorer.py):
#   qwen2_vl / qwen3_vl: AutoProcessor + apply_chat_template, text=/images= kwargs
#   internvl           : InternVL3.5 (-HF checkpoint), native 4.57 processor
#   molmo2             : allenai custom code; trust_remote_code is required and the processor form differs
# Large-model fine-tuning/inference on small VRAM may require --4bit; whether each
# model encodes A/B/C/D as single tokens is strictly checked at scorer init.
MODELS = {
    # Superseded by qwen3vl2b (user decision on 2026-10-07); no longer used for
    # training/evaluation. Kept in the registry only so older artifacts remain traceable.
    "qwen3b": {
        "id": os.environ.get("VL_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "family": "qwen2_vl", "size": "3B", "trust_remote_code": False,
        "note": "[Retired] Superseded by qwen3vl2b; old baseline results remain in data/heldout/"},
    "qwen3vl2b": {
        "id": os.environ.get("VL_QWEN3VL2B_ID", "Qwen/Qwen3-VL-2B-Instruct"),
        "family": "qwen3_vl", "size": "2B", "trust_remote_code": False,
        "note": "New-generation small primary model (Apache-2.0), same qwen3_vl family, natively supported in 4.57.1"},
    "qwen7b": {
        "id": os.environ.get("VL_QWEN7B_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "family": "qwen2_vl", "size": "7B", "trust_remote_code": False,
        "note": "Fine-tuning + cross-comparison member (4bit required on small VRAM); no longer used as the local distillation teacher"},
    "qwen3vl8b": {
        "id": os.environ.get("VL_QWEN3VL8B_ID", "Qwen/Qwen3-VL-8B-Instruct"),
        "family": "qwen3_vl", "size": "8B", "trust_remote_code": False,
        "note": "New-generation Qwen VL, cross-comparison member"},
    "internvl8b": {
        # Must use the -HF suffix (official standard transformers format); the unsuffixed
        # repo is the GitHub/custom format that only ships remote modeling code without a
        # matching processor. The native 4.57 InternVLProcessor is incompatible with its
        # Qwen2Tokenizer (start_image_token AttributeError).
        "id": os.environ.get("VL_INTERNVL8B_ID", "OpenGVLab/InternVL3_5-8B-HF"),
        "family": "internvl", "size": "8B", "trust_remote_code": False,
        "note": "Registered for completeness; not part of the released benchmark (no result files)"},
    "molmo4b": {
        "id": os.environ.get("VL_MOLMO4B_ID", "allenai/Molmo2-4B"),
        "family": "molmo2", "size": "4B", "trust_remote_code": True,
        "note": "Molmo2-4B (allenai custom code), cross-comparison member, official package about 19GB"},
}

# Default model (P0 smoke test / first training run): the primary lightweight model
MODEL_ID = MODELS["qwen3vl2b"]["id"]

# ---- Distillation teacher: moved to an external API (P5, optional) ----
# The API must be OpenAI chat.completions compatible and support image+text input plus
# logprobs (so the logprob of the first answer token is available). The four codes
# A/B/C/D are taken from top_logprobs and softmaxed to produce soft labels.
# The local 7B model no longer acts as the teacher.
TEACHER_API_MODEL = os.environ.get("VL_TEACHER_API_MODEL", "")          # e.g. qwen3-vl-plus
TEACHER_API_BASE_URL = os.environ.get("VL_TEACHER_API_BASE_URL", "")    # e.g. https://dashscope.aliyuncs.com/compatible-mode/v1
TEACHER_API_KEY_ENV = os.environ.get("VL_TEACHER_API_KEY_ENV", "VL_TEACHER_API_KEY")

# Temperature for softmax(logits / T); default 1.0, overridden after fitting on the calibration set (see scoring/calibration.py)
TEMPERATURE = float(os.environ.get("VL_TEMPERATURE", "1.0"))

# Single-letter candidate codes: A..Z (up to 26 candidates, each must encode to a single token)
CODE_TOKENS = [chr(ord("A") + i) for i in range(26)]

# ---- Datasets (fixed candidate sets; images are embedded in the parquet files) ----
DATASETS = {
    "snli_ve": {
        "train": (os.environ.get("VL_SNLI_VE_TRAIN_ID",
                                 "Multimodal-Fatima/SNLI-VE_train"), "train"),
        "test": (os.environ.get("VL_SNLI_VE_TEST_ID",
                                "Multimodal-Fatima/SNLI-VE_test"), "test"),
    },
    "vqa_v2": {
        "train": (os.environ.get("VL_VQAV2_TRAIN_ID",
                                 "Multimodal-Fatima/VQAv2_train"), "train"),
        "test": (os.environ.get("VL_VQAV2_VAL_ID",
                                "Multimodal-Fatima/VQAv2_validation"), "validation"),
    },
    "visual7w": {
        # mm-eval/Visual7W has only a single test split (69,817 MCQ items); the converter groups by source_id and splits it itself
        "train": (os.environ.get("VL_VISUAL7W_ID", "mm-eval/Visual7W"), "test"),
        "test": (os.environ.get("VL_VISUAL7W_ID", "mm-eval/Visual7W"), "test"),
    },
}

# Advanced datasets (no converter yet; free-text answers require a separate candidate
# scheme; ids are recorded for reference)
ADV_DATASETS = {
    "chartqa": os.environ.get("VL_CHARTQA_ID", "HuggingFaceM4/ChartQA"),
    "docvqa": os.environ.get("VL_DOCVQA_ID", "lmms-lab/DocVQA"),
}


# HF config.json model_type -> this project's family (used to auto-detect local merged paths)
MODEL_TYPE_TO_FAMILY = {
    "qwen2_5_vl": "qwen2_vl",
    "qwen3_vl": "qwen3_vl",
    "internvl_chat": "internvl",
    "molmo2": "molmo2",
}


def resolve_model(name_or_id):
    """Resolve a registry short name (e.g. qwen3b) to a spec dict; a bare HF id is
    reverse-looked-up by id; a local merged path has its config.json model_type read
    to infer the family; if nothing matches, fall back to qwen2_vl (use --family to
    specify it explicitly)."""
    if name_or_id in MODELS:
        return dict(MODELS[name_or_id], name=name_or_id)
    for name, spec in MODELS.items():
        if spec["id"] == name_or_id:
            return dict(spec, name=name)

    # Local path (LoRA merged / quantized artifact): read config.json for auto-detection
    cfg_path = Path(name_or_id) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        family = MODEL_TYPE_TO_FAMILY.get(cfg.get("model_type"))
        if family:
            return {"name": name_or_id, "id": name_or_id, "family": family,
                    "size": "?", "trust_remote_code": family == "molmo2"
                    or bool(cfg.get("auto_map")),
                    "note": f"Local path detected by model_type={cfg.get('model_type')}"}

    return {"name": name_or_id, "id": name_or_id, "family": "qwen2_vl",
            "size": "?", "trust_remote_code": False,
            "note": "Unregistered and unrecognized; falling back to qwen2_vl"}
