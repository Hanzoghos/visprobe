"""Load the VLM base model + processor (mirrors nimble/training/model_loading.py)."""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_processor(model_id, trust_remote_code=False):
    # The InternVL3.x tokenizer has a known mistral-regex defect; newer transformers provides a fix switch
    try:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, fix_mistral_regex=True)
    except TypeError:
        return AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)


def load_model(model_id, trust_remote_code=False, **kwargs):
    """Prefer image-text-to-text; fall back to vision2seq on failure (compatibility with
    different registered names).

    Repos such as molmo2 / internvl ship their own modeling_*.py and require trust_remote_code=True.
    """
    kwargs.setdefault("trust_remote_code", trust_remote_code)
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ValueError, KeyError, ImportError):
        from transformers import AutoModelForVision2Seq
        kwargs.pop("trust_remote_code", None)
        return AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, **kwargs)


def _language_decoder(model):
    """Take only the language decoder where possible (LoRA is not attached to the vision
    tower, saving VRAM and avoiding needless ViT training)."""
    # Most VLMs: <VLMForConditionalGeneration>.language_model or .model.language_model
    for obj in (model, getattr(model, "model", None)):
        if obj is not None and hasattr(obj, "language_model"):
            lm = obj.language_model
            if any(isinstance(m, torch.nn.Linear) for m in lm.modules()):
                return lm
    return model


def find_linear_names(model):
    """Automatically find the names of linear layers in the language decoder that LoRA can
    attach to (matched by suffix against PEFT target_modules), excluding lm_head; decoder
    naming differs across VLMs (q_proj/wqkv/gate_proj...)."""
    root = _language_decoder(model)
    names = set()
    for n, m in root.named_modules():
        if isinstance(m, torch.nn.Linear):
            names.add(n.split(".")[-1])
    return sorted(names - {"lm_head"})
