"""VLM scorer: read answer-token logits -> softmax (mirrors
nimble/scoring/parallel_scorer.py).

A single forward pass takes the logits at the answer position and softmaxes only the
candidate-code columns. No text is generated or parsed, so the probabilities are
natively calibratable.

Supports multiple model families (see family in config.MODELS):
  qwen2_vl / qwen3_vl / internvl: AutoProcessor.apply_chat_template + text=/images=
  molmo2                        : allenai custom processor (images=/text=), trust_remote_code
Whether A..Z is a single token for each model is strictly checked at initialization;
failure raises immediately (a pitfall P0 must handle).
"""

import torch
from transformers import AutoProcessor

from ..config import MODEL_ID, TEMPERATURE, CODE_TOKENS, resolve_model
from ..training.model_loading import load_model, load_processor
from .serving_schema import build_field_prompt


def build_model_inputs(processor, image, prompt_text, device, family="qwen2_vl"):
    """Build the model-input dict from image + text. Processor forms differ across VLMs
    and are dispatched by family.

    P0 real-hardware check: if a family errors on image fields, adjust here according to
    the error (common issues are documented in the RUNBOOK).
    """
    if family in ("qwen2_vl", "qwen3_vl", "internvl"):
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        # Official Qwen3-VL examples explicitly drop token_type_ids (some checkpoints error in the forward pass with it)
        if family == "qwen3_vl":
            inputs.pop("token_type_ids", None)
        return inputs.to(device)

    if family == "molmo2":
        # Molmo2 ships its own processing_molmo2.py: images= takes a list of PIL images, text= takes a string.
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return processor(images=[image], text=text, return_tensors="pt").to(device)

    raise ValueError(f"unknown model family {family!r}; register it in config.MODELS / build_model_inputs")


class VLMScorer:
    def __init__(self, model_id=MODEL_ID, family=None, temperature=TEMPERATURE,
                 dtype=None, device=None, trust_remote_code=None, adapter=None):
        spec = resolve_model(model_id)
        self.model_id = spec["id"]
        self.family = family or spec["family"]
        if trust_remote_code is None:
            trust_remote_code = bool(spec.get("trust_remote_code"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Without an explicit dtype, from_pretrained defaults to fp32 (8B is about 35GB,
        # guaranteed OOM on a 24GB card): default to bfloat16 on GPU; fall back to fp32 only without a GPU.
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.processor = load_processor(self.model_id, trust_remote_code)
        self.model = load_model(self.model_id, torch_dtype=dtype,
                                trust_remote_code=trust_remote_code,
                                device_map="auto" if self.device == "cuda" else None)
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        # After (LoRA) evaluation: attach the LoRA adapter directly (the adapter is only a
        # few hundred MB, avoiding the disk usage of a merge)
        self.adapter = adapter
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.temperature = temperature
        # Every code must be a single token, otherwise the answer-position logits are not clean.
        self.code_ids = {c: self._single_token_id(c) for c in CODE_TOKENS}

    def _single_token_id(self, text):
        ids = self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"{text!r} should encode to a single token, got {ids} (choose a code known to be a single token, or use the first answer token)")
        return ids[0]

    def _answer_logits(self, image, prompt_text):
        """Return the logits vector R^|vocab| at the answer position."""
        inputs = build_model_inputs(self.processor, image, prompt_text,
                                    self.device, self.family)
        with torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1, :].float()

    def score_field(self, image, field, context_text=""):
        logits = self._answer_logits(image, build_field_prompt(field, context_text))
        cand_ids = [self.code_ids[c] for c in field.choices]
        cand_logits = logits[cand_ids]
        probs = torch.softmax(cand_logits / self.temperature, dim=-1)
        best = int(torch.argmax(cand_logits).item())
        return {
            "name": field.name,
            "type": field.type,
            "code": field.choices[best],
            "value": field.criteria[field.choices[best]],
            "scores": {field.criteria[c]: round(probs[i].item(), 6) for i, c in enumerate(field.choices)},
            "logits": {c: round(cand_logits[i].item(), 6) for i, c in enumerate(field.choices)},
        }

    def score(self, image, fields, context_text=""):
        per_field = {f.name: self.score_field(image, f, context_text) for f in fields}
        return {
            "model": self.model_id,
            "temperature": self.temperature,
            "output": {n: r["value"] for n, r in per_field.items()},
            "fields": per_field,
        }
