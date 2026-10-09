# Molmo2-4B LoRA Training Crash — Fix Record

> Date: 2026-10-07
> Affected model: `allenai/Molmo2-4B` (short name `molmo4b`; transformers 4.57.1 + trust_remote_code)
> Status: fixed; CPU unit test passed; real-GPU smoke test passed end-to-end

---

## 1. Symptom

Frozen inference on Molmo2-4B works perfectly (the full baseline run completed: 0.872 / 0.867
/ 0.877 accuracy on the three datasets), but the first training forward/backward crashes:

```
File ".../allenai/Molmo2_hyphen_4B/<rev>/modeling_molmo2.py", line 1447, in build_input_embeddings
    x.view(-1, x.shape[-1])[is_image_patch] += image_features
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
```

- Reproducible on any gradient-enabled path (`model.train()` + LoRA + `enable_input_require_grads()`).
- Does not occur on the evaluation path (`torch.no_grad()` / `inference_mode()`), which is why
  smoke tests and baseline evaluation never exposed it.

## 2. Root cause

The remote model code in `Molmo2Model.build_input_embeddings` writes the vision backbone's
image features **in place** into the embedding rows occupied by image placeholder tokens:

```python
x = self.transformer.wte(input_ids)                                   # embedding output
x.view(-1, x.shape[-1])[is_image_patch] += image_features            # offending line
```

Why it only fails during training:

1. Evaluation runs under `no_grad`, so autograd does not track the in-place write.
2. LoRA fine-tuning calls `model.enable_input_require_grads()` (required with gradient
   checkpointing), which turns the input embedding output into a `requires_grad=True` **leaf**
   tensor so gradients can flow back to the LoRA parameters.
3. PyTorch autograd forbids in-place modification of leaf tensors that require grad, and
   indexed `+=` is an in-place op — hence the crash.

A full-file scan found exactly one such op on the training forward path. (The `weight.data`
write in `_init_weights` runs only at initialisation and is irrelevant.)

## 3. Fix

### 3.1 Operator rewrite: in-place → out-of-place (equivalent, differentiable)

The sliced in-place add is replaced by an `index_copy` expression that is numerically
identical but does not modify any leaf tensor; autograd supports its backward pass:

```python
# Remote original (in-place on a leaf — crashes in training)
x.view(-1, x.shape[-1])[is_image_patch] += image_features

# Fixed
flat = x.reshape(-1, x.shape[-1])
idx = is_image_patch.nonzero(as_tuple=False).squeeze(-1)
flat = flat.index_copy(0, idx, flat[idx] + image_features.to(dtype=flat.dtype))
x = flat.view_as(x)
```

- `index_copy` returns a **new** tensor; it is not in-place.
- The source values `flat[idx] + image_features` are exactly "original content plus image
  features at those positions", so the forward result is elementwise identical.
- `.to(dtype=...)` guards against vision/embedding dtype mismatch.

### 3.2 Monkey patch inside the project (HF cache left untouched)

The HF modules cache can be cleared or re-downloaded, so editing the remote file would be
lost and untraceable. The fix lives in
[`vlm/training/molmo_patch.py`](../vlm/training/molmo_patch.py):

- `_build_input_embeddings_trainable`: line-by-line equivalent of the remote method with the
  single non-in-place change;
- `apply_molmo2_patch(model)`: walks modules (through any PEFT wrapper), finds the
  `Molmo2Model` instance by class name, and replaces the bound method with
  `types.MethodType`; returns the hit count.

It is invoked automatically in
[`vlm/training/lora_train.py`](../vlm/training/lora_train.py) after loading and before
attaching LoRA, only when `family == "molmo2"`, with a hard assertion that exactly one module
was patched. The evaluation path is unchanged.

### 3.3 Second issue found in real-GPU smoke: Molmo2Processor cannot save

The 24-example smoke test passed all 12 training steps, then failed at save time:

```
File "transformers/processing_utils.py", line 860, in save_pretrained
    if self.audio_tokenizer is not None:
AttributeError: 'Molmo2Processor' object has no attribute 'audio_tokenizer'
```

transformers 4.57's `ProcessorMixin.save_pretrained` inspects attributes the custom Molmo2
processor does not define. Since the adapter evaluation always loads the processor from the
base model cache, the adapter directory never needs a saved processor: the trainer now skips
`processor.save_pretrained` at all three save points for `family == "molmo2"`. After the fix,
`adapter_config.json` + `adapter_model.safetensors` save correctly and
`evaluate --adapter <dir>` runs without merging.

## 4. Verification

1. **CPU numerical/gradient unit test**: the rewrite matches the original forward within
   `allclose(atol=1e-6)`, and `backward()` on a simulated requires-grad leaf produces finite
   gradients (the original in-place form raises at the same point).
2. **Real-GPU smoke test**: 24 examples (8 per dataset, covering 2/3/4 candidates) × 2 epochs
   complete; adapter saves; `evaluate --adapter` returns normal metrics on real images.
3. The full 100k × 1 epoch run then completed (best window loss 0.206); after-evaluation
   results are reported in `results/SUMMARY.md`.

## 5. Regression notes

After upgrading transformers or changing the Molmo2 checkpoint:

- a "patched 0 modules" error means the remote class name/structure changed — update the
  matching rule in `molmo_patch.py`;
- any new in-place crash should be handled the same way (out-of-place op, kept inside the
  patch file, with a unit test).

Minimal regression command:

```bash
PYTHONPATH=. python -m vlm.training.lora_train \
  --data <24-example smoke file> --model-id molmo4b \
  --epochs 1 --batch 4 --save-every 0 --no-best-ckpt --out models/_molmo_smoke
```

## 6. Related files

| File | Role |
|---|---|
| `vlm/training/molmo_patch.py` | non-in-place image-feature injection + instance patch |
| `vlm/training/lora_train.py` | auto-applies the patch for family=molmo2; skips processor saves |
| `scripts/train_eval_pipeline.sh` | smoke → train → after pipeline |
| Remote `modeling_molmo2.py:1447` | original offending line (not modified) |
