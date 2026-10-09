"""Training-compatibility patch for Molmo2 (allenai remote code).

Problem: in Molmo2Model.build_input_embeddings of the remote modeling_molmo2.py,
    x.view(-1, D)[is_image_patch] += image_features
performs an in-place indexed addition on the embedding output. This is harmless for
inference (no_grad), but during LoRA training enable_input_require_grads turns that tensor
into a requires_grad leaf, and the in-place op raises directly:
    RuntimeError: a view of a leaf Variable that requires grad is being used
    in an in-place operation.

Fix: rewrite equivalently as an out-of-place index_copy (numerically identical forward
pass, differentiable backward) and replace the method with an instance method after model
loading, without modifying HF cache files (the patch ships with the repo and is traceable
and reproducible).
"""
import types

import torch


def _build_input_embeddings_trainable(self, input_ids, images=None, token_pooling=None):
    """Line-by-line equivalent of the remote original; the only difference is that
    image-feature injection is non-in-place."""
    input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
    x = self.transformer.wte(input_ids)

    image_features = None
    if images is not None:
        image_features = self.vision_backbone(images, token_pooling).to(x.device)
        is_image_patch = input_ids.view(-1) == self.config.image_patch_id
        assert is_image_patch.sum() == len(image_features)
        flat = x.reshape(-1, x.shape[-1])
        idx = is_image_patch.nonzero(as_tuple=False).squeeze(-1)
        # Original: flat[idx] += image_features (in-place on a leaf; always crashes in training)
        # index_copy is out-of-place with a corresponding autograd backward; numerically identical to the original
        flat = flat.index_copy(
            0, idx, flat[idx] + image_features.to(dtype=flat.dtype))
        x = flat.view_as(x)

    x = self.transformer.emb_drop(x)
    return x, image_features


def apply_molmo2_patch(model):
    """Reach through the PeftModel wrapper, find the Molmo2Model instance by class name,
    and replace the method.

    Returns the number of modules replaced (normally 1; 0 means the model structure has
    changed and the caller should warn).
    """
    patched = 0
    for _, module in model.named_modules():
        if type(module).__name__ == "Molmo2Model" and hasattr(
                module, "build_input_embeddings"):
            module.build_input_embeddings = types.MethodType(
                _build_input_embeddings_trainable, module)
            patched += 1
    return patched
