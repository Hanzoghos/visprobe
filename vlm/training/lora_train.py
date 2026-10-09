"""LoRA fine-tuning: cross-entropy on the answer token (mirrors the training script in
nimble/training).

Supports --soft: use reference_distribution as soft labels (distillation data) with
soft-target CE loss.
Usage:
  python -m vlm.training.lora_train --data data/processed/snli_ve/train.jsonl --out models/snli_ve-lora-adapter
  python -m vlm.training.lora_train ... --soft      # distillation data
  python -m vlm.training.lora_train ... --4bit      # required for 7B on 8GB
  python -m vlm.training.lora_train ... --merge     # merge the adapter into the base after training and save merged/ separately

Field types are choice / boolean: the gold code is uniformly reverse-looked-up from the
reference value via criteria (boolean is fixed to A=JSON false, B=JSON true; the order is
guaranteed by serving_schema).
"""

import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training, TaskType)

from ..config import MODEL_ID, CODE_TOKENS, resolve_model
from ..paths import MODEL_DIR
from ..scoring.serving_schema import parse_schema, build_field_prompt
from ..scoring.vlm_scorer import build_model_inputs
from .model_loading import load_model, load_processor, find_linear_names


def load_training_rows(jsonl_path, use_soft=False):
    """Expand jsonl into a list of (image, field, gold code, soft label), one entry per field."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            fields = parse_schema(ex["questions"])
            state = ex.get("state") or {}
            # Context gets the JSON of the whole state (e.g. SNLI premise/hypothesis); omit when empty
            context = json.dumps(state, ensure_ascii=False) if state else ""
            for field in fields:
                ref = ex["reference"][field.name]
                code = next(c for c, label in field.criteria.items() if label == ref)
                soft = None
                if use_soft and ex.get("reference_distribution"):
                    dist = ex["reference_distribution"].get(field.name, {})
                    # Soft labels follow candidate-code order; missing entries are filled with 0.
                    # JSON keys are strings; boolean candidate values are true/false, so use json.dumps for compatibility
                    soft = [float(dist.get(v, dist.get(json.dumps(v), 0.0)))
                            for c in field.choices for v in [field.criteria[c]]]
                    # Normalization happens in the training loop
                rows.append({"image": ex["image"], "field": field, "code": code,
                             "context": context, "soft": soft})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="processed jsonl")
    ap.add_argument("--model-id", default=None,
                    help="base model: short name in config.MODELS (qwen3vl2b/qwen3vl8b/qwen7b/molmo4b) or HF id")
    ap.add_argument("--family", default=None, help="override the auto-detected model family")
    ap.add_argument("--out", default=str(MODEL_DIR / "lora-adapter"))
    ap.add_argument("--epochs", type=int, default=1,
                    help="default 1 (unified setting: one epoch over about 100k mixed examples to avoid overfitting; validated effective on 8B)")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA peak learning rate (a constant 2e-4 without clipping diverged around step ~2300; lowered since)")
    ap.add_argument("--warmup-ratio", type=float, default=0.03,
                    help="linear warmup over the first 3% of steps, then cosine decay")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient clipping by max norm (prevents a single large-gradient sample from corrupting LoRA; 0=off)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--soft", action="store_true", help="train with soft labels (distillation data)")
    ap.add_argument("--4bit", action="store_true", help="required for 7B on 8GB")
    ap.add_argument("--merge", action="store_true", help="merge the adapter after training and save merged/ separately")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="overwrite checkpoint-latest every N optimizer steps (0=off)")
    ap.add_argument("--log-every", type=int, default=500, help="print recent-window loss every N steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="resume training from <out>/checkpoint-latest (restore adapter/optimizer/RNG)")
    ap.add_argument("--no-best-ckpt", action="store_true",
                    help="disable the best checkpoint (by default checkpoint-best is saved separately at the lowest recent-window loss)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = resolve_model(args.model_id or MODEL_ID)
    model_id = spec["id"]
    family = args.family or spec["family"]
    trust_remote_code = bool(spec.get("trust_remote_code"))
    print(f"Base model: {model_id} (family={family})")
    processor = load_processor(model_id, trust_remote_code=trust_remote_code)

    quant = None
    if getattr(args, "4bit"):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
    if quant:
        model = load_model(model_id, torch_dtype=torch.bfloat16,
                           quantization_config=quant, device_map="auto",
                           trust_remote_code=trust_remote_code)
    else:
        model = load_model(model_id, torch_dtype=torch.bfloat16,
                           trust_remote_code=trust_remote_code).to(device)

    # Molmo2 remote code uses an in-place op when injecting image features into the
    # embeddings, which always crashes with enable_input_require_grads; replace it with a
    # differentiable out-of-place implementation before training
    if family == "molmo2":
        from .molmo_patch import apply_molmo2_patch
        n_patched = apply_molmo2_patch(model)
        if n_patched != 1:
            raise RuntimeError(
                f"molmo2 patch matched {n_patched} Molmo2Model(s) (expected 1); "
                "the remote model structure may have changed, check modeling_molmo2.py")
        print("molmo2 in-place embedding patch applied", flush=True)

    # Gradient checkpointing: image sequences are often over a thousand tokens; without it
    # the 8B activations OOM on 24GB.
    # 4-bit LoRA requires prepare_model_for_kbit_training first, otherwise quantized weights have no gradients
    if quant:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    import random as _random
    from PIL import Image

    out_dir = pathlib.Path(args.out)
    ckpt_dir = out_dir / "checkpoint-latest"
    resume_from = args.resume and (ckpt_dir / "training_state.pt").exists()

    targets = find_linear_names(model) if args.rank else ["q_proj", "v_proj"]
    print("LoRA target layers (language decoder only):", targets)
    if resume_from:
        # Resume: re-attach the checkpoint's adapter directly onto the base model (is_trainable=True)
        model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=True)
        model.enable_input_require_grads()
        print(f"Resume mode: restoring adapter from {ckpt_dir}", flush=True)
    else:
        lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          target_modules=targets, task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora)
        model.enable_input_require_grads()  # required for checkpointing + LoRA
        model.print_trainable_parameters()

    code_ids = {c: processor.tokenizer(c, add_special_tokens=False)["input_ids"][0]
                for c in CODE_TOKENS}

    rows = load_training_rows(args.data, use_soft=args.soft)
    if not rows:
        raise SystemExit("no training samples; check the --data path")
    n_step_per_epoch = (len(rows) + args.batch - 1) // args.batch
    print(f"training samples {len(rows)} | {args.epochs} epoch | {n_step_per_epoch} step/epoch",
          flush=True)

    total_steps = n_step_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # warmup (linear ramp to peak lr over the first 3%) + cosine decay to ~0, avoiding late-stage divergence from a constant high lr
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps)
    rng = _random.Random(args.seed)
    start_epoch, start_batch, global_step = 0, 0, 0
    if resume_from:
        state = torch.load(ckpt_dir / "training_state.pt", map_location=device,
                           weights_only=False)
        rng.setstate(state["rng"])  # state before that epoch's shuffle, keeping data order reproducible
        start_epoch = state["epoch"]
        start_batch = state["next_batch"]
        global_step = state["global_step"]
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        print(f"resumed at epoch={start_epoch} batch={start_batch} "
              f"global_step={global_step}", flush=True)
    print(f"total steps {total_steps} | peak lr {args.lr} warmup "
          f"{args.warmup_ratio:.0%} + cosine | grad_clip={args.grad_clip}", flush=True)
    model.train()

    # The molmo2 remote Molmo2Processor.save_pretrained is incompatible with
    # transformers 4.57 (missing the audio_tokenizer attribute); moreover, during adapter
    # evaluation the processor is always loaded from the base model and need not be saved
    # with the adapter -- molmo2 always skips writing the processor to disk
    save_processor = (lambda d: None) if family == "molmo2" \
        else (lambda d: processor.save_pretrained(d))

    def save_checkpoint(ep, next_batch, gs, epoch_rng_state, subdir="checkpoint-latest"):
        d = out_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)
        save_processor(d)
        # Write a temp file first and atomically replace, avoiding a corrupt checkpoint if power fails mid-write
        tmp = d / "training_state.tmp"
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": epoch_rng_state,
                    "epoch": ep, "next_batch": next_batch, "global_step": gs}, tmp)
        tmp.replace(d / "training_state.pt")
        print(f"[{subdir}] step {gs} (epoch {ep} {next_batch}/{n_step_per_epoch})",
              flush=True)

    from collections import deque
    recent = deque(maxlen=args.log_every)   # recent-window batch loss, used to monitor the true instantaneous level and pick the best
    best_loss = float("inf")

    for ep in range(start_epoch, args.epochs):
        epoch_rng_state = rng.getstate()  # before this epoch's shuffle; persisted with the checkpoint
        rng.shuffle(rows)
        b0 = start_batch if ep == start_epoch else 0
        total = 0.0
        n_batches = 0
        for bi in range(b0, n_step_per_epoch):
            micro = rows[bi * args.batch:(bi + 1) * args.batch]
            opt.zero_grad()
            running = 0.0
            # Key: backprop each micro-batch immediately (loss / number of micro-batches);
            # never keep a whole batch of graphs alive -- four full forward graphs coexisting
            # would make the 8B model OOM on 24GB
            for r in micro:
                image = Image.open(r["image"]).convert("RGB")
                prompt = build_field_prompt(r["field"], r["context"])
                inputs = build_model_inputs(processor, image, prompt, device, family)
                out = model(**inputs)
                logits = out.logits[0, -1, :]
                cand_ids = [code_ids[c] for c in r["field"].choices]
                cand_logits = logits[cand_ids].unsqueeze(0)
                if r["soft"] is not None:
                    target = torch.tensor(r["soft"], dtype=torch.float32, device=device)
                    target = target / target.sum().clamp(min=1e-8)  # normalize soft labels
                    loss = F.cross_entropy(cand_logits, target.unsqueeze(0))
                else:
                    gold_idx = r["field"].choices.index(r["code"])
                    loss = F.cross_entropy(cand_logits, torch.tensor([gold_idx], device=device))
                (loss / len(micro)).backward()  # immediately free this sample's computation graph
                running += loss.item()
            # Gradient clipping: VLM vision sequences are over a thousand tokens and some
            # samples may produce large gradients; without clipping a single optimizer step
            # can corrupt the LoRA weights (the 3B model diverged around step ~2300)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            global_step += 1
            b_loss = running / len(micro)
            total += b_loss
            recent.append(b_loss)
            n_batches += 1
            if global_step % args.log_every == 0:
                win = sum(recent) / len(recent)
                print(f"epoch {ep} step {bi+1}/{n_step_per_epoch} "
                      f"(global {global_step}) epoch_avg {total/n_batches:.4f} "
                      f"win{args.log_every}_avg {win:.4f} lr {sched.get_last_lr()[0]:.2e}",
                      flush=True)
                # best: save separately based on recent-window loss, ensuring a usable best adapter remains even after divergence
                if win < best_loss and not args.no_best_ckpt:
                    best_loss = win
                    save_checkpoint(ep, bi + 1, global_step, epoch_rng_state,
                                    subdir="checkpoint-best")
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(ep, bi + 1, global_step, epoch_rng_state)
        print(f"epoch {ep} done epoch_avg {total / max(n_batches,1):.4f} "
              f"best recent-window loss {best_loss:.4f}", flush=True)

    # Deliver the best adapter by default at the end of training (rather than the last
    # step -- the last step may already be corrupt after divergence)
    final_dir = out_dir
    if not args.no_best_ckpt and (out_dir / "checkpoint-best" / "adapter_config.json").exists():
        import shutil
        final_dir = out_dir / "final-from-best"
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in (out_dir / "checkpoint-best").iterdir():
            if f.name != "training_state.pt" and f.name != "training_state.tmp":
                shutil.copy2(f, final_dir / f.name)
        print(f"final adapter taken from checkpoint-best (recent-window loss {best_loss:.4f})"
              f" -> {final_dir}", flush=True)
    model.save_pretrained(final_dir)
    save_processor(final_dir)
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir / "merged"))
        save_processor(str(out_dir / "merged"))
    print(f"saved -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
