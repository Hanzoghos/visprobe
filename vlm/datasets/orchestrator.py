"""Closed-loop orchestrator: convert -> QC -> [baseline evaluation] -> train ->
[after (LoRA) evaluation] -> error mining -> API distillation.

Core order (revised 2026-09): for each model, first evaluate the "original model"
(baseline) with the probability pipeline, then LoRA-finetune, then evaluate "merged"
(after) -- whether training helped is judged by after.acc vs baseline.acc.

convert/qc iterate over all three datasets; baseline/train/eval/mine/distill run only on
the dataset given by --dataset, once for each model listed in --models. Default is
--dry-run, which only prints commands; pass --run to actually execute.

Usage:
  python -m vlm.datasets.orchestrator                                  # show all commands
  python -m vlm.datasets.orchestrator --stages baseline train eval --run
  python -m vlm.datasets.orchestrator --dataset vqa_v2 \
      --models qwen3vl2b qwen3vl8b molmo4b qwen7b --run
"""

STAGES = ["convert", "qc", "baseline", "train", "eval", "mine", "distill"]
ALL_DATASETS = ["snli_ve", "vqa_v2", "visual7w"]
# The four experimental models by default; large models may need --4bit on small GPUs
ALL_MODELS = ["qwen3vl2b", "qwen3vl8b", "molmo4b", "qwen7b"]


def build_commands(args):
    ds = args.ds
    cmds = []
    if "convert" in args.stages:
        for name in ALL_DATASETS:
            for split in ("train", "test"):
                cmds.append(["python", "-m", "vlm.datasets.public_benchmarks",
                             "--dataset", name, "--split", split])
    if "qc" in args.stages:
        for name in ALL_DATASETS:
            cmds.append(["python", "-m", "vlm.datasets.qc",
                         "--data", f"data/processed/{name}/train.jsonl",
                         "--overlap-with", f"data/processed/{name}/test.jsonl"])

    # 1. Baseline evaluation: run every model's original weights on the held-out set first (before finetuning)
    if "baseline" in args.stages:
        for m in args.models:
            cmds.append(["python", "-m", "vlm.evaluation.evaluate",
                         "--data", f"data/processed/{ds}/test.jsonl",
                         "--model-id", m,
                         "--save", f"data/heldout/{ds}.baseline-{m}.json"])

    # 2. Training: LoRA-finetune each model separately
    if "train" in args.stages:
        for m in args.models:
            cmd = ["python", "-m", "vlm.training.lora_train",
                   "--model-id", m,
                   "--data", f"data/processed/{ds}/train.jsonl",
                   "--out", f"models/{ds}-{m}-lora",
                   "--epochs", str(args.epochs), "--rank", str(args.rank),
                   "--merge"]
            if args.bit4:
                cmd.append("--4bit")
            cmds.append(cmd)

    # 3. After (LoRA) evaluation: run the merged model on the same held-out set again
    if "eval" in args.stages:
        for m in args.models:
            cmds.append(["python", "-m", "vlm.evaluation.evaluate",
                         "--data", f"data/processed/{ds}/test.jsonl",
                         "--model-id", f"models/{ds}-{m}-lora/merged",
                         "--save", f"data/heldout/{ds}.after-{m}.json"])
        # Cross-comparison: original vs finetuned (see the acc gain on the same row)
        cmds.append(["python", "-m", "vlm.evaluation.compare",
                     "--data", f"data/processed/{ds}/test.jsonl",
                     "--models", *[m for pair in
                                   [(m, f"models/{ds}-{m}-lora/merged")
                                    for m in args.models] for m in pair]])

    if "mine" in args.stages:
        for m in args.models:
            cmds.append(["python", "-m", "vlm.evaluation.evaluate",
                         "--data", f"data/processed/{ds}/test.jsonl",
                         "--model-id", f"models/{ds}-{m}-lora/merged",
                         "--errors-out", f"data/heldout/{ds}-{m}-errors.jsonl"])

    if "distill" in args.stages:
        # The teacher uses an external API (configuration from VL_TEACHER_API_* environment variables)
        cmds.append(["python", "-m", "vlm.datasets.distill",
                     "--data", f"data/processed/{ds}/train.jsonl",
                     "--out", f"data/processed/{ds}/train.distilled.jsonl"])
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    ap.add_argument("--dataset", dest="ds", default="snli_ve", choices=ALL_DATASETS,
                    help="dataset targeted by baseline/train/eval/mine/distill")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    help="short names from config.MODELS (default: the four experimental models)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--4bit", dest="bit4", action="store_true",
                    help="required for 7/8/9B models on small GPUs (NF4)")
    ap.add_argument("--run", action="store_true", help="actually execute (default is dry-run, print only)")
    args = ap.parse_args()

    cmds = build_commands(args)
    for c in cmds:
        print(" ".join(c))
    if args.run:
        for c in cmds:
            print(f"\n==> {' '.join(c)}")
            subprocess.run(c, check=True)
    else:
        print("\n(dry-run: pass --run to actually execute)")


if __name__ == "__main__":
    main()
