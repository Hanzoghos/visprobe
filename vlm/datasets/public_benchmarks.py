"""Public image datasets -> the project's unified jsonl format
(mirrors nimble/datasets/public_benchmarks.py).

Unified format (one record per line):
  {"id", "image" (local path), "state" (optional text-context dict),
   "questions" {field: {"type":"choice"|"boolean", "instructions", "criteria"}},
   "reference" {field: correct answer}}   # boolean answers are JSON true/false

Design revision (2026-09): only datasets with a fixed candidate set are supported
(distractors are no longer improvised from sibling answers of the same image):

  snli_ve  Multimodal-Fatima/SNLI-VE_{train,test}
           image + premise/hypothesis (placed in state) -> entailment/neutral/contradiction
           (choice, fixed 3 candidates)
  vqa_v2   yes/no subset of Multimodal-Fatima/VQAv2_{train,validation}
           image + question -> true/false (boolean)
  visual7w mm-eval/Visual7W (officially only a test split)
           the prompt itself carries four fixed candidates A-D -> choice; train/test are
           split locally by hashing source_id groups

Reads parquet shards directly from the HF cache (small pyarrow batches,
BATCH_SIZE=32 rows/batch, friendly to low-memory machines; the repo must first be fully
downloaded with scripts/prefetch_parallel.sh); images are deduplicated to disk by the
dataset's native image id (SNLI/VQA have multiple questions per image).
"""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

from ..config import DATASETS
from ..paths import RAW_DIR, PROCESSED_DIR, relpath

# Low-memory machines (~2GB RAM): decode parquet in small batches to avoid the datasets
# streaming default of reading a whole row-group, which causes swapping/OOM; the image
# field is ~1MB per row, so 32 rows per batch is safe
BATCH_SIZE = 32


def _iter_parquet_rows(repo, columns):
    """Sequentially read all parquet shards from the HF cache (snapshot) and yield dict
    rows in small batches of BATCH_SIZE.

    Requires the repo to be fully downloaded (scripts/prefetch_parallel.sh); only the
    given columns are projected.
    """
    local_dir = snapshot_download(
        repo_id=repo, repo_type="dataset", allow_patterns=["*.parquet"],
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    files = sorted(Path(local_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in the cache for {repo}; complete the download first")
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            for row in batch.to_pylist():
                yield row


def _pil(cell):
    """parquet image cell -> PIL.Image: handles an already-decoded PIL image, {'bytes','path'} dict, or raw bytes."""
    if hasattr(cell, "convert"):
        return cell.convert("RGB")
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if hasattr(cell.get("image"), "convert"):
            return cell["image"].convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise ValueError(f"cannot decode image cell to PIL: {type(cell)}")

# Fixed candidate set (constant across samples; does not vary per question)
SNLI_VE_CRITERIA = {"A": "entailment", "B": "neutral", "C": "contradiction"}
SNLI_VE_LABELS = ["entailment", "neutral", "contradiction"]
SNLI_VE_INSTRUCTIONS = (
    "Assume the image and the premise are true. "
    "How does the hypothesis relate to them?")


def dump_jsonl(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return out_path, n


def _save_unique_image(image, dataset, key, ext="jpg"):
    """Save each image only once (SNLI-VE/VQAv2 have multiple questions per image).
    The key uses the dataset's native image id.

    Returns a POSIX path relative to the project root (portable when moving the project
    to a GPU machine).
    """
    path = RAW_DIR / dataset / "images" / f"{key}.{ext}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=95)
    return relpath(path)


# ---- SNLI-VE: image + premise/hypothesis -> fixed 3-candidate choice ----

def snli_ve(split="train", out_path=None, limit=None):
    repo, _ = DATASETS["snli_ve"][split]
    rows_iter = _iter_parquet_rows(repo,
        ["image", "filename", "premise", "hypothesis", "label"])
    stats = {"scanned": 0, "kept": 0, "bad_label": 0}

    def gen():
        for ex in rows_iter:
            stats["scanned"] += 1
            label = int(ex["label"])  # 0=entailment 1=neutral 2=contradiction
            if not 0 <= label < len(SNLI_VE_LABELS):
                stats["bad_label"] += 1
                continue  # SNLI occasionally has dirty rows with no gold label
            if limit and stats["kept"] >= limit:
                break
            premise = str(ex["premise"]).strip()
            hypothesis = str(ex["hypothesis"]).strip()
            img_path = _save_unique_image(_pil(ex["image"]),
                                          "snli_ve", Path(ex["filename"]).stem)
            stats["kept"] += 1
            yield {
                "id": f"snli_ve-{split}-{stats['scanned']-1:07d}",
                "image": img_path,
                "state": {"premise": premise, "hypothesis": hypothesis},
                "questions": {"relation": {
                    "type": "choice",
                    "instructions": SNLI_VE_INSTRUCTIONS,
                    "criteria": SNLI_VE_CRITERIA,
                }},
                "reference": {"relation": SNLI_VE_LABELS[label]},
            }

    out = out_path or (PROCESSED_DIR / "snli_ve" / f"{split}.jsonl")
    out, n = dump_jsonl(gen(), out)
    print(f"  [stats] scanned {stats['scanned']} / kept {n} / skipped (bad_label) {stats['bad_label']}"
          + (f" (truncated early at limit={limit}; remaining rows not scanned)" if limit and stats["kept"] >= limit else ""))
    return out, n


# ---- VQAv2 yes/no subset -> boolean (reference is JSON true/false) ----

def vqa_v2_yesno(split="train", out_path=None, limit=None):
    repo, _ = DATASETS["vqa_v2"][split]
    # Project only the required columns (this repo carries dozens of heavy annotation
    # columns per row; projecting only what is needed is essential or low-memory machines OOM)
    rows_iter = _iter_parquet_rows(repo,
        ["question", "multiple_choice_answer", "answer_type",
         "id_image", "question_id", "image"])
    stats = {"scanned": 0, "kept": 0, "skipped_not_yn": 0}

    def gen():
        for ex in rows_iter:
            stats["scanned"] += 1
            if ex.get("answer_type") != "yes/no":
                stats["skipped_not_yn"] += 1
                continue
            ans = str(ex.get("multiple_choice_answer", "")).strip().lower()
            if ans not in ("yes", "no"):
                stats["skipped_not_yn"] += 1
                continue
            if limit and stats["kept"] >= limit:
                break
            img_path = _save_unique_image(_pil(ex["image"]),
                                          "vqa_v2", str(ex["id_image"]))
            stats["kept"] += 1
            yield {
                "id": f"vqa_v2-{split}-{ex['question_id']}",
                "image": img_path,
                "state": {},
                "questions": {"answer": {
                    "type": "boolean",
                    "instructions": str(ex["question"]).strip(),
                }},
                "reference": {"answer": ans == "yes"},  # JSON true / false
            }

    out = out_path or (PROCESSED_DIR / "vqa_v2" / f"{split}.jsonl")
    out, n = dump_jsonl(gen(), out)
    print(f"  [stats] scanned {stats['scanned']} / kept {n} / skipped (not yes/no) {stats['skipped_not_yn']}"
          + (f" (truncated early at limit={limit})" if limit and stats["kept"] >= limit else ""))
    return out, n


# ---- Visual7W: the prompt carries fixed A-D candidates -> choice; train/test split by image hash ----

_V7W_Q = re.compile(r"Question:\s*(.*?)\s*Choices:\s*(.*)", re.S)
_V7W_C = re.compile(r"(?m)^\s*([A-D])[.、)]\s*(.+?)\s*\.?\s*$")


def _parse_visual7w_question(messages):
    """messages is a JSON string: [{'role':'user','question':'Question: ... Choices: A...'}]."""
    data = json.loads(messages)
    text = data[0]["question"]
    m = _V7W_Q.search(text)
    if not m:
        return None
    pairs = dict(_V7W_C.findall(m.group(2)))
    if len(pairs) != 4 or any(k not in pairs for k in "ABCD"):
        return None
    return m.group(1).strip(), {k: pairs[k].strip() for k in "ABCD"}


def _visual7w_group_split(source_id, test_ratio=0.1):
    """A given source_id (image) always lands in the same train or test split, preventing image-level leakage."""
    h = int(hashlib.md5(source_id.encode("utf-8")).hexdigest(), 16)
    return "test" if h % 100 < round(test_ratio * 100) else "train"


def visual7w(split="train", out_path=None, limit=None):
    repo, _ = DATASETS["visual7w"][split]
    rows_iter = _iter_parquet_rows(repo,
        ["id", "media", "messages", "answer", "question_type", "source_id"])
    stats = {"scanned": 0, "kept": 0, "other_split": 0, "bad_mcq": 0}

    def gen():
        for ex in rows_iter:
            stats["scanned"] += 1
            if ex.get("question_type") != "mcq":
                stats["bad_mcq"] += 1
                continue
            parsed = _parse_visual7w_question(ex["messages"])
            if parsed is None or ex.get("answer") not in "ABCD":
                stats["bad_mcq"] += 1
                continue  # failed prompt parse / abnormal answer code: count then skip; never drop silently
            source_id = ex["source_id"]
            if _visual7w_group_split(source_id) != split:
                stats["other_split"] += 1
                continue  # this image belongs to the other local split
            if limit and stats["kept"] >= limit:
                break
            question, criteria = parsed
            media_item = ex["media"][0] if isinstance(ex["media"], list) else ex["media"]
            img_path = _save_unique_image(_pil(media_item),
                                          "visual7w", source_id)
            stats["kept"] += 1
            yield {
                "id": f"visual7w-{source_id}-{ex['id']}",
                "image": img_path,
                "state": {},
                "questions": {"answer": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": criteria,
                }},
                "reference": {"answer": criteria[ex["answer"]]},
            }

    out = out_path or (PROCESSED_DIR / "visual7w" / f"{split}.jsonl")
    out, n = dump_jsonl(gen(), out)
    print(f"  [stats] scanned {stats['scanned']} / kept in this split {n} / in other split {stats['other_split']}"
          f" / skipped (not mcq or parse failure) {stats['bad_mcq']}"
          + (f" (truncated early at limit={limit})" if limit and stats["kept"] >= limit else ""))
    return out, n


CONVERTERS = {"snli_ve": snli_ve, "vqa_v2": vqa_v2_yesno, "visual7w": visual7w}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(CONVERTERS))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="convert only the first N records (smoke test; counted in post-filter order)")
    args = ap.parse_args()
    out, n = CONVERTERS[args.dataset](args.split, args.out, args.limit)
    print(f"{args.dataset}/{args.split}: {n} records -> {out}", flush=True)
    # Some datasets/huggingface_hub background threads are non-daemon and can hang the
    # process at exit on low-memory machines; files were already flushed and closed line
    # by line, so flush stdout/stderr before forcing exit (0 = success).
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
