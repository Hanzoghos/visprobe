"""Project path constants (mirrors nimble/paths.py)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # Raw downloads of public datasets
PROCESSED_DIR = DATA_DIR / "processed"  # Training data converted to this project's jsonl
HELDOUT_DIR = DATA_DIR / "heldout"      # Final evaluation set that never enters training
MODEL_DIR = PROJECT_ROOT / "models"     # Model snapshots / LoRA adapters / quantized artifacts


def relpath(abs_path):
    """Convert to a POSIX string relative to PROJECT_ROOT so jsonl files stay portable across machines.

    Training/evaluation commands run from the project root, so relative paths work
    directly; the jsonl files can be copied together with the project directory.
    """
    return Path(abs_path).resolve().relative_to(PROJECT_ROOT).as_posix()
