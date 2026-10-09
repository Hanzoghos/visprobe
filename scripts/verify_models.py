"""Integrity gate for the four experimental models: every shard declared in
the index is present, safetensors files can be opened, and no tensors are
missing.

Exit code 0 and print ALL_OK if everything passes; otherwise exit code 1.
Uses a pure numpy backend, no torch required.
"""
import glob
import json
import os
import struct
import sys

from safetensors import safe_open


def shard_complete(sf):
    """Determine exactly whether a shard is complete per the safetensors
    container format (no torch dependency, no bf16 conversion)."""
    size = os.path.getsize(sf)
    try:
        with open(sf, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return False, "header incomplete (still downloading)"
            n = struct.unpack("<Q", head)[0]
            hdr = json.loads(f.read(n))
    except Exception:
        return False, "header incomplete (still downloading)"
    max_end = 0
    for k, v in hdr.items():
        if isinstance(v, dict) and "data_offsets" in v:
            max_end = max(max_end, v["data_offsets"][1])
    expected = 8 + n + max_end
    if size != expected:
        return False, f"truncated actual {size/1e9:.2f}GB/expected {expected/1e9:.2f}GB"
    return True, ""

MODELS = {
    "qwen3vl2b": "models--Qwen--Qwen3-VL-2B-Instruct",
    "qwen7b": "models--Qwen--Qwen2.5-VL-7B-Instruct",
    "qwen3vl8b": "models--Qwen--Qwen3-VL-8B-Instruct",
    "molmo4b": "models--allenai--Molmo2-4B",
}


def check(d):
    snaps = glob.glob(f"hf-cache/hub/{d}/snapshots/*")
    if not snaps:
        return False, "no snapshot"
    snap = snaps[0]
    idx_p = f"{snap}/model.safetensors.index.json"
    if not os.path.exists(idx_p):
        return False, "index not ready"
    declared = set(json.load(open(idx_p))["weight_map"])
    shards = sorted(glob.glob(f"{snap}/model-*.safetensors"))
    seen = set()
    for sf in shards:
        if os.path.getsize(sf) == 0:
            return False, f"{os.path.basename(sf)} is 0 bytes (download may still be in progress)"
        complete, why = shard_complete(sf)
        if not complete:
            return False, f"{os.path.basename(sf)} {why}"
        try:
            with safe_open(sf, framework="numpy") as st:
                seen.update(st.keys())
        except Exception as e:
            return False, f"{os.path.basename(sf)} header error {repr(e)[:50]}"
    missing = declared - seen
    if missing:
        return False, f"missing {len(missing)} tensors (shards not fully downloaded)"
    return True, f"{len(shards)} shards/{len(seen)} tensors"


def main():
    ok_all = True
    for name, d in MODELS.items():
        ok, msg = check(d)
        print(f"{'OK ' if ok else '...'} {name}: {msg}")
        ok_all = ok_all and ok
    if ok_all:
        print("ALL_OK")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
