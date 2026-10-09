"""Direct resumable download of missing model shards into the HF snapshot
directory (bypassing the hf parallel chunker that tends to hang).

Usage: python scripts/direct_shard_dl.py <repo> <file> [<file> ...]
The mirror is taken from HF_ENDPOINT; each file runs in its own thread, with
Range resume, retry on failure, and progress written to the log.
Completed files are real files inside the snapshot (coexisting with symlinks;
transformers loads by name, so there is no impact).
"""
import os
import sys
import time
import glob
import threading
import urllib.request


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range")  # bytes 0-0/TOTAL
        if cr and "/" in cr:
            return int(cr.split("/")[-1])
        return int(r.headers.get("Content-Length", 0))


def download_one(repo, fname, snap, log):
    ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{ep}/{repo}/resolve/main/{fname}"
    target = os.path.join(snap, fname)

    def emit(msg):
        line = f"[{time.strftime('%F %T')}] {fname}: {msg}"
        with open(log, "a") as f:
            f.write(line + "\n")

    total = 0
    for attempt in range(100):
        try:
            total = remote_size(url)
            break
        except Exception as e:
            emit(f"failed to get size {repr(e)[:80]}, retry in 10s")
            time.sleep(10)
    if not total:
        emit("cannot determine target size, giving up on this shard")
        return False

    emit(f"target size {total/1e9:.2f}GB")
    while True:
        have = os.path.getsize(target) if os.path.exists(target) else 0
        if total and have >= total:
            emit(f"DONE size check passed ({have/1e9:.2f}GB)")
            return True
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            last = have
            with urllib.request.urlopen(req, timeout=60) as r, open(target, "ab") as out:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    if time.time() - t0 > 20:
                        spd = (have - last) / (time.time() - t0) / 1e6
                        pct = have / total * 100 if total else 0
                        emit(f"{pct:5.1f}% {have/1e9:.2f}/{total/1e9:.2f}GB {spd:.2f}MB/s")
                        t0, last = time.time(), have
        except Exception as e:
            emit(f"connection error {repr(e)[:90]}, resume in 5s")
            time.sleep(5)


def main():
    repo = sys.argv[1]
    files = sys.argv[2:]
    snap = glob.glob(f"hf-cache/hub/models--{repo.replace('/', '--')}/snapshots/*/")[0]
    log = "logs/direct_shard_dl.log"
    threads = []
    for fname in files:
        th = threading.Thread(target=download_one, args=(repo, fname, snap, log), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    with open(log, "a") as f:
        f.write(f"[{time.strftime('%F %T')}] ALL_SHARDS_DONE {files}\n")


if __name__ == "__main__":
    main()
