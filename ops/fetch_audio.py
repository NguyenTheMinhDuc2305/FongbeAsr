"""Standalone Kaggle-side audio fetcher for presigned MinIO URLs (stdlib + PyYAML only).

Deliberately independent of src/: a resumed session checks out the run's pinned commit (whose
downloader only knows the original GCS source), while the notebook fetches this file from main.
Data identity is still enforced by the training code: split CSV hashes + preflight effective-ID hash.

    python fetch_audio.py --manifest audio_manifest.json --project /kaggle/temp/FongbeAsr \
        --root /kaggle/temp/fongbe [--include-test] [--parallel 3]

The manifest maps archive -> {url, size, md5(base64)}; URLs are short-lived read-only presigns.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request

CHUNK = 4 * 2**20


class HashingReader:
    def __init__(self, stream):
        self.stream, self.md5, self.bytes = stream, hashlib.md5(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.md5.update(data)
        self.bytes += len(data)
        return data


def safe_member(name, root):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    dest = root.joinpath(*rel.parts)
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes audio root: {name}")
    return rel.as_posix(), dest


def extract_stream(tar, root, needed, written):
    count = 0
    for member in tar:
        name, dest = safe_member(member.name, root)
        if name not in needed:
            continue
        if not member.isfile():
            raise ValueError(f"Expected regular WAV file: {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        with tar.extractfile(member) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, CHUNK)
        tmp.replace(dest)
        written.append(dest)
        count += 1
    return count


def fetch(name, entry, root, needed, retries=4):
    for attempt in range(1, retries + 1):
        written, start = [], time.monotonic()
        try:
            with urllib.request.urlopen(entry["url"], timeout=300) as response:
                reader = HashingReader(response)
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    extracted = extract_stream(tar, root, needed, written)
                while reader.read(CHUNK):
                    pass
            digest = base64.b64encode(reader.md5.digest()).decode()
            if reader.bytes != entry["size"] or digest != entry["md5"]:
                raise IOError(f"Integrity check failed for {name}: {reader.bytes}/{entry['size']} B, "
                              f"md5 {digest} != {entry['md5']}")
            seconds = time.monotonic() - start
            return {"name": name, "size": entry["size"], "md5": entry["md5"], "extracted": extracted,
                    "seconds": round(seconds, 1), "mb_per_s": round(entry["size"] / 1e6 / max(seconds, 1e-9), 1),
                    "attempts": attempt}
        except (OSError, urllib.error.URLError, tarfile.TarError, ValueError) as exc:
            for path in written:
                path.unlink(missing_ok=True)
            if isinstance(exc, ValueError) or attempt == retries:
                raise
            print(f"Retry {attempt}/{retries} for {name}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * attempt)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--parallel", type=int, default=3)
    a = p.parse_args()
    project, root = Path(a.project), Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((project / a.config).read_text(encoding="utf-8"))
    rows = []
    for key in ("train_csv", "valid_general_csv", "valid_religious_csv"):
        rows += read_rows(project / cfg["data"][key])
    if a.include_test:
        rows += read_rows(project / cfg["data"]["test_csv"])
    needed = {r["audio_path"] for r in rows}
    missing = {p for p in needed if not (root / p).is_file()}
    if not missing:
        print(f"All {len(needed)} audio files already present in {root}")
        return
    estimate = sum(float(r.get("duration") or 10) * 32000 + 44 for r in rows if r["audio_path"] in missing)
    free = shutil.disk_usage(root).free
    print(f"Need ~{estimate/1e9:.1f} GB for {len(missing)} WAVs; free {free/1e9:.1f} GB", flush=True)
    if free < estimate + 2e9:
        raise RuntimeError("Insufficient disk; never train a silent subset")
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    groups = {PurePosixPath(p).parts[1] for p in missing}
    archives = {n: e for n, e in manifest.items() if any(n.startswith(f"audio_{g}_") for g in groups)}
    if not archives:
        raise RuntimeError(f"Manifest has no archives for {sorted(groups)}")
    audit = {"source": "minio-presigned", "archives": [],
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lock, start = threading.Lock(), time.monotonic()
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        futures = {pool.submit(fetch, n, e, root, missing): n for n, e in sorted(archives.items())}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                audit["archives"].append(result)
                (root / "download_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            print(f"OK {result['name']}: {result['extracted']} files, {result['mb_per_s']} MB/s, md5 verified",
                  flush=True)
    still = sorted(p for p in needed if not (root / p).is_file())
    if still:
        raise FileNotFoundError(f"{len(still)} referenced files absent from archives, first: {still[:5]}")
    total = sum(x["size"] for x in audit["archives"])
    audit.update(seconds=round(time.monotonic() - start, 1), total_gb=round(total / 1e9, 2))
    (root / "download_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Ready: {len(needed)} files in {root}; {total/1e9:.1f} GB in {audit['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
