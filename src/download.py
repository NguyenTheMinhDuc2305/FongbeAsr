"""Download competition audio from the organizers' public GCS bucket.

Streams each audio_*.tar straight from HTTP into a tar reader (no 2 GB temporary archive),
extracts only the WAVs referenced by the configured CSVs, and verifies the server-side MD5
(x-goog-hash) of the full byte stream. Files from an archive that fails verification are deleted
and the archive is retried. Object generations are recorded for provenance.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from src.common import atomic_json, audio_root, load_config, project_path, read_csv

CHUNK = 4 * 2**20


class HashingReader:
    """File-like wrapper: every byte read is counted and hashed (MD5) as it streams."""

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
    """Sequential (stream-mode) extraction of requested regular files; returns count."""
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


def extract_needed(archive, root, needed):
    """Extract requested files from a local archive (also used by tests)."""
    with tarfile.open(archive, "r:*") as tar:
        return extract_stream(tar, Path(root), needed, [])


def list_bucket(base_url):
    """Public GCS XML listing (paginated)."""
    objects, marker = [], None
    while True:
        url = base_url + (f"?marker={urllib.parse.quote(marker)}" if marker else "")
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        ns = {"s": root.tag.split("}")[0].strip("{")}
        page = [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in root.findall("s:Contents", ns)]
        objects += page
        truncated = root.find("s:IsTruncated", ns)
        if truncated is None or truncated.text != "true" or not page:
            return objects
        marker = page[-1][0]


def object_metadata(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        hashes = dict(h.split("=", 1) for v in response.headers.get_all("x-goog-hash") or [] for h in v.split(","))
        if "md5" not in hashes:
            raise ValueError(f"No server MD5 for {url}; refusing unverifiable download")
        return {"size": int(response.headers["Content-Length"]), "md5": hashes["md5"].strip(),
                "generation": response.headers.get("x-goog-generation")}


def fetch_archive(base_url, name, root, needed, retries=4):
    url = base_url + name
    meta = object_metadata(url)
    for attempt in range(1, retries + 1):
        written, start = [], time.monotonic()
        try:
            # Pin the exact object generation so a mid-download replacement cannot mix versions.
            pinned = url + (f"?generation={meta['generation']}" if meta["generation"] else "")
            with urllib.request.urlopen(pinned, timeout=300) as response:
                reader = HashingReader(response)
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    extracted = extract_stream(tar, root, needed, written)
                while reader.read(CHUNK):  # hash trailing padding after the tar end marker
                    pass
            digest = base64.b64encode(reader.md5.digest()).decode()
            if reader.bytes != meta["size"] or digest != meta["md5"]:
                raise IOError(f"Integrity check failed for {name}: {reader.bytes}/{meta['size']} bytes, "
                              f"md5 {digest} != {meta['md5']}")
            seconds = time.monotonic() - start
            return {"name": name, **meta, "extracted": extracted, "seconds": round(seconds, 1),
                    "mb_per_s": round(meta["size"] / 1e6 / max(seconds, 1e-9), 1), "attempts": attempt}
        except (OSError, urllib.error.URLError, tarfile.TarError, ValueError) as exc:
            for path in written:  # never keep files from an unverified stream
                path.unlink(missing_ok=True)
            if isinstance(exc, ValueError) or attempt == retries:  # content errors are not transient
                raise
            print(f"Retry {attempt}/{retries} for {name}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(5 * attempt)


def download(config, include_test=False):
    cfg = load_config(config)
    source = cfg["download"]
    if source.get("source") != "gcs_http":
        raise ValueError("download.source must be gcs_http (organizers' public bucket)")
    base_url = source["base_url"].rstrip("/") + "/"
    root = audio_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("train_csv", "valid_general_csv", "valid_religious_csv"):
        rows += read_csv(project_path(cfg["data"][key]))
    if include_test or source.get("include_test"):
        rows += read_csv(project_path(cfg["data"]["test_csv"]))
    needed = {r["audio_path"] for r in rows}
    missing = {p for p in needed if not (root / p).is_file()}
    if not missing:
        print(f"All {len(needed)} audio files already exist in {root}; training preflight checks headers.")
        return
    estimate = sum(float(r.get("duration") or 10) * 32000 + 44 for r in rows if r["audio_path"] in missing)
    free = shutil.disk_usage(root).free
    print(f"Need ~{estimate/1e9:.1f} GB for {len(missing)} WAVs; free {free/1e9:.1f} GB at {root}", flush=True)
    if free < estimate + 2e9:
        raise RuntimeError(f"Insufficient disk: need ~{(estimate+2e9)/1e9:.1f} GB, have {free/1e9:.1f} GB. "
                           "Point FONGBE_AUDIO_ROOT at a larger volume; never train a silent subset.")
    groups = {PurePosixPath(p).parts[1] for p in missing}  # e.g. train_reliable, test
    archives = sorted(name for name, _ in list_bucket(base_url)
                      if name.endswith(".tar") and any(name.startswith(f"audio_{g}_") for g in groups))
    if not archives:
        raise RuntimeError(f"No audio_*.tar for {sorted(groups)} at {base_url}")
    audit = {"source": base_url, "archives": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    lock, start = threading.Lock(), time.monotonic()
    with ThreadPoolExecutor(max_workers=int(source.get("parallel", 3))) as pool:
        futures = {pool.submit(fetch_archive, base_url, name, root, missing): name for name in archives}
        for future in as_completed(futures):
            result = future.result()  # re-raises: one bad archive aborts the whole download
            with lock:
                audit["archives"].append(result)
                atomic_json(root / "download_manifest.json", audit)
            print(f"OK {result['name']}: {result['extracted']} files, {result['mb_per_s']} MB/s, "
                  f"generation {result['generation']}", flush=True)
    still = sorted(p for p in needed if not (root / p).is_file())
    if still:
        raise FileNotFoundError(f"{len(still)} referenced audio files absent from bucket archives, first: {still[:5]}")
    total = sum(a["size"] for a in audit["archives"])
    audit.update(seconds=round(time.monotonic() - start, 1), total_gb=round(total / 1e9, 2))
    atomic_json(root / "download_manifest.json", audit)
    print(f"Ready: {len(needed)} files in {root}; {total/1e9:.1f} GB in {audit['seconds']} s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    p.add_argument("--include-test", action="store_true")
    a = p.parse_args()
    download(a.config, a.include_test)
