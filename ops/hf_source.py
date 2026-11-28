"""Hugging Face (gated) competition dataset -> short-lived signed CDN URLs for Kaggle.

The dataset JaimeMaLangue/JML-Hackathon-fon-asr-data is gated ("auto" approval): downloads need a
token of an account that accepted its terms. The token stays LOCAL (.env HF_TOKEN). Kaggle only gets
per-file CDN URLs that HF signs on redirect (validity ~1 h), resolved at submission time.

    python -m ops.hf_source check      # access, revision pin, archive list, SHA256SUMS cross-check
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import dotenv_values
from src.common import ROOT

REPO = "JaimeMaLangue/JML-Hackathon-fon-asr-data"
MIN_VALIDITY_S = 50 * 60  # downloads finish ~25 min after submission; refuse links valid < 50 min


def token():
    value = os.environ.get("HF_TOKEN") or {k.strip(): (v or "").strip()
                                           for k, v in dotenv_values(ROOT / ".env").items()}.get("HF_TOKEN")
    if not value:
        raise RuntimeError("HF_TOKEN missing: add a READ token to .env from an account that accepted the "
                           f"dataset terms at https://huggingface.co/datasets/{REPO}")
    return value


def _request(url, method="GET", auth=True):
    headers = {"Authorization": f"Bearer {token()}"} if auth else {}
    return urllib.request.Request(url, method=method, headers=headers)


def _open(url, method="GET"):
    try:
        return urllib.request.urlopen(_request(url, method), timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(
                f"HF denied access (HTTP {exc.code}). Accept the terms at https://huggingface.co/datasets/{REPO} "
                "with the account that owns HF_TOKEN, and use a token with read access.") from exc
        raise


def repo_info(revision="main"):
    """Pinned commit sha + {file: {size, sha256}} (LFS sha256 for archives)."""
    with _open(f"https://huggingface.co/api/datasets/{REPO}/revision/{revision}?blobs=true") as response:
        info = json.load(response)
    files = {s["rfilename"]: {"size": s.get("size"), "sha256": (s.get("lfs") or {}).get("sha256")}
             for s in info.get("siblings", [])}
    return info["sha"], files


def sha256sums(revision):
    with _open(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/SHA256SUMS") as response:
        text = response.read().decode()
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*").removeprefix("./")] = digest.lower()
    return sums


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def signed_url(revision, filename, expected_size):
    """Resolve -> 302 Location (signed CDN URL). Returns (url, expires_epoch)."""
    opener = urllib.request.build_opener(_NoRedirect)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{urllib.parse.quote(filename)}"
    try:
        opener.open(_request(url, "HEAD"), timeout=60)
        raise RuntimeError(f"Expected a CDN redirect for {filename}")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise PermissionError(f"HF denied {filename} (HTTP {exc.code}); accept the dataset terms") from exc
        if exc.code not in (302, 307):
            raise
        location = exc.headers["Location"]
        linked = exc.headers.get("X-Linked-Size")
        if linked is not None and int(linked) != expected_size:
            raise ValueError(f"{filename}: CDN size {linked} != repo size {expected_size}")
        query = urllib.parse.parse_qs(urllib.parse.urlparse(location).query)
        expires = int((query.get("Expires") or query.get("X-Amz-Expires-At") or [0])[0])
        if not expires and "X-Amz-Expires" in query and "X-Amz-Date" in query:
            start = time.mktime(time.strptime(query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ")) - time.timezone
            expires = int(start + int(query["X-Amz-Expires"][0]))
        return location, expires


def archive_manifest(groups, revision="main"):
    """{archive: {url, size, sha256, expires}} for audio_<group>_*.tar, pinned to one commit."""
    sha, files = repo_info(revision)
    sums = sha256sums(sha)
    chosen = sorted(n for n in files if n.endswith(".tar") and any(n.startswith(f"audio_{g}_") for g in groups))
    for group in groups:
        if not any(n.startswith(f"audio_{group}_") for n in chosen):
            raise RuntimeError(f"No archives for group '{group}' in {REPO}@{sha}")
    manifest, now = {}, time.time()
    for name in chosen:
        meta = files[name]
        if meta["sha256"] and sums.get(name) and meta["sha256"] != sums[name]:
            raise ValueError(f"{name}: LFS sha256 differs from SHA256SUMS")
        digest = sums.get(name) or meta["sha256"]
        if not digest:
            raise ValueError(f"{name}: no SHA256 available; refusing unverifiable download")
        url, expires = signed_url(sha, name, meta["size"])
        if expires and expires - now < MIN_VALIDITY_S:
            raise RuntimeError(f"{name}: signed URL valid only {int(expires - now)} s")
        manifest[name] = {"url": url, "size": meta["size"], "sha256": digest, "expires": expires}
    return sha, manifest


def check():
    sha, files = repo_info()
    sums = sha256sums(sha)
    tars = sorted(n for n in files if n.startswith("audio_") and n.endswith(".tar"))
    mismatch = [n for n in tars if files[n]["sha256"] and sums.get(n) and files[n]["sha256"] != sums[n]]
    print(f"{REPO}@{sha}: {len(tars)} archives, {sum(files[n]['size'] for n in tars)/1e9:.2f} GB; "
          f"SHA256SUMS entries {len(sums)}; LFS/SHA256SUMS mismatches {mismatch}")
    sha_, manifest = archive_manifest(["train_religious"], sha)
    first = next(iter(manifest.values()))
    print(f"signed CDN URL ok: host={urllib.parse.urlparse(first['url']).netloc}, "
          f"valid {int((first['expires'] - time.time())/60)} min")
    return sha


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["check"])
    p.parse_args()
    check()
