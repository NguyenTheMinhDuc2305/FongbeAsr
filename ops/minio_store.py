"""Mirror competition audio archives into the private MinIO bucket and issue presigned URLs.

Credentials stay LOCAL (.env: MINIO_ENDPOINT_URL, AccessKey, Secretkey, BUCKET). Kaggle only ever
receives short-lived, read-only presigned GET URLs for individual archives.

    python -m ops.minio_store status
    python -m ops.minio_store mirror --source https://storage.googleapis.com/fongbe/
    python -m ops.minio_store wait-and-mirror --source https://storage.googleapis.com/fongbe/ --interval 1800

The bucket must stay private: the data is "confidential, for use in this challenge only".
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

if not __package__:  # allow execution as a script (Task Scheduler)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import dotenv_values
from src.common import ROOT

PREFIX = "competition/v1/"  # object layout: <PREFIX>audio_*.tar, <PREFIX>manifest.json
MANIFEST = PREFIX + "manifest.json"
PART = 64 * 2**20


def settings():
    cfg = {k.strip(): (v or "").strip() for k, v in dotenv_values(ROOT / ".env").items()}
    missing = [k for k in ("MINIO_ENDPOINT_URL", "AccessKey", "Secretkey", "BUCKET") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f".env is missing {missing}")
    # S3 bucket names must be lowercase; the configured 'FongbeAsr' maps to 'fongbeasr'.
    return cfg["MINIO_ENDPOINT_URL"], cfg["AccessKey"], cfg["Secretkey"], cfg["BUCKET"].lower()


def client():
    import boto3
    from botocore.config import Config
    endpoint, key, secret, _ = settings()
    return boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key, aws_secret_access_key=secret,
                        region_name="us-east-1",
                        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                                      connect_timeout=20, read_timeout=300))


def bucket():
    return settings()[3]


class HashingReader(io.RawIOBase):
    """Read-only stream wrapper computing MD5 + byte count as boto3 consumes it."""

    def __init__(self, stream):
        self.stream, self.md5, self.bytes = stream, hashlib.md5(), 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self.stream.read(size if size and size > 0 else PART)
        self.md5.update(data)
        self.bytes += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class GcsHttpSource:
    """The organizers' public GCS bucket (XML API). Raises PermissionError on 401/403."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/") + "/"

    def archives(self):
        from src.download import list_bucket
        try:
            return sorted(n for n, _ in list_bucket(self.base) if n.startswith("audio_") and n.endswith(".tar"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError(f"Source {self.base} denies anonymous access (HTTP {exc.code})") from exc
            raise

    def metadata(self, name):
        from src.download import object_metadata
        return object_metadata(self.base + name)  # size, md5 (base64), generation

    def open(self, name, meta):
        pinned = self.base + name + (f"?generation={meta['generation']}" if meta.get("generation") else "")
        return urllib.request.urlopen(pinned, timeout=300)


class LocalDirSource:
    """Archives already on disk (e.g. a manual download); also used by the self-test."""

    def __init__(self, directory):
        self.dir = Path(directory)

    def archives(self):
        return sorted(p.name for p in self.dir.glob("audio_*.tar"))

    def metadata(self, name):
        md5 = hashlib.md5()
        with (self.dir / name).open("rb") as f:
            for block in iter(lambda: f.read(PART), b""):
                md5.update(block)
        return {"size": (self.dir / name).stat().st_size, "md5": base64.b64encode(md5.digest()).decode(),
                "generation": None}

    def open(self, name, meta):
        return (self.dir / name).open("rb")


def read_manifest(s3, prefix=PREFIX):
    try:
        return json.loads(s3.get_object(Bucket=bucket(), Key=prefix + "manifest.json")["Body"].read())
    except s3.exceptions.NoSuchKey:
        return {"archives": {}}


def mirror(source, prefix=PREFIX, only=None):
    s3, name_bucket = client(), bucket()
    manifest = read_manifest(s3, prefix)
    names = source.archives() if only is None else only
    if only is None:  # the full source listing defines completeness for pre-submission checks
        manifest["expected_archives"] = names
    print(f"{len(names)} source archives -> s3://{name_bucket}/{prefix}", flush=True)
    for name in names:
        meta = source.metadata(name)
        have = manifest["archives"].get(name)
        if have and have["md5"] == meta["md5"] and have["size"] == meta["size"]:
            try:
                head = s3.head_object(Bucket=name_bucket, Key=prefix + name)
                if head["ContentLength"] == meta["size"]:
                    print(f"skip {name} (already mirrored, md5 match)", flush=True)
                    continue
            except s3.exceptions.ClientError:
                pass
        start = time.monotonic()
        with source.open(name, meta) as stream:
            reader = HashingReader(stream)
            from boto3.s3.transfer import TransferConfig
            s3.upload_fileobj(reader, name_bucket, prefix + name,
                              ExtraArgs={"Metadata": {"source-md5": meta["md5"],
                                                      "source-generation": str(meta.get("generation"))}},
                              Config=TransferConfig(multipart_chunksize=PART, max_concurrency=4))
        digest = base64.b64encode(reader.md5.digest()).decode()
        if reader.bytes != meta["size"] or digest != meta["md5"]:
            s3.delete_object(Bucket=name_bucket, Key=prefix + name)
            raise IOError(f"{name}: integrity mismatch ({reader.bytes}/{meta['size']} B, md5 {digest} != {meta['md5']}); deleted")
        seconds = time.monotonic() - start
        manifest["archives"][name] = {**meta, "mirrored_utc": datetime.now(timezone.utc).isoformat(),
                                      "seconds": round(seconds, 1)}
        manifest["source"] = getattr(source, "base", str(getattr(source, "dir", "")))
        s3.put_object(Bucket=name_bucket, Key=prefix + "manifest.json",
                      Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json")
        print(f"OK {name}: {meta['size']/1e9:.2f} GB, {meta['size']/1e6/max(seconds,1e-9):.1f} MB/s, md5 verified",
              flush=True)
    s3.put_object(Bucket=name_bucket, Key=prefix + "manifest.json",
                  Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json")
    return manifest


def presigned_archives(groups, expires=24 * 3600, prefix=PREFIX):
    """{archive: {url, size, md5}} for archives whose name matches audio_<group>_*.tar."""
    s3 = client()
    manifest = read_manifest(s3, prefix)["archives"]
    chosen = {n: m for n, m in manifest.items() if any(n.startswith(f"audio_{g}_") for g in groups)}
    for group in groups:
        if not any(n.startswith(f"audio_{group}_") for n in chosen):
            raise RuntimeError(f"No mirrored archives for group '{group}' in s3://{bucket()}/{prefix}")
    return {n: {"url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket(), "Key": prefix + n},
                                                 ExpiresIn=expires),
                "size": m["size"], "md5": m["md5"]} for n, m in sorted(chosen.items())}


def status(prefix=PREFIX):
    s3 = client()
    manifest = read_manifest(s3, prefix)["archives"]
    total = sum(m["size"] for m in manifest.values())
    print(f"s3://{bucket()}/{prefix}: {len(manifest)} archives mirrored, {total/1e9:.2f} GB")
    for name, meta in sorted(manifest.items()):
        print(f"  {name}  {meta['size']/1e9:.3f} GB  md5={meta['md5']}")
    return manifest


def wait_and_mirror(source_url, interval, max_hours, train_slug=None, resume_from=None, chain_sessions=2):
    """Poll the organizers' source; when readable, mirror everything, then optionally resume training
    from the private mirror and keep watching it (all from this one local process)."""
    from ops.kaggle_run import gmt7, submit, watch
    deadline = time.monotonic() + max_hours * 3600
    source = GcsHttpSource(source_url)
    while time.monotonic() < deadline:
        try:
            source.archives()
        except PermissionError as exc:
            print(f"{gmt7()} source still closed: {exc}", flush=True)
            time.sleep(interval)
            continue
        except OSError as exc:  # transient network issue
            print(f"{gmt7()} source check failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(interval)
            continue
        mirror(source)
        print(f"{gmt7()} Mirror complete.", flush=True)
        if train_slug:
            state = submit("train", train_slug, resume_from=resume_from, audio_source="minio")
            print(f"{gmt7()} submitted {train_slug} (resume from {resume_from}) using the MinIO mirror", flush=True)
            watch(state, 600, 24, chain_sessions)
        return True
    print(f"{gmt7()} Gave up waiting for the source.", flush=True)
    return False


def schedule(args):
    """Windows Task Scheduler wrapper (survives the launching shell); pythonw logs to .cache/minio_mirror.log."""
    import subprocess
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    command = f'"{pythonw}" "{ROOT / "ops" / "minio_store.py"}" {args}'
    if len(command) > 230:  # schtasks silently truncates long /TR commands (observed ~250)
        raise RuntimeError(f"Task command too long for schtasks ({len(command)} chars)")
    subprocess.run(["schtasks", "/Create", "/TN", "FongbeMinioMirror", "/TR", command, "/SC", "ONCE",
                    "/ST", "23:59", "/F"], check=True, capture_output=True, text=True)
    subprocess.run(["schtasks", "/Run", "/TN", "FongbeMinioMirror"], check=True, capture_output=True, text=True)
    print(f"Scheduled task FongbeMinioMirror started: {command}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["status", "mirror", "wait-and-mirror", "schedule"])
    p.add_argument("--source", default="https://storage.googleapis.com/fongbe/",
                   help="organizers' bucket URL, or a local directory with audio_*.tar")
    p.add_argument("--interval", type=int, default=1800)
    p.add_argument("--max-hours", type=float, default=72)
    p.add_argument("--train-slug", help="after mirroring, submit this training session from the mirror")
    p.add_argument("--resume-from", help="previous training kernel to resume (with --train-slug)")
    p.add_argument("--chain-sessions", type=int, default=2)
    p.add_argument("--task-args", help="schedule: arguments for the scheduled wait-and-mirror run")
    a = p.parse_args()
    if a.action == "status":
        status()
    elif a.action == "mirror":
        src = LocalDirSource(a.source) if Path(a.source).is_dir() else GcsHttpSource(a.source)
        mirror(src)
    elif a.action == "schedule":
        schedule(a.task_args)
    else:
        wait_and_mirror(a.source, a.interval, a.max_hours, a.train_slug, a.resume_from, a.chain_sessions)


if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:  # pythonw / Task Scheduler
        (ROOT / ".cache").mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(ROOT / ".cache" / "minio_mirror.log", "a", encoding="utf-8", buffering=1)
    main()
