"""Download only needed audio archives, verify SHA256, extract safely and sequentially."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from huggingface_hub import HfApi, hf_hub_download
from src.common import atomic_json, audio_root, load_config, project_path, read_csv, sha256


def extract_needed(archive, root, needed):
    count = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive member: {member.name}")
            name = rel.as_posix()
            if name not in needed:
                continue
            if not member.isfile():
                raise ValueError(f"Expected regular WAV file: {name}")
            dest = root.joinpath(*rel.parts)
            if not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Path escapes audio root: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".partial")
            with tar.extractfile(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 2**20)
            tmp.replace(dest)
            count += 1
    return count


def download(config, include_test=False):
    cfg = load_config(config)
    root = audio_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in ("train_csv", "valid_general_csv", "valid_religious_csv"):
        rows += read_csv(project_path(cfg["data"][key]))
    if include_test or cfg["download"]["include_test"]:
        rows += read_csv(project_path(cfg["data"]["test_csv"]))
    needed = {r["audio_path"] for r in rows}
    missing = {p for p in needed if not (root / p).is_file()}
    if not missing:
        print(f"All {len(needed)} audio files already exist in {root}; training preflight checks headers.")
        return
    # 16kHz * 16-bit mono + one archive + working headroom.
    estimate = sum(float(r.get("duration") or 10) * 32000 for r in rows if r["audio_path"] in missing)
    free = shutil.disk_usage(root).free
    if free < estimate + 6e9:
        raise RuntimeError(f"Need approximately {(estimate+6e9)/1e9:.1f} GB free; have {free/1e9:.1f} GB. "
                           "Use a larger FONGBE_AUDIO_ROOT; do not silently train a subset.")
    repo = cfg["download"]["repo_id"]
    api = HfApi()
    revision = api.dataset_info(repo, revision=cfg["download"]["revision"]).sha
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    checksums_path = hf_hub_download(repo, "SHA256SUMS", repo_type="dataset", revision=revision,
                                    local_dir=root / ".downloads")
    sums = {}
    for line in Path(checksums_path).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*").removeprefix("./")] = digest
    prefixes = {f"audio_{PurePosixPath(p).parts[1]}_" for p in missing}
    archives = sorted(f for f in files if f.endswith(".tar") and any(Path(f).name.startswith(p) for p in prefixes))
    if not archives:
        raise RuntimeError("No matching audio_*.tar archives in the configured dataset repository")
    audit = {"repo": repo, "revision": revision, "archives": []}
    for name in archives:
        if name not in sums:
            raise ValueError(f"No official SHA256 for {name}")
        print(f"Downloading {name}", flush=True)
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision,
                                   local_dir=root / ".downloads"))
        if sha256(path) != sums[name]:
            path.unlink()
            raise ValueError(f"SHA256 mismatch: {name}; deleted corrupt download")
        extracted = extract_needed(path, root, needed)
        path.unlink()  # retain metadata, not a second 2GB archive
        audit["archives"].append({"name": name, "sha256": sums[name], "extracted": extracted})
        atomic_json(root / "download_manifest.json", audit)
    missing = sorted(p for p in needed if not (root / p).is_file())
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files still missing, first: {missing[:5]}")
    print(f"Ready: {len(needed)} files in {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    p.add_argument("--include-test", action="store_true")
    a = p.parse_args()
    download(a.config, a.include_test)
