"""Shared paths, immutable run identity, and atomic artifact I/O."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    with open(project_path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["model"]["type"] != "zipformer_ctc":
        raise ValueError("Only zipformer_ctc is implemented; do not silently substitute models")
    for key in ("gradient_accumulation", "batch_size", "max_epochs", "log_every", "save_every", "eval_every"):
        if cfg["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if cfg["training"]["session_hours"] <= 0:
        raise ValueError("session_hours must be positive")
    if cfg["training"]["max_steps"] is not None and cfg["training"]["max_steps"] < 1:
        raise ValueError("max_steps must be positive or null")
    if cfg["evaluation"]["decoder"] != "greedy" or cfg["evaluation"]["primary_metric"] != "all_syllable_wer":
        raise ValueError("Only greedy CTC and pooled corpus syllable-WER are implemented")
    if cfg["features"]["sample_rate"] != 16000 or not cfg["features"]["snip_edges"] or cfg["features"]["dither"] != 0:
        raise ValueError("P01 requires deterministic 16kHz fbank: snip_edges=true, dither=0")
    if cfg["model"]["causal"] or cfg["model"]["output_downsampling_factor"] != 2:
        raise ValueError("P01 supports non-streaming with output_downsampling_factor=2 only")
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def audio_root(cfg):
    return Path(os.environ.get("FONGBE_AUDIO_ROOT", cfg["paths"]["audio_root"])).resolve()


def run_dir(cfg):
    name = cfg["run_name"]
    if Path(name).name != name or name in ("", ".", "..") or "/" in name or "\\" in name:
        raise ValueError("run_name must be a simple directory name")
    return Path(os.environ.get("FONGBE_OUTPUT_ROOT", cfg["paths"]["output_root"])) / name


def identity(cfg):
    # Absolute runtime roots and a resume path do not change the experiment.
    core = {k: v for k, v in cfg.items() if k not in ("paths", "resume")}
    paths = [cfg["tokenizer"] + "/tokenizer.json", cfg["data"]["manifest"]]
    paths += [cfg["data"][k] for k in ("train_csv", "valid_general_csv", "valid_religious_csv")]
    paths += ["data/normalize.py", "data/syllabify.py", "data/score.py"]
    # Catch code edits too, including uncommitted changes.
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src").rglob("*.py"))]
    hashes = {p: sha256(project_path(p)) for p in paths}
    return {"config_sha256": json_hash(core), "files": hashes, "sha256": json_hash([core, hashes])}


def git_info():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
