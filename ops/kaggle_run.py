"""Authorized local deployment + durable polling. Credentials stay in ignored .env."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from dotenv import load_dotenv
from src.common import ROOT, atomic_json, load_config, sha256

CONFIG = "data/configs/training/proposal01.yaml"


def now():
    return datetime.now(timezone.utc).isoformat()


def api_client():
    load_dotenv(ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def note(path, marker, body):
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = start + "\n" + body + "\n" + end
    if start in text and end in text:
        text = text.split(start)[0] + section + text.split(end, 1)[1]
    else:
        text += "\n\n" + section + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def record_progress(job):
    next_check = job.get("next_check", "N/A")
    note(ROOT / "docs/progress.md", "kaggle-" + job["slug"],
         f"## Kaggle job: {job['mode']}\n"
         f"- Updated UTC: {now()}\n- Kernel: {job['kernel']}\n- URL: {job['url']}\n"
         f"- Commit: {job['commit']}\n- Mode: {job['mode']} (smoke is NOT competition training)\n"
         f"- Status: {job.get('status', 'submitted')}\n- Next poll UTC: {next_check}\n"
         f"- Local state: {job['state_path']}\n- Output: {job.get('output_dir', 'not downloaded')}\n"
         f"- Details: {job.get('details', '')}\n"
         f"- Latest from log: {json.dumps(job.get('latest', {}), ensure_ascii=False)}\n"
         f"- Resume from: {job.get('resume_from') or 'fresh run'}")


def submit(mode, slug, resume_from=None, commit=None):
    api = api_client()
    cfg = load_config(CONFIG)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Commit/push the intended changes before submitting; working tree is dirty")
    if resume_from and not commit:
        # Exact resume requires the SAME code/config fingerprint as the previous session.
        previous = ROOT / ".cache/kaggle" / resume_from.split("/")[-1] / "job.json"
        if not previous.exists():
            raise RuntimeError(f"No local job record for {resume_from}; pass --commit explicitly")
        commit = json.loads(previous.read_text(encoding="utf-8"))["commit"]
    commit = commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if mode == "train" and resume_from is None and slug.endswith(("-s2", "-s3", "-s4")):
        raise RuntimeError("Continuation sessions must pass --resume-from <previous kernel>")
    if mode == "train":
        # Fail locally BEFORE spending GPU quota if the data source is inaccessible.
        from src.download import list_bucket, object_metadata
        base = cfg["download"]["base_url"].rstrip("/") + "/"
        tars = [n for n, _ in list_bucket(base) if n.startswith("audio_train_") and n.endswith(".tar")]
        if not tars:
            raise RuntimeError(f"No training archives reachable at {base}")
        object_metadata(base + tars[0])  # verifies anonymous access + server MD5 availability
    folder = ROOT / ".cache/kaggle" / slug
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "job.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("status") not in ("complete", "error", "cancelled", "failed"):
            raise RuntimeError("Existing active job state; inspect it before launching another billed run")
    nb = json.loads((ROOT / "notebook/proposal01_kaggle.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"]).replace('GIT_REF = "main"', f'GIT_REF = "{commit}"')
            if resume_from:
                text = text.replace('RESUME_CHECKPOINT = ""', 'RESUME_CHECKPOINT = "auto"')
            cell["source"] = text.splitlines(keepends=True)
            cell["outputs"], cell["execution_count"] = [], None
    if mode == "diagnose":
        # CPU-only (no GPU quota): clone pinned commit, run 2-rank Gloo determinism/resume diagnosis.
        clone = "".join(nb["cells"][1]["source"]).replace(
            'subprocess.run(["bash", "bash/setup.sh"], cwd=PROJECT, check=True)',
            'subprocess.run(["python", "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], cwd=PROJECT, check=True)\n'
            'r = subprocess.run(["python", "-m", "tests.ddp_diagnose"], cwd=PROJECT)\n'
            'import shutil\n'
            'if (PROJECT / "ddp_diagnose.json").exists():\n'
            '    shutil.copy(PROJECT / "ddp_diagnose.json", "/kaggle/working/ddp_diagnose.json")\n'
            'r2 = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_distributed.py"], cwd=PROJECT)\n'
            'print("diagnose rc", r.returncode, "pytest test_distributed rc", r2.returncode)\n'
            'assert r.returncode == 0, r.returncode\n')
        nb["cells"] = [nb["cells"][0], dict(nb["cells"][1], source=clone.splitlines(keepends=True))]
    if mode == "smoke":
        nb["cells"][2]["source"] = [
            '# Batch-size probe on one T4 (synthetic features): memory + throughput for config choice.\n',
            'subprocess.run(["python", "-m", "src.smoke", "--probe", "4,8,12,16,24",\n',
            '                "--report", "/kaggle/working/gpu_probe.json"], cwd=PROJECT, check=True)\n',
            'import json, torch\n',
            'from datetime import datetime, timezone\n',
            'report = {"mode": "DDP GPU smoke only; no competition training",\n',
            '          "utc": datetime.now(timezone.utc).isoformat(),\n',
            '          "torch": str(torch.__version__),\n',
            '          "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],\n',
            f'          "commit": "{commit}", "setup_tests_and_ddp_capacity": "passed"}}\n',
            'Path("/kaggle/working/ddp_smoke_report.json").write_text(json.dumps(report, indent=2))\n',
            'print(report)\n',
        ]
        nb["cells"][3]["source"] = [
            'print("Smoke job complete. No audio download or competition training attempted.")\n'
        ]
    notebook_path = folder / "notebook.ipynb"
    atomic_json(notebook_path, nb)
    username = api.get_config_value("username")
    kernel = f"{username}/{slug}"
    atomic_json(folder / "kernel-metadata.json", {
        "id": kernel, "title": slug.replace("-", " "), "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": mode != "diagnose", "machine_shape": None if mode == "diagnose" else "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [],
        "kernel_sources": [resume_from] if resume_from else [], "model_sources": [],
    })
    response = api.kernels_push(str(folder), acc=None if mode == "diagnose" else "NvidiaTeslaT4",
                                timeout="43200" if mode == "train" else "3600")
    errors = getattr(response, "error", None)
    if errors:
        raise RuntimeError(str(errors))
    job = {"kernel": kernel, "url": f"https://www.kaggle.com/code/{kernel}", "slug": slug,
           "mode": mode, "commit": commit, "submitted_utc": now(), "status": "submitted",
           "state_path": str(state_path), "notebook_sha256": sha256(notebook_path),
           "version": getattr(response, "version_number", None), "resume_from": resume_from}
    atomic_json(state_path, job)
    record_progress(job)
    print(json.dumps(job, indent=2))
    return state_path


def stream_log(api, kernel, budget=45):
    import threading
    chunks = []

    def reader():
        try:
            for event in api.kernels_logs_stream(kernel):
                chunks.append(event.get("data", ""))
        except Exception as exc:
            chunks.append(f"\n[log stream error {type(exc).__name__}]")
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(budget)
    return "".join(chunks)


def summarize_log(raw):
    """Latest progress lines from the Kaggle log (JSON stream or plain text)."""
    try:
        text = "".join(r.get("data", "") for r in json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        text = raw
    out = {}
    patterns = {"train": r"step=(\d+) epoch=([\d.]+) loss=([\d.]+) lr=([\d.]+)",
                "validation": r"Validation step=(\d+) general=([\d.]+) religious=([\d.]+) all=([\d.]+)",
                "download": r"(OK audio_\S+: .*|Ready: .*|Need ~.*)"}
    for key, pattern in patterns.items():
        found = re.findall(pattern, text)
        if found:
            out[key] = found[-1] if isinstance(found[-1], str) else " ".join(found[-1])
            out[key + "_count"] = len(found)
    errors = re.findall(r"(\w+Error: .*)", text)
    if errors:
        out["last_error"] = errors[-1][:300]
    return out


def normalized_status(value):
    s = str(value).lower().split(".")[-1]
    return "complete" if s in ("completed", "complete") else s


def poll(state_path):
    state_path = Path(state_path)
    job = json.loads(state_path.read_text(encoding="utf-8"))
    api = api_client()
    response = api.kernels_status(job["kernel"])
    job["status"] = normalized_status(response.status)
    job["checked_utc"] = now()
    job["details"] = getattr(response, "failure_message", None) or ""
    # Logs are text from our own notebook. Never put tokens in notebook code/output.
    try:
        # Finished sessions expose the full log; running ones only via the SSE stream (time-boxed).
        log = api.kernels_logs(job["kernel"]) or stream_log(api, job["kernel"])
        (state_path.parent / "remote.log").write_text(log, encoding="utf-8", newline="\n")
        job["latest"] = summarize_log(log)
    except Exception as exc:
        job["log_fetch_error"] = type(exc).__name__
    terminal = job["status"] in ("complete", "error", "cancelled", "failed")
    if terminal and not job.get("downloaded"):
        destination = ROOT / "docs/kaggle_runs" / job["slug"] / job["commit"][:12]
        # Skip resume checkpoints (~0.5 GB): continuation sessions read them on Kaggle directly.
        api.kernels_output(job["kernel"], str(destination), file_pattern=r"^(?!.*checkpoints/)",
                           force=True, quiet=True)
        job.update(downloaded=True, output_dir=str(destination), next_check="N/A")
        metrics = list(destination.rglob("metrics.json"))
        summaries = []
        for path in metrics:
            try:
                summaries.append({"path": str(path), "metrics": json.loads(path.read_text(encoding="utf-8"))})
            except (ValueError, OSError):
                pass
        atomic_json(destination / "collected_summary.json", {
            "job": job, "metrics": summaries,
            "note": "Smoke outputs are software diagnostics, not Fongbe accuracy.",
        })
        note(ROOT / "docs/results.md", "kaggle-" + job["slug"],
             f"## Kaggle {job['mode']}: {job['kernel']}\n"
             f"- UTC: {now()}\n- Commit: {job['commit']}\n- Status: {job['status']}\n"
             f"- Config: {CONFIG}; run hyperparameters are in resolved_config.yaml for training jobs.\n"
             f"- Local outputs: {destination}\n- Details: {job['details']}\n"
             f"- {'No competition WER: this was a GPU software smoke test.' if job['mode']=='smoke' else 'See collected_summary.json and downloaded run manifest/logs.'}")
    atomic_json(state_path, job)
    record_progress(job)
    print(f"{job['checked_utc']} {job['kernel']}: {job['status']}", flush=True)
    return terminal


def watch(state_path, interval, max_hours):
    end = time.monotonic() + max_hours*3600
    path = Path(state_path)
    while time.monotonic() < end:
        try:
            if poll(path):
                return
        except Exception as exc:
            # Do not log credentials/HTTP headers.
            print(f"{now()} polling error: {type(exc).__name__}", flush=True)
        job = json.loads(path.read_text(encoding="utf-8"))
        job["next_check"] = (datetime.now(timezone.utc)+timedelta(seconds=interval)).isoformat()
        job["watcher_pid"] = os.getpid()
        atomic_json(path, job)
        record_progress(job)
        time.sleep(interval)
    job = json.loads(path.read_text(encoding="utf-8"))
    job["details"] = "Watcher deadline reached; poll manually. Job was NOT cancelled."
    job["next_check"] = "manual check required"
    atomic_json(path, job)
    record_progress(job)


def detach(state_path, interval):
    path = Path(state_path).resolve()
    log = open(path.parent / "watcher.log", "ab", buffering=0)
    kwargs = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-u", "-m", "ops.kaggle_run", "watch",
                             "--state", str(path), "--interval", str(interval)],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            env=dict(os.environ, PYTHONUTF8="1"), **kwargs)
    log.close()
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update(watcher_pid=proc.pid, next_check=now())
    atomic_json(path, job)
    record_progress(job)
    print(f"Started local watcher PID={proc.pid}, every {interval}s. Requires this PC to remain running.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "poll", "watch", "detach"])
    p.add_argument("--mode", choices=["smoke", "diagnose", "train"], default="smoke")
    p.add_argument("--slug", default="fongbe-p01-2t4-smoke")
    p.add_argument("--state")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--resume-from", help="previous training kernel owner/slug whose Output holds last.pt")
    p.add_argument("--commit", help="override pinned commit (default: HEAD, or the resumed job's commit)")
    p.add_argument("--max-hours", type=float, default=24)
    a = p.parse_args()
    if a.action == "submit":
        submit(a.mode, a.slug, a.resume_from, a.commit)
    elif not a.state:
        p.error("--state is required")
    elif a.action == "poll":
        poll(a.state)
    elif a.action == "detach":
        detach(a.state, a.interval)
    else:
        watch(a.state, a.interval, a.max_hours)
