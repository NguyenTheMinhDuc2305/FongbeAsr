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
         "- Training blocker: competition audio repository from starter/README returns 401; awaiting current Zindi Chat URL.")


def submit(mode, slug):
    api = api_client()
    cfg = load_config(CONFIG)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Commit/push the intended changes before submitting; working tree is dirty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if mode == "train":
        # Fail locally BEFORE spending GPU quota if the data source is inaccessible.
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN") or False).dataset_info(
            cfg["download"]["repo_id"], revision=cfg["download"]["revision"])
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
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    })
    response = api.kernels_push(str(folder), acc=None if mode == "diagnose" else "NvidiaTeslaT4",
                                timeout="43200" if mode == "train" else "3600")
    errors = getattr(response, "error", None)
    if errors:
        raise RuntimeError(str(errors))
    job = {"kernel": kernel, "url": f"https://www.kaggle.com/code/{kernel}", "slug": slug,
           "mode": mode, "commit": commit, "submitted_utc": now(), "status": "submitted",
           "state_path": str(state_path), "notebook_sha256": sha256(notebook_path),
           "version": getattr(response, "version_number", None)}
    atomic_json(state_path, job)
    record_progress(job)
    print(json.dumps(job, indent=2))
    return state_path


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
        log = api.kernels_logs(job["kernel"])
        (state_path.parent / "remote.log").write_text(log, encoding="utf-8", newline="\n")
    except Exception as exc:
        job["log_fetch_error"] = type(exc).__name__
    terminal = job["status"] in ("complete", "error", "cancelled", "failed")
    if terminal and not job.get("downloaded"):
        destination = ROOT / "docs/kaggle_runs" / job["slug"] / job["commit"][:12]
        api.kernels_output(job["kernel"], str(destination), force=True, quiet=True)
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
    p.add_argument("--max-hours", type=float, default=24)
    a = p.parse_args()
    if a.action == "submit":
        submit(a.mode, a.slug)
    elif not a.state:
        p.error("--state is required")
    elif a.action == "poll":
        poll(a.state)
    elif a.action == "detach":
        detach(a.state, a.interval)
    else:
        watch(a.state, a.interval, a.max_hours)
