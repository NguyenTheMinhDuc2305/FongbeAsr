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

if not __package__:  # executed as a script (e.g. by Windows Task Scheduler): make repo importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from src.common import ROOT, atomic_json, load_config, sha256

TASK_NAME = "FongbeKaggleWatcher"
# No flashing console windows when git is spawned from the windowless scheduled watcher.
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

CONFIG = "data/configs/training/proposal01.yaml"


GMT7 = timezone(timedelta(hours=7))  # user-facing times (docs, logs) are reported in GMT+7


def gmt7(value=None):
    """Format a UTC ISO string (or now) as 'YYYY-MM-DD HH:MM GMT+7' for human-facing notes."""
    if value in (None, "", "N/A") or not isinstance(value, str) and value is not None:
        moment = datetime.now(timezone.utc) if value in (None, "") else None
    else:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return value
    return moment.astimezone(GMT7).strftime("%Y-%m-%d %H:%M GMT+7") if moment else str(value)


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
         f"- Updated: {gmt7()}\n- Kernel: {job['kernel']}\n- URL: {job['url']}\n"
         f"- Commit: {job['commit']}\n- Mode: {job['mode']} (smoke is NOT competition training)\n"
         f"- Status: {job.get('status', 'submitted')}\n- Next poll: {gmt7(next_check)}\n"
         f"- Local state: {job['state_path']}\n- Output: {job.get('output_dir', 'not downloaded')}\n"
         f"- Details: {job.get('details', '')}\n"
         f"- Latest from log: {json.dumps(job.get('latest', {}), ensure_ascii=False)}\n"
         f"- Resume from: {job.get('resume_from') or 'fresh run'}")


def submit(mode, slug, resume_from=None, commit=None, audio_source="gcs"):
    api = api_client()
    cfg = load_config(CONFIG)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, **NO_WINDOW)
    if dirty.strip():
        raise RuntimeError("Commit/push the intended changes before submitting; working tree is dirty")
    if resume_from and not commit:
        # Exact resume requires the SAME code/config fingerprint as the previous session.
        previous = ROOT / ".cache/kaggle" / resume_from.split("/")[-1] / "job.json"
        if not previous.exists():
            raise RuntimeError(f"No local job record for {resume_from}; pass --commit explicitly")
        commit = json.loads(previous.read_text(encoding="utf-8"))["commit"]
    commit = commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, **NO_WINDOW).strip()
    if mode == "train" and resume_from is None and slug.endswith(("-s2", "-s3", "-s4")):
        raise RuntimeError("Continuation sessions must pass --resume-from <previous kernel>")
    audio_manifest = None
    if mode == "train" and audio_source == "minio":
        # Private mirror: verify completeness locally, then hand Kaggle only presigned GET URLs.
        from ops.minio_store import presigned_archives, client, read_manifest
        mirrored = read_manifest(client())
        expected = [n for n in mirrored.get("expected_archives", []) if n.startswith("audio_train_")]
        absent = [n for n in expected if n not in mirrored["archives"]]
        if not expected or absent:
            raise RuntimeError(f"MinIO mirror incomplete (expected {len(expected)}, missing {absent[:5]})")
        audio_manifest = presigned_archives(["train_reliable", "train_uncertain", "train_religious"])
    elif mode == "train":
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
            if audio_manifest:
                text = text.replace('AUDIO_MANIFEST_JSON = ""', f"AUDIO_MANIFEST_JSON = {json.dumps(audio_manifest)!r}")
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
           "version": getattr(response, "version_number", None), "resume_from": resume_from,
           "audio_source": audio_source}
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
             f"- Time: {gmt7()}\n- Commit: {job['commit']}\n- Status: {job['status']}\n"
             f"- Config: {CONFIG}; run hyperparameters are in resolved_config.yaml for training jobs.\n"
             f"- Local outputs: {destination}\n- Details: {job['details']}\n"
             f"- {'No competition WER: this was a GPU software smoke test.' if job['mode']=='smoke' else 'See collected_summary.json and downloaded run manifest/logs.'}")
    atomic_json(state_path, job)
    record_progress(job)
    print(f"{gmt7(job['checked_utc'])} {job['kernel']}: {job['status']}", flush=True)
    return terminal


def _session_text(job):
    parts = []
    for path in [Path(job["state_path"]).parent / "remote.log", *Path(job["output_dir"]).glob("*.log"),
                 *Path(job["output_dir"]).rglob("console.log")]:
        try:
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    return "\n".join(parts)


def next_session(job):
    """Chain a continuation when the session paused cleanly, or after an infrastructure failure
    (OOM kill, hang) that happened AFTER this session saved a checkpoint. Never after divergence."""
    if job.get("mode") != "train" or job.get("status") not in ("complete", "error"):
        return None, f"not chaining: mode={job.get('mode')} status={job.get('status')}"
    if job["status"] == "complete":
        statuses = list(Path(job["output_dir"]).rglob("status.json"))
        if len(statuses) != 1:
            return None, f"not chaining: found {len(statuses)} status.json in output"
        run_status = json.loads(statuses[0].read_text(encoding="utf-8"))
        if run_status.get("status") != "paused":
            return None, f"not chaining: training status={run_status.get('status')}"
        step = run_status.get("step")
    else:
        text = _session_text(job)
        if "Training diverged" in text or "FloatingPointError" in text or "DDP replicas diverged" in text:
            return None, "not chaining: numerical divergence needs human review"
        consoles = list(Path(job["output_dir"]).rglob("console.log"))
        saved = re.findall(r"Saved step=(\d+) checkpoint", "".join(
            c.read_text(encoding="utf-8", errors="ignore") for c in consoles))
        if not saved:
            return None, "not chaining: failed session saved no checkpoint"
        step = saved[-1]
    match = re.fullmatch(r"(.*-s)(\d+)", job["slug"])
    if not match:
        return None, "not chaining: slug lacks -sN suffix"
    slug = f"{match.group(1)}{int(match.group(2)) + 1}"
    return (submit("train", slug, resume_from=job["kernel"], audio_source=job.get("audio_source", "gcs")),
            f"chained {slug} from step {step} (previous status {job['status']})")


def watch(state_path, interval, max_hours, chain_sessions=0):
    end = time.monotonic() + max_hours*3600
    path = Path(state_path)
    if not path.is_absolute():
        path = ROOT / path
    if WINDOWLESS:  # scheduled task: switch the bootstrap log to the job's watcher.log
        sys.stdout = sys.stderr = open(path.parent / "watcher.log", "a", encoding="utf-8", buffering=1)
    while time.monotonic() < end:
        try:
            if poll(path):
                if chain_sessions > 0:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    new_state, message = next_session(job)
                    print(f"{gmt7()} {message}", flush=True)
                    if new_state is not None:
                        path, chain_sessions = Path(new_state), chain_sessions - 1
                        end = time.monotonic() + max_hours*3600
                        continue
                return
        except Exception as exc:
            # Do not log credentials/HTTP headers.
            print(f"{gmt7()} polling error: {type(exc).__name__}", flush=True)
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


def detach(state_path, interval, chain_sessions=0):
    path = Path(state_path).resolve()
    log = open(path.parent / "watcher.log", "ab", buffering=0)
    kwargs = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([sys.executable, "-u", "-m", "ops.kaggle_run", "watch",
                             "--state", str(path), "--interval", str(interval),
                             "--chain-sessions", str(chain_sessions)],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            env=dict(os.environ, PYTHONUTF8="1"), **kwargs)
    log.close()
    job = json.loads(path.read_text(encoding="utf-8"))
    job.update(watcher_pid=proc.pid, next_check=now())
    atomic_json(path, job)
    record_progress(job)
    print(f"Started local watcher PID={proc.pid}, every {interval}s, chaining up to {chain_sessions} "
          "continuation session(s). Requires this PC to remain running.")


def schedule(state_path, interval, chain_sessions):
    """Windows: run the watcher as a Task Scheduler task (survives the launching shell).
    Runs while the user is logged on and the PC is awake."""
    if os.name != "nt":
        raise RuntimeError("schedule is Windows-only; use detach on POSIX")
    rel = Path(state_path)
    rel = rel.resolve().relative_to(ROOT) if rel.is_absolute() else rel
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    command = (f'"{pythonw}" "{ROOT / "ops" / "kaggle_run.py"}" watch --state {rel.as_posix()} '
               f"--interval {interval} --chain-sessions {chain_sessions}")
    if len(command) > 230:  # schtasks silently truncates long /TR commands (observed ~250)
        raise RuntimeError(f"Task command too long for schtasks ({len(command)} chars)")
    subprocess.run(["schtasks", "/Create", "/TN", TASK_NAME, "/TR", command, "/SC", "ONCE",
                    "/ST", "23:59", "/F"], check=True, capture_output=True, text=True)
    subprocess.run(["schtasks", "/Run", "/TN", TASK_NAME], check=True, capture_output=True, text=True)
    print(f"Scheduled task {TASK_NAME} started: {command}")


WINDOWLESS = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "poll", "watch", "detach", "schedule"])
    p.add_argument("--mode", choices=["smoke", "diagnose", "train"], default="smoke")
    p.add_argument("--slug", default="fongbe-p01-2t4-smoke")
    p.add_argument("--state")
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--resume-from", help="previous training kernel owner/slug whose Output holds last.pt")
    p.add_argument("--commit", help="override pinned commit (default: HEAD, or the resumed job's commit)")
    p.add_argument("--audio-source", choices=["gcs", "minio"], default="gcs",
                   help="train: organizers' GCS bucket, or the private MinIO mirror via presigned URLs")
    p.add_argument("--chain-sessions", type=int, default=0,
                   help="watch/detach: auto-submit up to N continuation sessions while training is 'paused'")
    p.add_argument("--max-hours", type=float, default=24)
    a = p.parse_args()
    if a.action == "submit":
        submit(a.mode, a.slug, a.resume_from, a.commit, a.audio_source)
    elif not a.state:
        p.error("--state is required")
    elif a.action == "poll":
        poll(a.state)
    elif a.action == "detach":
        detach(a.state, a.interval, a.chain_sessions)
    elif a.action == "schedule":
        schedule(a.state, a.interval, a.chain_sessions)
    else:
        watch(a.state, a.interval, a.max_hours, a.chain_sessions)


if __name__ == "__main__":
    if sys.stdout is None or sys.stderr is None:  # pythonw (Task Scheduler): capture everything
        WINDOWLESS = True
        (ROOT / ".cache").mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(ROOT / ".cache" / "watcher_task.log", "a", encoding="utf-8", buffering=1)
        print(f"{now()} watcher task start: {sys.argv[1:]}", flush=True)
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
