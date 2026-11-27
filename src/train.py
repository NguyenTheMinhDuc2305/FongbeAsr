"""Single-GPU, restartable Zipformer-S CTC training; no distributed/k2 dependency."""
import argparse
from contextlib import nullcontext
import json
import importlib.metadata
import logging
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
import yaml
from src.charts import plot
from src.checkpoint import atomic_torch_save, capture_rng, restore_rng, export_best
from src.common import (append_jsonl, atomic_json, audio_root, git_info, identity,
                        json_hash, load_config, project_path, read_csv, run_dir, sha256)
from src.dataset import SpeechDataset, epoch_batches, make_loader, preflight, specaugment
from src.evaluate import evaluate
from src.model import ZipformerCTC, ctc_losses
from src.tokenizer import CharacterTokenizer
from src.vendor.icefall.optim import ScaledAdam, Eden


def verify_artifacts(cfg):
    manifest = json.loads(project_path(cfg["data"]["manifest"]).read_text(encoding="utf-8"))
    for name, key in (("train", "train_csv"), ("valid_general", "valid_general_csv"),
                      ("valid_religious", "valid_religious_csv")):
        if sha256(project_path(cfg["data"][key])) != manifest["summary"][name]["sha256"]:
            raise ValueError(f"Split hash mismatch: {name}")
    if sha256(project_path(cfg["tokenizer"]) / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash mismatch")
    if cfg["data"]["max_train_seconds"] != manifest.get("max_train_seconds", cfg["data"]["max_train_seconds"]):
        raise ValueError("Duration filter differs from saved split; prepare a new version first")
    tokenizer = CharacterTokenizer(project_path(cfg["tokenizer"]))
    if tokenizer.config["normalizer_sha256"] != sha256(project_path("data/normalize.py")):
        raise ValueError("Normalizer changed since tokenizer preparation")
    if tokenizer.config["fit_split_sha256"] != manifest["summary"]["train"]["sha256"]:
        raise ValueError("Tokenizer was fitted on a different split")
    return tokenizer


def train(config):
    cfg = load_config(config)
    tcfg = cfg["training"]
    torch.set_num_threads(tcfg["cpu_threads"])
    device = torch.device(tcfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested. Enable Kaggle GPU; CPU training is not silently substituted.")
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = False
    tokenizer = verify_artifacts(cfg)
    fingerprint = identity(cfg)
    run = run_dir(cfg)
    resume_path = os.environ.get("FONGBE_RESUME") or cfg["resume"]["checkpoint"]
    if run.exists() and not resume_path and any(run.iterdir()):
        raise FileExistsError(f"{run} exists: set a new run_name or explicitly resume its last.pt")
    if resume_path:
        # A full training checkpoint contains Python RNG state. Load ONLY your own trusted checkpoint.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("format") != "fongbe_resume_v1":
            raise ValueError("Expected checkpoints/last.pt, not inference-only best/model.pt")
        if saved["identity"]["sha256"] != fingerprint["sha256"]:
            raise ValueError("Resume rejected: config/code/tokenizer/split fingerprint differs. "
                             "Restore the exact commit/config; warm-start is a separate experiment.")
    else:
        saved = None
    existing_manifest = run / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing["identity"]["sha256"] != fingerprint["sha256"]:
            raise ValueError("Output directory belongs to another experiment; use a different output root")
    run.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run / "console.log", encoding="utf-8")],
                        force=True)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                             encoding="utf-8", newline="\n")
    root = audio_root(cfg)
    download_manifest = root / "download_manifest.json"
    if download_manifest.exists():
        atomic_json(run / "reports/download_manifest.json",
                    json.loads(download_manifest.read_text(encoding="utf-8")))
    rows = read_csv(project_path(cfg["data"]["train_csv"]))
    rows = preflight(rows, root, tokenizer, cfg["features"], True, run / "reports/train_preflight.json")
    for name in ("general", "religious"):
        preflight(read_csv(project_path(cfg["data"][f"valid_{name}_csv"])), root, tokenizer,
                  cfg["features"], False, run / f"reports/valid_{name}_preflight.json")
    effective_ids = json_hash([r["id"] for r in rows])
    if saved and saved["effective_train_ids"] != effective_ids:
        raise ValueError("Usable audio set changed since checkpoint")
    dataset = SpeechDataset(rows, root, tokenizer, cfg["features"])
    model = ZipformerCTC(cfg["model"], len(tokenizer), cfg["features"]["num_mel_bins"]).to(device)
    ocfg = dict(tcfg["optimizer"])
    if ocfg.pop("name") != "ScaledAdam":
        raise ValueError("Only ScaledAdam supported by P01")
    ocfg["betas"] = tuple(ocfg["betas"])
    optimizer = ScaledAdam(model.named_parameters(), **ocfg)
    scfg = dict(tcfg["scheduler"])
    if scfg.pop("name") != "Eden":
        raise ValueError("Only Eden supported by P01")
    scheduler = Eden(optimizer, **scfg)
    precision = tcfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; update config before launching")
    amp = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "fp16", init_scale=1.0)
    state = {"step": 0, "epoch": 0, "batch_index": 0, "best_wer": None,
             "last_eval_step": -1, "skipped_updates": 0, "wall_seconds": 0.0,
             "log_loss_sum": 0.0, "log_sample_count": 0}
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state.update(saved["state"])
        # Reconstruct durable log history, removing logs newer than checkpoint on crash recovery.
        for name, records in saved["logs"].items():
            p = run / "logs" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                         encoding="utf-8", newline="\n")
        if saved.get("best_payload") is not None:
            atomic_torch_save(run / "best/model.pt", saved["best_payload"])
            atomic_json(run / "best/tokenizer.json", tokenizer.config)
            atomic_json(run / "best/config.json", {"model": cfg["model"], "features": cfg["features"],
                                                   "step": saved["best_payload"]["step"]})
        else:
            # Rolling back before first eval must not retain a best model from the future.
            for name in ("model.pt", "tokenizer.json", "config.json"):
                (run / "best" / name).unlink(missing_ok=True)
        restore_rng(saved["rng"])
        del saved
    atomic_json(run / "tokenizer/tokenizer.json", tokenizer.config)
    atomic_json(run / "manifest.json", {
        "identity": fingerprint, "effective_train_ids": effective_ids, "git": git_info(),
        "python": sys.version, "torch": str(torch.__version__),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchaudio", "numpy", "PyYAML", "soundfile", "rapidfuzz", "huggingface-hub", "matplotlib")},
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "parameters": sum(p.numel() for p in model.parameters()), "train_utterances": len(rows),
        "resume_from": str(resume_path) if resume_path else None,
        "runtime_audio_root": str(root), "runtime_output": str(run),
        "notes": "Pure PyTorch Swoosh, plain CTC, scratch; not paper/CR-CTC reproduction.",
    })
    logging.info("Run %s: %s parameters, %s train examples, resume step %s",
                 cfg["run_name"], sum(p.numel() for p in model.parameters()), len(rows), state["step"])
    stopped = {"requested": False}
    def stop_handler(signum, frame):
        stopped["requested"] = True
        logging.warning("Stop requested; finishing current optimizer update and saving.")
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop_handler)
    start = time.monotonic()
    prior_wall = state["wall_seconds"]
    loss_sum = state["log_loss_sum"]
    loss_count = state["log_sample_count"]

    def save():
        state["log_loss_sum"] = loss_sum
        state["log_sample_count"] = loss_count
        state["wall_seconds"] = prior_wall + time.monotonic() - start
        logs = {}
        for name in ("train.jsonl", "eval.jsonl"):
            p = run / "logs" / name
            logs[name] = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
        # Embed best inference weights so last.pt alone is sufficient to resume on a new Kaggle session.
        best_path = run / "best/model.pt"
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
        payload = {"format": "fongbe_resume_v1", "identity": fingerprint, "effective_train_ids": effective_ids,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "state": dict(state), "rng": capture_rng(), "logs": logs, "best_payload": best_payload}
        atomic_torch_save(run / "checkpoints/last.pt", payload)
        logging.info("Saved resumable checkpoint at step %s", state["step"])

    def do_eval():
        scores = evaluate(model, tokenizer, cfg, root, device, run / "predictions/latest")
        record = {"step": state["step"], "epoch": state["epoch"]}
        for name, score in scores.items():
            for key, value in score.items():
                record[f"{name}_{key}"] = value
        append_jsonl(run / "logs/eval.jsonl", record)
        wer = scores["all"]["syllable_wer"]
        if state["best_wer"] is None or wer < state["best_wer"]:
            state["best_wer"] = wer
            export_best(run, model, tokenizer, cfg, state["step"], fingerprint)
        state["last_eval_step"] = state["step"]
        logging.info("Validation step %s: general=%.4f religious=%.4f pooled=%.4f",
                     state["step"], scores["general"]["syllable_wer"], scores["religious"]["syllable_wer"], wer)
        plot(run)
        model.train()

    finished = False
    try:
        while state["epoch"] < tcfg["max_epochs"]:
            epoch = state["epoch"]
            batches = epoch_batches(rows, tcfg["batch_size"], tcfg["bucket_size"], tcfg["seed"], epoch)
            cursor = state["batch_index"]
            loader = iter(make_loader(dataset, batches[cursor:], tcfg["num_workers"], tcfg["seed"]+epoch))
            model.train()
            while cursor < len(batches):
                if stopped["requested"] or time.monotonic()-start >= tcfg["session_hours"]*3600:
                    break
                if tcfg["max_steps"] is not None and state["step"] >= tcfg["max_steps"]:
                    finished = True
                    break
                count = min(tcfg["gradient_accumulation"], len(batches)-cursor)
                samples = sum(len(batches[j]) for j in range(cursor, cursor+count))
                optimizer.zero_grad(set_to_none=True)
                model.set_step(state["step"])
                scheduler.step_epoch(epoch)
                scheduler.step_batch(state["step"])
                update_loss = 0.0
                for _ in range(count):
                    batch = next(loader)
                    features = specaugment(batch["features"].to(device), batch["feature_lengths"], cfg["augmentation"])
                    context = torch.autocast("cuda", dtype=dtype) if amp else nullcontext()
                    with context:
                        probs, lengths = model(features, batch["feature_lengths"].to(device))
                    with torch.autocast(device.type, enabled=False):
                        losses = ctc_losses(probs.float(), lengths, batch, tokenizer.blank_id)
                        loss = losses.sum()/samples
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {state['step']}; resume previous last.pt")
                    scaler.scale(loss).backward()
                    update_loss += float(losses.detach().sum())
                if not scaler.is_enabled():
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(f"Nonfinite gradient: {name}; resume previous last.pt")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                cursor += count
                state["batch_index"] = cursor
                successful = scaler.get_scale() >= previous_scale
                if not successful:
                    state["skipped_updates"] += 1
                    logging.warning("AMP skipped update at cursor %s, scale=%s", cursor, scaler.get_scale())
                    if state["skipped_updates"] > 100:
                        raise FloatingPointError("Too many AMP skips; inspect numerics")
                    continue
                state["step"] += 1
                loss_sum += update_loss
                loss_count += samples
                if state["step"] % tcfg["log_every"] == 0:
                    record = {"step": state["step"], "epoch": epoch+cursor/len(batches),
                              "train_loss": loss_sum/loss_count, "samples": loss_count,
                              "learning_rate": optimizer.param_groups[0]["lr"], "amp_scale": scaler.get_scale(),
                              "wall_seconds": prior_wall+time.monotonic()-start,
                              "peak_gpu_gb": torch.cuda.max_memory_allocated(device)/1e9 if device.type=="cuda" else 0.0}
                    append_jsonl(run / "logs/train.jsonl", record)
                    logging.info("step=%s epoch=%.3f loss=%.4f lr=%.5f", state["step"], record["epoch"],
                                 record["train_loss"], record["learning_rate"])
                    loss_sum = loss_count = 0
                if state["step"] % tcfg["eval_every"] == 0:
                    do_eval()
                    save()
                elif state["step"] % tcfg["save_every"] == 0:
                    save()
            del loader
            if cursor == len(batches):
                state["epoch"] += 1
                state["batch_index"] = 0
                if tcfg["evaluate_each_epoch"] and state["last_eval_step"] != state["step"]:
                    do_eval()
                save()
            if finished or stopped["requested"] or time.monotonic()-start >= tcfg["session_hours"]*3600:
                break
        finished = finished or state["epoch"] >= tcfg["max_epochs"]
        if finished and state["last_eval_step"] != state["step"]:
            do_eval()
        save()
        plot(run)
        atomic_json(run / "status.json", {"status": "completed" if finished else "paused", **state})
        logging.info("%s. Persist %s before terminating the Kaggle session.",
                     "Completed" if finished else "Paused", run)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    train(p.parse_args().config)
