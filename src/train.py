"""Restartable single-node DDP Zipformer-S CTC training (NCCL on two T4 GPUs)."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import logging
import os
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
from src.dataset import SpeechDataset, distributed_batches, make_loader, preflight, specaugment
from src.distributed import DistributedContext, assert_ranks_synchronized, optimizer_step
from src.evaluate import evaluate
from src.model import ZipformerCTC, ctc_losses, load_transient_state, transient_state
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config):
    cfg = load_config(config)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    context = DistributedContext(cfg["training"])
    try:
        _train(cfg, context)
    finally:
        context.close()


def _train(cfg, ctx):
    tcfg, device = cfg["training"], ctx.device
    seed_everything(tcfg["seed"])
    torch.backends.cudnn.benchmark = False
    tokenizer = verify_artifacts(cfg)
    fingerprint = identity(cfg)
    run, root = run_dir(cfg), audio_root(cfg)
    resume_path = os.environ.get("FONGBE_RESUME") or cfg["resume"]["checkpoint"]
    saved = None
    if resume_path:
        # Only load trusted own full resume files (pickle); inference weights are weights_only-safe.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("format") != "fongbe_resume_v3":
            raise ValueError("Need a v3 last.pt (DDP + per-rank module state); older checkpoints need a warm-start run")
        if saved["world_size"] != ctx.world_size:
            raise ValueError("Exact resume requires the same world_size")
        if saved["identity"]["sha256"] != fingerprint["sha256"]:
            raise ValueError("Resume rejected: config/code/tokenizer/split fingerprint differs")

    def prepare_run():
        if run.exists() and not resume_path and any(run.iterdir()):
            raise FileExistsError(f"{run} exists: select a new run_name or explicitly resume last.pt")
        existing_manifest = run / "manifest.json"
        if existing_manifest.exists():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if existing["identity"]["sha256"] != fingerprint["sha256"]:
                raise ValueError("Output directory belongs to a different experiment")
        run.mkdir(parents=True, exist_ok=True)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                                                 encoding="utf-8", newline="\n")
        if (root / "download_manifest.json").exists():
            atomic_json(run / "reports/download_manifest.json",
                        json.loads((root / "download_manifest.json").read_text(encoding="utf-8")))
    ctx.primary_call(prepare_run)
    handlers = [logging.StreamHandler()]
    if ctx.primary:
        handlers.append(logging.FileHandler(run / "console.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO if ctx.primary else logging.WARNING,
                        format=f"%(asctime)s rank={ctx.rank} %(levelname)s %(message)s",
                        handlers=handlers, force=True)
    rows = read_csv(project_path(cfg["data"]["train_csv"]))
    def check_audio():
        kept = preflight(rows, root, tokenizer, cfg["features"], True, run / "reports/train_preflight.json")
        for name in ("general", "religious"):
            preflight(read_csv(project_path(cfg["data"][f"valid_{name}_csv"])), root, tokenizer,
                      cfg["features"], False, run / f"reports/valid_{name}_preflight.json")
        kept_ids = {r["id"] for r in kept}
        return [r["id"] for r in rows if r["id"] not in kept_ids]
    rejected = set(ctx.primary_call(check_audio))
    rows = [r for r in rows if r["id"] not in rejected]
    effective_ids = json_hash([r["id"] for r in rows])
    if saved and saved["effective_train_ids"] != effective_ids:
        raise ValueError("Usable audio set changed since checkpoint")
    dataset = SpeechDataset(rows, root, tokenizer, cfg["features"])
    raw_model = ZipformerCTC(cfg["model"], len(tokenizer), cfg["features"]["num_mel_bins"]).to(device)
    if saved:
        raw_model.load_state_dict(saved["model"])
    model = ctx.wrap(raw_model)
    ocfg = dict(tcfg["optimizer"])
    if ocfg.pop("name") != "ScaledAdam":
        raise ValueError("Only ScaledAdam supported")
    ocfg["betas"] = tuple(ocfg["betas"])
    optimizer = ScaledAdam(raw_model.named_parameters(), **ocfg)
    scfg = dict(tcfg["scheduler"])
    if scfg.pop("name") != "Eden":
        raise ValueError("Only Eden supported")
    scheduler = Eden(optimizer, **scfg)
    precision = tcfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("T4 does not support bf16; use fp16")
    amp = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "fp16", init_scale=1.0)
    state = {"step": 0, "epoch": 0, "batch_index": 0, "best_wer": None, "last_eval_step": -1,
             "skipped_updates": 0, "consecutive_skips": 0, "wall_seconds": 0.0,
             "log_loss_sum": 0.0, "log_sample_count": 0}
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["rank_states"][ctx.rank]["scaler"])
        load_transient_state(raw_model, saved["rank_states"][ctx.rank]["modules"])
        state.update(saved["state"])
        def restore_files():
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
                for name in ("model.pt", "tokenizer.json", "config.json"):
                    (run / "best" / name).unlink(missing_ok=True)
        ctx.primary_call(restore_files)
        restore_rng(saved["rank_states"][ctx.rank]["rng"], device)
        del saved
    else:
        # Same initial parameters, different augmentation/stochastic branches on each rank.
        seed_everything(tcfg["seed"] + ctx.rank)
    hardware = ctx.gather(torch.cuda.get_device_name(device) if device.type == "cuda" else str(device))
    def write_manifest():
        atomic_json(run / "tokenizer/tokenizer.json", tokenizer.config)
        atomic_json(run / "manifest.json", {
            "identity": fingerprint, "effective_train_ids": effective_ids, "git": git_info(),
            "python": sys.version, "torch": str(torch.__version__),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchaudio", "numpy", "PyYAML", "soundfile", "rapidfuzz", "huggingface-hub", "matplotlib")},
            "cuda_runtime": torch.version.cuda, "devices": hardware, "world_size": ctx.world_size,
            "effective_batch_size": tcfg["batch_size"] * ctx.world_size * tcfg["gradient_accumulation"],
            "parameters": sum(p.numel() for p in raw_model.parameters()), "train_utterances": len(rows),
            "resume_from": str(resume_path) if resume_path else None,
            "runtime_audio_root": str(root), "runtime_output": str(run),
            "notes": "DDP, native PyTorch Swoosh, plain CTC, scratch; not paper reproduction."})
    ctx.primary_call(write_manifest)
    logging.info("Run %s: ranks=%s parameters=%s train=%s resume step=%s", cfg["run_name"], ctx.world_size,
                 sum(p.numel() for p in raw_model.parameters()), len(rows), state["step"])
    stopped = {"requested": False}
    def stop_handler(signum, frame):
        stopped["requested"] = True
    old_handlers = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    start, prior_wall = time.monotonic(), state["wall_seconds"]
    loss_sum, loss_count = state["log_loss_sum"], state["log_sample_count"]

    def save():
        assert_ranks_synchronized(raw_model, ctx, f"step {state['step']}")
        rank_states = ctx.gather({"rng": capture_rng(device), "scaler": scaler.state_dict(),
                                  "modules": transient_state(raw_model)})
        def write_checkpoint():
            state.update(wall_seconds=prior_wall + time.monotonic()-start,
                         log_loss_sum=loss_sum, log_sample_count=loss_count)
            logs = {}
            for name in ("train.jsonl", "eval.jsonl"):
                p = run / "logs" / name
                logs[name] = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
            best_path = run / "best/model.pt"
            best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
            atomic_torch_save(run / "checkpoints/last.pt", {
                "format": "fongbe_resume_v3", "identity": fingerprint, "world_size": ctx.world_size,
                "effective_train_ids": effective_ids, "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rank_states": rank_states, "state": dict(state), "logs": logs, "best_payload": best})
            logging.info("Saved step=%s checkpoint (all rank RNG/scaler states)", state["step"])
        ctx.primary_call(write_checkpoint)

    def do_eval():
        scores = evaluate(raw_model, tokenizer, cfg, root, device, run / "predictions/latest", distributed=ctx)
        def log_eval():
            record = {"step": state["step"], "epoch": state["epoch"]}
            for name, score in scores.items():
                record.update({f"{name}_{k}": v for k, v in score.items()})
            append_jsonl(run / "logs/eval.jsonl", record)
            wer = scores["all"]["syllable_wer"]
            if state["best_wer"] is None or wer < state["best_wer"]:
                export_best(run, raw_model, tokenizer, cfg, state["step"], fingerprint)
            plot(run)
            logging.info("Validation step=%s general=%.4f religious=%.4f all=%.4f", state["step"],
                         scores["general"]["syllable_wer"], scores["religious"]["syllable_wer"], wer)
            return wer
        wer = ctx.primary_call(log_eval)
        state["best_wer"] = wer if state["best_wer"] is None else min(wer, state["best_wer"])
        state["last_eval_step"] = state["step"]
        model.train()

    finished, pause = False, False
    try:
        while state["epoch"] < tcfg["max_epochs"]:
            epoch = state["epoch"]
            batches = distributed_batches(rows, tcfg["batch_size"], tcfg["bucket_size"],
                                           tcfg["seed"], epoch, ctx.world_size)
            cursor = state["batch_index"]
            local_batches = [b[ctx.rank::ctx.world_size] for b in batches[cursor:]]
            loader = iter(make_loader(dataset, local_batches, tcfg["num_workers"], tcfg["seed"]+epoch+ctx.rank))
            model.train()
            while cursor < len(batches):
                pause = ctx.any(stopped["requested"] or time.monotonic()-start >= tcfg["session_hours"]*3600)
                if pause:
                    break
                if tcfg["max_steps"] is not None and state["step"] >= tcfg["max_steps"]:
                    finished = True
                    break
                count = min(tcfg["gradient_accumulation"], len(batches)-cursor)
                samples = sum(len(batches[j]) for j in range(cursor, cursor+count))
                optimizer.zero_grad(set_to_none=True)
                raw_model.set_step(state["step"])
                scheduler.step_epoch(epoch)
                scheduler.step_batch(state["step"])
                # Accumulate on device: no host sync inside the microbatch loop.
                update_loss = torch.zeros((), dtype=torch.float64, device=device)
                for micro in range(count):
                    batch = next(loader)
                    features = specaugment(batch["features"].to(device, non_blocking=True),
                                           batch["feature_lengths"], cfg["augmentation"])
                    with ctx.accumulation_context(model, synchronize=micro == count-1):
                        with torch.autocast("cuda", dtype=dtype) if amp else nullcontext():
                            probs, lengths = model(features, batch["feature_lengths"].to(device, non_blocking=True))
                        with torch.autocast(device.type, enabled=False):
                            losses = ctc_losses(probs.float(), lengths, batch, tokenizer.blank_id)
                            # DDP averages gradients: compensate so every GLOBAL utterance has weight 1/N.
                            loss = losses.sum() * ctx.world_size / samples
                        scaler.scale(loss).backward()
                    update_loss += losses.detach().sum()
                # Nonfinite loss implies nonfinite grads: checked once, synchronized across ranks.
                successful = optimizer_step(raw_model, optimizer, scaler, ctx)
                cursor += count
                state["batch_index"] = cursor
                if not successful:
                    state["skipped_updates"] += 1
                    state["consecutive_skips"] += 1
                    if state["consecutive_skips"] > 20:
                        raise FloatingPointError("20+ consecutive AMP skips: numerics diverged; resume last.pt")
                    continue
                state["consecutive_skips"] = 0
                state["step"] += 1
                if state["step"] == 1:
                    assert_ranks_synchronized(raw_model, ctx, "first optimizer step")
                loss_sum += ctx.sum(update_loss.item())
                loss_count += samples
                if state["step"] % tcfg["log_every"] == 0:
                    peaks = ctx.gather(torch.cuda.max_memory_allocated(device)/1e9 if device.type == "cuda" else 0.0)
                    def log_train():
                        record = {"step": state["step"], "epoch": epoch+cursor/len(batches),
                                  "train_loss": loss_sum/loss_count, "samples": loss_count,
                                  "learning_rate": optimizer.param_groups[0]["lr"], "amp_scale": scaler.get_scale(),
                                  "skipped_updates": state["skipped_updates"],
                                  "wall_seconds": prior_wall+time.monotonic()-start, "world_size": ctx.world_size,
                                  "peak_gpu_gb_per_rank": peaks}
                        append_jsonl(run / "logs/train.jsonl", record)
                        logging.info("step=%s epoch=%.3f loss=%.4f lr=%.5f", state["step"], record["epoch"],
                                     record["train_loss"], record["learning_rate"])
                    ctx.primary_call(log_train)
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
            if finished or pause:
                break
        finished = finished or state["epoch"] >= tcfg["max_epochs"]
        if finished and state["last_eval_step"] != state["step"]:
            do_eval()
        save()
        def finish():
            plot(run)
            atomic_json(run / "status.json", {"status": "completed" if finished else "paused", **state})
            logging.info("%s. Persist Output: %s", "Completed" if finished else "Paused", run)
        ctx.primary_call(finish)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    train(p.parse_args().config)
