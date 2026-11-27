"""Configured DDP/FP16/longest-shape capacity check before downloading the corpus."""
import argparse
from contextlib import nullcontext
import time
import torch
from src.common import load_config
from src.distributed import DistributedContext, optimizer_step
from src.model import ZipformerCTC, ctc_losses
from src.train import verify_artifacts, seed_everything
from src.vendor.icefall.optim import ScaledAdam


def smoke(config):
    cfg = load_config(config)
    ctx = DistributedContext(cfg["training"])
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("Capacity check requires GPUs")
        torch.set_num_threads(cfg["training"]["cpu_threads"])
        seed_everything(42)
        tok = verify_artifacts(cfg)
        raw = ZipformerCTC(cfg["model"], len(tok), cfg["features"]["num_mel_bins"]).to(ctx.device).train()
        raw.set_step(max(cfg["model"]["dropout_end_step"], cfg["model"]["warmup_batches"]))
        model = ctx.wrap(raw)
        seed_everything(42+ctx.rank)
        batch_size = cfg["training"]["batch_size"]
        length = int((cfg["data"]["max_train_seconds"]*1000-cfg["features"]["frame_length"])
                     /cfg["features"]["frame_shift"])+1
        x = torch.randn(batch_size, length, cfg["features"]["num_mel_bins"], device=ctx.device)
        lens = torch.full((batch_size,), length, dtype=torch.long, device=ctx.device)
        ids = tok.encode("Un nɔ́ ɖu nǔ")
        batch = {"labels": torch.tensor(ids*batch_size),
                 "label_lengths": torch.full((batch_size,), len(ids), dtype=torch.long),
                 "minimum_lengths": torch.full((batch_size,), tok.minimum_ctc_frames(ids), dtype=torch.long)}
        kwargs = dict(cfg["training"]["optimizer"])
        kwargs.pop("name")
        kwargs["betas"] = tuple(kwargs["betas"])
        optimizer = ScaledAdam(raw.named_parameters(), **kwargs)
        precision = cfg["training"]["precision"]
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("T4 needs FP16, not BF16")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1.0)
        torch.cuda.reset_peak_memory_stats(ctx.device)
        t0 = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(2):
                with ctx.accumulation_context(model, synchronize=micro == 1):
                    with torch.autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                        probs, out_lens = model(x, lens)
                    with torch.autocast("cuda", enabled=False):
                        loss = ctc_losses(probs.float(), out_lens, batch, tok.blank_id).mean()/2
                    if ctx.any(not bool(torch.isfinite(loss))):
                        raise FloatingPointError("Nonfinite smoke loss")
                    scaler.scale(loss).backward()
            if not optimizer_step(raw, optimizer, scaler, ctx):
                raise FloatingPointError("Smoke AMP overflow; inspect before full training")
        torch.cuda.synchronize(ctx.device)
        reports = ctx.gather({"rank": ctx.rank, "gpu": torch.cuda.get_device_name(ctx.device),
                              "peak_gb": torch.cuda.max_memory_allocated(ctx.device)/1e9,
                              "seconds": time.perf_counter()-t0})
        if ctx.primary:
            print("DDP capacity/AMP smoke OK:", reports, flush=True)
    finally:
        ctx.close()


def probe(config, sizes, report_path):
    """Single-GPU (no DDP: an OOM on one rank would hang the other) batch-size probe.
    Worst case = every utterance at max_train_seconds; typical = 10 s. Synthetic features:
    measures memory/throughput only, not accuracy."""
    import json
    cfg = load_config(config)
    device = torch.device("cuda", 0)
    tok = verify_artifacts(cfg)
    seed_everything(42)
    model = ZipformerCTC(cfg["model"], len(tok), cfg["features"]["num_mel_bins"]).to(device).train()
    model.set_step(max(cfg["model"]["dropout_end_step"], cfg["model"]["warmup_batches"]))
    kwargs = dict(cfg["training"]["optimizer"])
    kwargs.pop("name")
    kwargs["betas"] = tuple(kwargs["betas"])
    optimizer = ScaledAdam(model.named_parameters(), **kwargs)
    scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
    ids = tok.encode("Un nɔ́ ɖu nǔ " * 6)
    total_hours = 335.36  # train split metadata; for a rough epoch-time estimate only
    results = []

    def frames(seconds):
        return int((seconds*1000 - cfg["features"]["frame_length"]) / cfg["features"]["frame_shift"]) + 1

    def run(bs, seconds, repeats):
        t = frames(seconds)
        x = torch.randn(bs, t, cfg["features"]["num_mel_bins"], device=device)
        lens = torch.full((bs,), t, dtype=torch.long)
        batch = {"labels": torch.tensor(ids*bs), "feature_lengths": lens,
                 "label_lengths": torch.full((bs,), len(ids), dtype=torch.long),
                 "minimum_lengths": torch.full((bs,), tok.minimum_ctc_frames(ids), dtype=torch.long)}
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                probs, out = model(x, lens.to(device))
            loss = ctc_losses(probs.float(), out, batch, tok.blank_id).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        return min(times[1:]) if len(times) > 1 else times[0]

    capacity = torch.cuda.get_device_properties(device).total_memory / 1e9
    for bs in sizes:
        entry = {"batch_per_gpu": bs}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            entry["sec_per_update_30s"] = run(bs, cfg["data"]["max_train_seconds"], 3)
            entry["peak_allocated_gb_30s"] = torch.cuda.max_memory_allocated(device) / 1e9
            entry["peak_reserved_gb_30s"] = torch.cuda.max_memory_reserved(device) / 1e9
            entry["sec_per_update_10s"] = run(bs, 10.0, 4)
            per_gpu_audio_rate = bs * 10.0 / entry["sec_per_update_10s"]  # audio seconds / second
            entry["est_hours_per_epoch_2gpu_no_overhead"] = total_hours * 3600 / (2 * per_gpu_audio_rate) / 3600
            entry["fits_with_25pct_headroom"] = entry["peak_reserved_gb_30s"] < 0.75 * capacity
        except torch.OutOfMemoryError:
            entry["oom"] = True
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        results.append(entry)
        print(json.dumps(entry), flush=True)
    report = {"gpu": torch.cuda.get_device_name(device), "capacity_gb": capacity, "precision": "fp16",
              "model_parameters": sum(p.numel() for p in model.parameters()), "results": results,
              "note": "Synthetic worst-case lengths; single GPU; DDP adds communication + bucket memory."}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("GPU_PROBE", json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    p.add_argument("--probe", help="comma-separated per-GPU batch sizes, e.g. 4,8,16,24 (single GPU)")
    p.add_argument("--report", default="gpu_probe.json")
    a = p.parse_args()
    if a.probe:
        probe(a.config, [int(x) for x in a.probe.split(",")], a.report)
    else:
        smoke(a.config)
