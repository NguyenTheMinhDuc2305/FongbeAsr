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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    smoke(p.parse_args().config)
