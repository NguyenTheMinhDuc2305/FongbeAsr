"""Check configured GPU model/AMP/optimizer and longest training shape before download."""
import argparse
import time
import torch
from src.common import load_config
from src.model import ZipformerCTC, ctc_losses
from src.train import verify_artifacts
from src.vendor.icefall.optim import ScaledAdam


def smoke(config):
    cfg = load_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("This capacity test requires Kaggle GPU")
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    tok = verify_artifacts(cfg)
    model = ZipformerCTC(cfg["model"], len(tok), cfg["features"]["num_mel_bins"]).cuda().train()
    # Test the fully warmed-up encoder, not only the cheaper initial layerdrop path.
    model.set_step(max(cfg["model"]["dropout_end_step"], cfg["model"]["warmup_batches"]))
    batch_size = cfg["training"]["batch_size"]
    length = int((cfg["data"]["max_train_seconds"]*1000 - cfg["features"]["frame_length"])
                 / cfg["features"]["frame_shift"]) + 1
    x = torch.randn(batch_size, length, cfg["features"]["num_mel_bins"], device="cuda")
    lens = torch.full((batch_size,), length, dtype=torch.long, device="cuda")
    ids = tok.encode("Un nɔ́ ɖu nǔ")
    batch = {"labels": torch.tensor(ids*batch_size, dtype=torch.long),
             "label_lengths": torch.full((batch_size,), len(ids), dtype=torch.long),
             "minimum_lengths": torch.full((batch_size,), tok.minimum_ctc_frames(ids), dtype=torch.long)}
    kwargs = dict(cfg["training"]["optimizer"])
    kwargs.pop("name")
    kwargs["betas"] = tuple(kwargs["betas"])
    optimizer = ScaledAdam(model.named_parameters(), **kwargs)
    precision = cfg["training"]["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bf16 is unsupported on this GPU")
    enabled = precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=1.0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            probs, out_lens = model(x, lens)
        with torch.autocast("cuda", enabled=False):
            loss = ctc_losses(probs.float(), out_lens, batch, tok.blank_id).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite GPU smoke loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite GPU smoke gradient: {name}")
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    print(f"GPU capacity/AMP smoke OK: {sum(p.numel() for p in model.parameters()):,} params, "
          f"batch={batch_size}, length={cfg['data']['max_train_seconds']}s, "
          f"peak allocated={torch.cuda.max_memory_allocated()/1e9:.2f} GB, "
          f"two updates={time.perf_counter()-t0:.1f}s. Not an accuracy/CPU benchmark.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/training/proposal01.yaml")
    smoke(p.parse_args().config)
