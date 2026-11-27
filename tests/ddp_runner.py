"""Spawn 2-rank Gloo training runs (Linux). Usage: python -m tests.ddp_runner CONFIG DIR PHASE...

Phases: full (captures step1.pt and copies final last.pt to full.pt), again (fresh rerun ->
again.pt), resume (from step1.pt). Avoids pytest fork/CUDA state.
"""
import os
from pathlib import Path
import shutil
import sys
import torch.multiprocessing as mp
import yaml


def worker(rank, config, directory, phase):
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      FONGBE_DDP_INIT_METHOD=(Path(directory)/("store_"+phase)).resolve().as_uri())
    from src import train as trainer
    checkpoint = Path(directory)/"step1.pt"
    if phase == "resume":
        os.environ["FONGBE_RESUME"] = str(checkpoint)
    original = trainer.atomic_torch_save
    def capture(path, value):
        original(path, value)
        if rank == 0 and phase == "full" and value.get("state", {}).get("step") == 1:
            original(checkpoint, value)
    trainer.atomic_torch_save = capture
    trainer.train(config)


def run(config, directory, phases):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out = Path(cfg["paths"]["output_root"])/cfg["run_name"]
    for phase in phases:
        if phase in ("full", "again") and out.exists():
            shutil.rmtree(out)
        mp.spawn(worker, args=(config, directory, phase), nprocs=2, join=True)
        if phase in ("full", "again"):
            shutil.copy2(out/"checkpoints/last.pt", Path(directory)/(phase+".pt"))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:] or ["full", "resume"])
