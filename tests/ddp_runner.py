"""Subprocess harness avoids pytest/fork/CUDA state; file store also works on Windows."""
import os
from pathlib import Path
import shutil
import sys
import torch.multiprocessing as mp


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


if __name__ == "__main__":
    config, directory = sys.argv[1:]
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    mp.spawn(worker, args=(config, directory, "full"), nprocs=2, join=True)
    run = Path(cfg["paths"]["output_root"])/cfg["run_name"]
    shutil.copy2(run/"checkpoints/last.pt", Path(directory)/"full.pt")
    mp.spawn(worker, args=(config, directory, "resume"), nprocs=2, join=True)
