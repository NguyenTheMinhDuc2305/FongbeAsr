"""Two-process Gloo tests: no GPU/account needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest
import yaml
from src.dataset import distributed_batches
from tests.test_training import fixture_config


@pytest.mark.parametrize("n", [2, 3, 5, 9, 15, 17, 103])
def test_shards_cover_each_sample_once(n):
    rows = [{"duration": i%7+1} for i in range(n)]
    batches = distributed_batches(rows, 4, 32, 42, 0, 2)
    assert sorted(i for b in batches for i in b) == list(range(n))
    for b in batches:
        shards = [b[r::2] for r in range(2)]
        assert all(0 < len(s) <= 4 for s in shards)
        assert set(shards[0]).isdisjoint(shards[1])


def test_two_rank_training_resume(tmp_path):
    if os.name == "nt":
        try:
            torch.distributed.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as exc:
            pytest.skip(f"This Windows torch build cannot create a Gloo device: {exc}")
    cfg, path = fixture_config(tmp_path)
    cfg["training"].update(num_devices=2, batch_size=1, gradient_accumulation=2,
                           max_epochs=2, max_steps=2)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = dict(os.environ)
    for k in ("FONGBE_RESUME", "FONGBE_AUDIO_ROOT", "FONGBE_OUTPUT_ROOT", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
        env.pop(k, None)
    env["PYTHONWARNINGS"] = "ignore::FutureWarning"
    proc = subprocess.run([sys.executable, "-m", "tests.ddp_runner", str(path), str(tmp_path)],
                          capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-6000:]
    run = Path(cfg["paths"]["output_root"])/cfg["run_name"]
    full = torch.load(tmp_path/"full.pt", weights_only=False, map_location="cpu")
    resumed = torch.load(run/"checkpoints/last.pt", weights_only=False, map_location="cpu")
    assert resumed["world_size"] == 2 and len(resumed["rank_states"]) == 2
    assert resumed["state"]["step"] == 2
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert full["scheduler"] == resumed["scheduler"]
    metrics = json.loads((run/"predictions/latest/metrics.json").read_text())
    assert metrics["all"]["utterances"] == 4
    assert json.loads((run/"manifest.json").read_text())["effective_batch_size"] == 4
