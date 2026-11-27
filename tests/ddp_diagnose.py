"""Quantify DDP determinism/resume drift on Linux CPU (Gloo). Prints a JSON report.

For two variants (with and without mid-training evaluation) compares:
fresh run A vs fresh run B (determinism) and A vs resume-from-step-1.
Rank sync is asserted inside training (assert_ranks_synchronized).
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import yaml
from tests.test_training import fixture_config


def diff(a, b):
    rows = []
    for k in a["model"]:
        d = (a["model"][k].double() - b["model"][k].double()).abs().max().item()
        if d > 0:
            rows.append((d, k))
    rows.sort(reverse=True)
    return {"tensors": len(a["model"]), "different": len(rows),
            "max_abs": rows[0][0] if rows else 0.0, "largest": [k for _, k in rows[:5]],
            "first_in_order": next((k for k in a["model"] if any(k == n for _, n in rows)), None),
            "steps_equal": a["state"]["step"] == b["state"]["step"],
            "scheduler_equal": a["scheduler"] == b["scheduler"]}


def variant(name, evaluate_midway):
    tmp = Path(tempfile.mkdtemp(prefix=f"ddp_{name}_"))
    cfg, path = fixture_config(tmp)
    cfg["training"].update(num_devices=2, batch_size=1, gradient_accumulation=2, max_epochs=2, max_steps=2)
    if not evaluate_midway:
        cfg["training"].update(eval_every=1000, evaluate_each_epoch=False)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("FONGBE_RESUME", "FONGBE_AUDIO_ROOT", "FONGBE_OUTPUT_ROOT", "WORLD_SIZE", "RANK", "LOCAL_RANK")}
    proc = subprocess.run([sys.executable, "-m", "tests.ddp_runner", str(path), str(tmp), "full", "again", "resume"],
                          capture_output=True, text=True, env=env)
    if proc.returncode:
        return {"error": proc.stderr[-5000:]}
    run = Path(cfg["paths"]["output_root"])/cfg["run_name"]
    load = lambda p: torch.load(p, weights_only=False, map_location="cpu")
    a, b, r = load(tmp/"full.pt"), load(tmp/"again.pt"), load(run/"checkpoints/last.pt")
    return {"fresh_vs_fresh": diff(a, b), "fresh_vs_resume": diff(a, r)}


def main():
    report = {"torch": torch.__version__,
              "with_midway_eval": variant("eval", True),
              "without_midway_eval": variant("noeval", False),
              "rank_sync": "asserted at step 1 and every save (raises on divergence)"}
    print("DDP_DIAGNOSE", json.dumps(report, indent=2))
    Path("ddp_diagnose.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
