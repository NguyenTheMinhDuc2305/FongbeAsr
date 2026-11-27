"""Quantify DDP determinism/resume drift on Linux CPU (Gloo). Prints a JSON report.

Compares: fresh run A vs fresh run B (determinism), A vs resume-from-step-1.
Rank-sync is asserted inside training (assert_ranks_synchronized).
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import yaml
from tests.test_training import fixture_config


def diff(a, b):
    out = {"tensors": 0, "different": 0, "max_abs": 0.0, "first": None}
    for k in a["model"]:
        out["tensors"] += 1
        d = (a["model"][k].double() - b["model"][k].double()).abs().max().item()
        if d > 0:
            out["different"] += 1
            out["first"] = out["first"] or k
            out["max_abs"] = max(out["max_abs"], d)
    out["optimizer_steps_equal"] = a["state"]["step"] == b["state"]["step"]
    out["scheduler_equal"] = a["scheduler"] == b["scheduler"]
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ddp_diag_"))
    cfg, path = fixture_config(tmp)
    cfg["training"].update(num_devices=2, batch_size=1, gradient_accumulation=2, max_epochs=2, max_steps=2)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "tests.ddp_runner", str(path), str(tmp), "full", "again", "resume"],
                          capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout[-4000:], proc.stderr[-8000:])
        raise SystemExit(proc.returncode)
    run = Path(cfg["paths"]["output_root"])/cfg["run_name"]
    a = torch.load(tmp/"full.pt", weights_only=False, map_location="cpu")
    b = torch.load(tmp/"again.pt", weights_only=False, map_location="cpu")
    r = torch.load(run/"checkpoints/last.pt", weights_only=False, map_location="cpu")
    report = {"torch": torch.__version__, "fresh_vs_fresh": diff(a, b), "fresh_vs_resume": diff(a, r),
              "rank_sync": "asserted at step 1 and every save"}
    print("DDP_DIAGNOSE", json.dumps(report, indent=2))
    Path("ddp_diagnose.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
