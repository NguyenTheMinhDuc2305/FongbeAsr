"""End-to-end smoke on generated WAVs; no competition audio/network/GPU required."""
import copy
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import yaml
from src import train as trainer
from src.common import atomic_json, read_csv, sha256, write_csv
from src.checkpoint import load_inference
from src.infer import infer
from src.tokenizer import CharacterTokenizer
from tests.test_model import small_config


def fixture_config(tmp_path):
    cfg = small_config()
    root = tmp_path / "audio_root"
    root.mkdir()
    source_tok = CharacterTokenizer(Path("data/tokenizers/fon_char_nfd_v1"))
    tok_dir = tmp_path / "tokenizer"
    rows = []
    for i in range(8):
        name = f"sample{i}.wav"
        rng = np.random.default_rng(i)
        wave = (rng.standard_normal(16000) * 0.03).astype(np.float32)
        sf.write(root / name, wave, 16000, subtype="PCM_16")
        rows.append({"id": str(i), "audio_path": name, "duration": 1.0,
                     "transcription": "nɔ́ ɖu" if i % 2 else "a a"})
    keys = ["id", "audio_path", "duration", "transcription"]
    splits = {"train": rows[:4], "valid_general": rows[4:6], "valid_religious": rows[6:]}
    summary = {}
    for name, values in splits.items():
        p = tmp_path / f"{name}.csv"
        write_csv(p, values, keys)
        cfg["data"][name+"_csv"] = str(p)
        summary[name] = {"sha256": sha256(p)}
    tc = copy.deepcopy(source_tok.config)
    tc["fit_split_sha256"] = summary["train"]["sha256"]
    atomic_json(tok_dir / "tokenizer.json", tc)
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"summary": summary, "tokenizer_sha256": sha256(tok_dir / "tokenizer.json")})
    cfg["tokenizer"] = str(tok_dir)
    cfg["data"]["manifest"] = str(manifest)
    cfg["paths"] = {"audio_root": str(root), "output_root": str(tmp_path / "outputs")}
    cfg["run_name"] = "smoke"
    cfg["data"]["max_train_seconds"] = 1.0
    cfg["training"].update(device="cpu", num_devices=1, precision="fp32", max_duration_per_gpu=2.0,
                           gradient_accumulation=1, num_workers=0, max_epochs=1,
                           max_steps=2, log_every=1, save_every=1, eval_every=1,
                           session_hours=1.0, cpu_threads=2)
    cfg["evaluation"].update(num_workers=0, batch_size=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, path


def test_train_resume_exact_and_infer(tmp_path, monkeypatch):
    monkeypatch.delenv("FONGBE_AUDIO_ROOT", raising=False)
    monkeypatch.delenv("FONGBE_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("FONGBE_RESUME", raising=False)
    cfg, path = fixture_config(tmp_path)
    saved_step1 = tmp_path / "step1.pt"
    real_save = trainer.atomic_torch_save
    def save_with_intermediate(path, value):
        real_save(path, value)
        if value.get("state", {}).get("step") == 1:
            real_save(saved_step1, value)
    monkeypatch.setattr(trainer, "atomic_torch_save", save_with_intermediate)
    trainer.train(path)
    run = Path(cfg["paths"]["output_root"]) / cfg["run_name"]
    full = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert full["state"]["step"] == 2
    assert (run / "charts/loss.png").is_file()
    assert (run / "charts/metrics.png").is_file()
    assert json.loads((run / "status.json").read_text())["status"] == "completed"
    monkeypatch.setenv("FONGBE_RESUME", str(saved_step1))
    trainer.train(path)
    resumed = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert full["scheduler"] == resumed["scheduler"]
    model, tok, features = load_inference(run / "best/model.pt", torch.device("cpu"))
    assert len(tok) == 37
    output = tmp_path / "submission.csv"
    infer(run / "best/model.pt", cfg["data"]["valid_general_csv"], cfg["paths"]["audio_root"],
          output, syllabified=True)
    sub = read_csv(output)
    assert [r["ID"] for r in sub] == ["4", "5"]
    assert output.with_suffix(".plain.csv").is_file()


def test_download_source_is_not_experiment_identity():
    import copy
    from src.common import identity, load_config
    cfg = load_config("data/configs/training/proposal01.yaml")
    moved = copy.deepcopy(cfg)
    moved["download"]["repo_id"] = "organizers/moved-repo"
    assert identity(cfg)["sha256"] == identity(moved)["sha256"]
    changed = copy.deepcopy(cfg)
    changed["training"]["optimizer"]["lr"] = 0.01
    assert identity(cfg)["sha256"] != identity(changed)["sha256"]


def test_resume_rejects_changed_hyperparameters(tmp_path, monkeypatch):
    monkeypatch.delenv("FONGBE_AUDIO_ROOT", raising=False)
    monkeypatch.delenv("FONGBE_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("FONGBE_RESUME", raising=False)
    cfg, path = fixture_config(tmp_path)
    cfg["training"].update(max_steps=1)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    trainer.train(path)
    run = Path(cfg["paths"]["output_root"]) / cfg["run_name"]
    monkeypatch.setenv("FONGBE_RESUME", str(run / "checkpoints/last.pt"))
    cfg["training"]["optimizer"]["lr"] = 0.001
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="fingerprint differs"):
        trainer.train(path)
