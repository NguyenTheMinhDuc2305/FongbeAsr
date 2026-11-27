"""Atomic checkpoints. Full resume files must only be loaded from trusted runs."""
import os
from pathlib import Path
import random
import numpy as np
import torch
from src.common import atomic_json


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(value, tmp)
    os.replace(tmp, path)


def capture_rng(device=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device is not None and device.type == "cuda" else None}


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and device is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def inference_payload(model, tokenizer, cfg, step, fingerprint):
    return {"format": "fongbe_zipformer_ctc_v1", "model_config": cfg["model"],
            "feature_config": cfg["features"], "tokenizer": tokenizer.config,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "step": step, "identity": fingerprint}


def export_best(run, model, tokenizer, cfg, step, fingerprint):
    path = Path(run) / "best"
    atomic_torch_save(path / "model.pt", inference_payload(model, tokenizer, cfg, step, fingerprint))
    atomic_json(path / "tokenizer.json", tokenizer.config)
    atomic_json(path / "config.json", {"model": cfg["model"], "features": cfg["features"], "step": step})


def load_inference(path, device):
    from src.model import ZipformerCTC
    from src.tokenizer import CharacterTokenizer
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "fongbe_zipformer_ctc_v1":
        raise ValueError("Not an exported Fongbe CTC model")
    tokenizer = CharacterTokenizer(config=payload["tokenizer"])
    model = ZipformerCTC(payload["model_config"], len(tokenizer), payload["feature_config"]["num_mel_bins"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), tokenizer, payload["feature_config"]
