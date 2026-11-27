"""Offline inference. Plain transcripts for frugality; syllabified CSV for Zindi."""
import argparse
from pathlib import Path
import time
import torch
import yaml
from tqdm import tqdm
from data.syllabify import syllabify
from src.checkpoint import load_inference
from src.common import atomic_json, project_path, read_csv, write_csv
from src.dataset import load_features


@torch.inference_mode()
def infer(checkpoint, csv_path, root, output, device="cpu", syllabified=False, threads=1):
    torch.set_num_threads(threads)
    start = time.perf_counter()
    model, tokenizer, features = load_inference(checkpoint, torch.device(device))
    rows = read_csv(csv_path)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate inference IDs")
    results = []
    for row in tqdm(rows, desc="Transcribing"):
        path = (Path(root) / row["audio_path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError("Audio path escapes root")
        x = load_features(path, features).unsqueeze(0).to(device)
        probs, lengths = model(x, torch.tensor([x.size(1)], device=device))
        plain = tokenizer.decode(probs[0, :int(lengths[0])].argmax(-1).cpu().tolist())
        results.append({"id": row["id"], "transcription": plain})
    output = Path(output)
    if syllabified:
        write_csv(output.with_suffix(".plain.csv"), results, ["id", "transcription"])
        write_csv(output, [{"ID": r["id"], "Target": syllabify(r["transcription"])} for r in results],
                  ["ID", "Target"])
    else:
        write_csv(output, results, ["id", "transcription"])
    atomic_json(output.with_suffix(".report.json"), {
        "utterances": len(rows), "empty": sum(not r["transcription"] for r in results),
        "elapsed_with_model_load_s": time.perf_counter()-start,
        "device": device, "threads": threads, "syllabified": syllabified,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--csv")
    p.add_argument("--root")
    p.add_argument("--output")
    p.add_argument("--device")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--syllabify", action="store_true")
    a = p.parse_args()
    job = {}
    if a.config:
        with open(project_path(a.config), encoding="utf-8") as f:
            job = yaml.safe_load(f)
    for key in ("checkpoint", "csv", "root", "output", "device"):
        if getattr(a, key) is not None:
            job[key] = getattr(a, key)
    missing = [k for k in ("checkpoint", "csv", "root", "output") if not job.get(k)]
    if missing:
        p.error(f"Missing arguments: {missing}")
    infer(job["checkpoint"], project_path(job["csv"]), job["root"], job["output"],
          job.get("device", "cpu"), a.syllabify or job.get("syllabify", False), a.threads)
