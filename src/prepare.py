"""Run LOCALLY once to create versioned splits/tokenizer. Not called by Kaggle."""
from collections import Counter
from pathlib import Path
import argparse
from data.normalize import tonal_target
from src.common import ROOT, atomic_json, read_csv, sha256, write_csv


def prepare(output=None):
    base = Path(output) if output else ROOT / "data"
    split = base / "splits/all_labels_valspk_jdg_v1"
    tok = base / "tokenizers/fon_char_nfd_v1"
    if (split / "manifest.json").exists() or (tok / "tokenizer.json").exists():
        raise FileExistsError("Artifacts already exist: use a new version/output directory, do not overwrite")
    groups = {"train": [], "valid_general": [], "valid_religious": [], "excluded": []}
    source_hashes = {}
    valid_speakers = set()
    inputs = {}
    for source in ("reliable", "uncertain", "religious"):
        p = ROOT / f"data/Train_{source}.csv"
        source_hashes[p.name] = sha256(p)
        inputs[source] = read_csv(p)
    valid_speakers = {r["speaker_id"] for r in inputs["reliable"] if r["recommended_validation"] == "1"}
    for source, rows in inputs.items():
        for raw in rows:
            row = dict(raw, source=source, target=tonal_target(raw["transcription"]), exclusion_reason="")
            if source == "reliable" and row["recommended_validation"] == "1":
                name = "valid_general"
            elif source == "religious" and row["book"] == "JDG":
                name = "valid_religious"
            elif row["speaker_id"] in valid_speakers:
                raise ValueError("Validation speaker appears in another source")
            elif not row["target"] or not 0 < float(row["duration"]) <= 30:
                name = "excluded"
                row["exclusion_reason"] = "empty_target" if not row["target"] else "duration"
            else:
                name = "train"
            groups[name].append(row)
    ids = [r["id"] for rows in groups.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ID across source files")
    fields = ["id", "audio_path", "transcription", "target", "speaker_id", "duration", "source",
              "recommended_validation", "protocol", "book", "chapter", "exclusion_reason"]
    for name, rows in groups.items():
        write_csv(split / f"{name}.csv", sorted(rows, key=lambda r: r["id"]), fields)
    chars = sorted(set("".join(r["target"] for r in groups["train"])) - {" "})
    atomic_json(tok / "tokenizer.json", {
        "name": "fon_char_nfd_v1", "type": "character_ctc",
        "normalization": "tonal_target_digits_false_nfd",
        "tokens": chars + ["|", "[UNK]", "[PAD]"],
        "fit_split_sha256": sha256(split / "train.csv"),
        "normalizer_sha256": sha256(ROOT / "data/normalize.py"),
        "bos_token": None, "eos_token": None,
    })
    summary = {}
    for name, rows in groups.items():
        summary[name] = {"utterances": len(rows), "hours": sum(float(r["duration"]) for r in rows) / 3600,
                         "sources": dict(Counter(r["source"] for r in rows)),
                         "sha256": sha256(split / f"{name}.csv")}
    atomic_json(split / "manifest.json", {
        "split_id": "all_labels_valspk_jdg_v1",
        "source_sha256": source_hashes, "validation_religious_books": ["JDG"],
        "validation_speakers": sorted(valid_speakers),
        "max_train_seconds": 30, "summary": summary,
        "tokenizer_sha256": sha256(tok / "tokenizer.json"),
    })
    print(summary)
    print("Vocabulary:", len(chars) + 3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", help="Alternative data output directory (must be new)")
    prepare(p.parse_args().output)
