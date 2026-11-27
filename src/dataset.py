"""WAV -> deterministic Kaldi-compatible 80-bin fbank; train augmentation stays in main process."""
import math
import random
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.common import atomic_json, json_hash


def output_frames(n_samples, feature_cfg):
    sr = feature_cfg["sample_rate"]
    window = round(feature_cfg["frame_length"] * sr / 1000)
    shift = round(feature_cfg["frame_shift"] * sr / 1000)
    frames = max(0, (n_samples - window) // shift + 1)
    return max(0, (((frames - 7) // 2) + 1) // 2)


def preflight(rows, root, tokenizer, features, training, report):
    kept, rejected = [], []
    seen = set()
    for row in tqdm(rows, desc="Audio preflight", leave=False):
        if row["id"] in seen:
            raise ValueError(f"Duplicate ID: {row['id']}")
        seen.add(row["id"])
        p = (root / row["audio_path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe path: {p}")
        info = sf.info(p)  # missing/corrupt is an error, never silent subset training
        if info.samplerate != features["sample_rate"] or info.channels != 1 or info.frames <= 0:
            raise ValueError(f"Expected nonempty mono 16kHz: {p}, {info}")
        if abs(info.duration - float(row["duration"])) > 0.1:
            raise ValueError(f"Duration metadata differs from audio: {p}")
        labels = tokenizer.encode(row["transcription"])
        if not labels:
            raise ValueError(f"Empty target: {row['id']}")
        feasible = output_frames(info.frames, features) >= tokenizer.minimum_ctc_frames(labels)
        if training and not feasible:
            rejected.append({"id": row["id"], "reason": "ctc_infeasible"})
        else:
            kept.append(row)
    if not kept:
        raise ValueError("No usable rows")
    atomic_json(report, {"input": len(rows), "kept": len(kept), "rejected": rejected,
                         "kept_ids_sha256": json_hash([r["id"] for r in kept]),
                         "checks": "headers/duration/paths/CTC; no full audio-content deduplication"})
    return kept


def load_features(path, cfg):
    wave, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != cfg["sample_rate"] or wave.shape[1] != 1:
        raise ValueError(f"Expected mono {cfg['sample_rate']}Hz audio: {path}")
    x = torch.from_numpy(wave[:, 0].copy()).unsqueeze(0) * 32768.0
    if not torch.isfinite(x).all() or x.shape[1] < int(sr * cfg["frame_length"] / 1000):
        raise ValueError(f"Invalid or too short waveform: {path}")
    return torchaudio.compliance.kaldi.fbank(
        x, sample_frequency=sr, num_mel_bins=cfg["num_mel_bins"],
        frame_length=cfg["frame_length"], frame_shift=cfg["frame_shift"],
        dither=cfg["dither"], snip_edges=cfg["snip_edges"],
        high_freq=cfg["high_freq"], energy_floor=0.0,
    )


class SpeechDataset(Dataset):
    def __init__(self, rows, root, tokenizer, features):
        self.rows, self.root, self.tokenizer, self.features = rows, Path(root), tokenizer, features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        labels = self.tokenizer.encode(row["transcription"]) if "transcription" in row else []
        return {"id": row["id"], "reference": row.get("transcription", ""),
                "features": load_features(self.root / row["audio_path"], self.features),
                "labels": torch.tensor(labels, dtype=torch.long)}


def collate(items):
    feats = [x["features"] for x in items]
    labels = [x["labels"] for x in items]
    return {
        "ids": [x["id"] for x in items],
        "references": [x["reference"] for x in items],
        "features": pad_sequence(feats, batch_first=True, padding_value=math.log(1e-10)),
        "feature_lengths": torch.tensor([len(x) for x in feats], dtype=torch.long),
        "labels": torch.cat(labels),
        "label_lengths": torch.tensor([len(x) for x in labels], dtype=torch.long),
        "minimum_lengths": torch.tensor([len(x) + int((x[1:] == x[:-1]).sum()) for x in labels]),
    }


def epoch_batches(rows, batch_size, bucket_size, seed, epoch):
    if batch_size < 1 or bucket_size < batch_size:
        raise ValueError("Invalid batch/bucket size")
    rng = random.Random(seed + epoch)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), bucket_size):
        bucket = sorted(indices[start:start + bucket_size], key=lambda i: float(rows[i]["duration"]))
        batches.extend(bucket[i:i + batch_size] for i in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches


def distributed_batches(rows, batch_size, bucket_size, seed, epoch, world_size):
    """Global microbatches; every row once, no DistributedSampler duplicate padding."""
    if len(rows) < world_size or bucket_size % (batch_size * world_size):
        raise ValueError("Need at least world_size rows; bucket_size must be divisible by global batch")
    batches = epoch_batches(rows, batch_size * world_size, bucket_size, seed, epoch)
    # Only the last partial bucket can produce a batch smaller than world_size.
    for batch in batches:
        while len(batch) < world_size:
            donor = max(batches, key=len)
            if len(donor) <= world_size:
                raise ValueError("Not enough samples to rebalance distributed tail")
            batch.append(donor.pop())
    return batches


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(dataset, batches, workers, seed):
    # Separate generator prevents worker creation consuming model's RNG at resume.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_sampler=batches, collate_fn=collate,
                      num_workers=workers, worker_init_fn=worker_init,
                      generator=generator, pin_memory=torch.cuda.is_available(),
                      **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}))


def specaugment(x, lengths, cfg):
    x = x.clone()
    for b, length in enumerate(lengths.tolist()):
        for _ in range(cfg["frequency_masks"]):
            width = int(torch.randint(min(cfg["frequency_width"], x.size(2)) + 1, ()).item())
            start = int(torch.randint(x.size(2) - width + 1, ()).item())
            x[b, :length, start:start + width] = 0
        max_width = min(cfg["time_width"], int(length * cfg["max_time_fraction"]))
        for _ in range(cfg["time_masks"]):
            width = int(torch.randint(max_width + 1, ()).item())
            start = int(torch.randint(length - width + 1, ()).item())
            x[b, start:start + width] = 0
    return x
