"""Full validation; optional disjoint rank shards, exact corpus metrics on rank zero."""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from src.common import atomic_json, load_config, project_path, read_csv, write_csv
from src.dataset import SpeechDataset, make_loader
from src.metrics import score_pairs


@torch.inference_mode()
def evaluate(model, tokenizer, cfg, root, device, output_dir, distributed=None):
    model.eval()  # Must be the unwrapped module: ranks may have unequal eval lengths.
    output_dir = Path(output_dir)
    rank = distributed.rank if distributed else 0
    world_size = distributed.world_size if distributed else 1
    def on_primary(fn):
        return distributed.primary_call(fn) if distributed else fn()
    results, all_pairs, all_loss, all_count = {}, [], 0.0, 0
    for name in ("general", "religious"):
        rows = read_csv(project_path(cfg["data"][f"valid_{name}_csv"]))
        dataset = SpeechDataset(rows[rank::world_size], root, tokenizer, cfg["features"])
        size = cfg["evaluation"]["batch_size"]
        batches = [list(range(i, min(i+size, len(dataset)))) for i in range(0, len(dataset), size)]
        loader = make_loader(dataset, batches, cfg["evaluation"]["num_workers"], 0)
        predictions, loss_sum, loss_count = [], 0.0, 0
        for batch in tqdm(loader, desc=f"Validation {name}", leave=False, disable=rank != 0):
            probs, lengths = model(batch["features"].to(device, non_blocking=True),
                                   batch["feature_lengths"].to(device, non_blocking=True))
            raw_losses = torch.nn.functional.ctc_loss(
                probs.transpose(0, 1).contiguous(), batch["labels"], lengths.cpu(),
                batch["label_lengths"], blank=tokenizer.blank_id, reduction="none", zero_infinity=False,
            ) / batch["label_lengths"].to(device).clamp_min(1)
            feasible = (lengths.cpu() >= batch["minimum_lengths"]).to(device)
            if not torch.isfinite(raw_losses[feasible]).all():
                raise FloatingPointError("Nonfinite eval loss on feasible samples")
            loss_sum += raw_losses[feasible].sum().item()
            loss_count += int(feasible.sum())
            ids = probs.argmax(-1).cpu()
            for i, uid in enumerate(batch["ids"]):
                predictions.append({"id": uid, "transcription": tokenizer.decode(ids[i, :int(lengths[i])].tolist())})
        shards = distributed.gather((predictions, loss_sum, loss_count)) if distributed else [(predictions, loss_sum, loss_count)]
        def summarize():
            by_id = {r["id"]: r for pred, _, _ in shards for r in pred}
            if len(by_id) != len(rows) or sum(len(p) for p, _, _ in shards) != len(rows):
                raise ValueError("Validation shards missing/duplicating IDs")
            ordered = [by_id[r["id"]] for r in rows]
            pairs = [(r["transcription"], h["transcription"]) for r, h in zip(rows, ordered)]
            total_loss = sum(s for _, s, _ in shards)
            count = sum(n for _, _, n in shards)
            scores = {**score_pairs(pairs), "loss": total_loss/count if count else None,
                      "loss_samples": count, "ctc_infeasible_samples": len(rows)-count}
            write_csv(output_dir / f"{name}.csv", ordered, ["id", "transcription"])
            return scores, pairs, total_loss, count
        scores, pairs, total_loss, count = on_primary(summarize)
        results[name] = scores
        all_pairs.extend(pairs)
        all_loss += total_loss
        all_count += count
    def finish():
        results["all"] = {**score_pairs(all_pairs), "loss": all_loss/all_count if all_count else None,
                          "loss_samples": all_count}
        atomic_json(output_dir / "metrics.json", results)
        return results
    return on_primary(finish)


if __name__ == "__main__":
    from src.checkpoint import load_inference
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/evaluation/proposal01.yaml")
    with open(project_path(p.parse_args().config), encoding="utf-8") as f:
        job = yaml.safe_load(f)
    cfg = load_config(job["training_config"])
    device = torch.device(job["device"])
    model, tokenizer, features = load_inference(job["checkpoint"], device)
    if cfg["features"] != features:
        raise ValueError("Evaluation feature config differs from checkpoint")
    print(evaluate(model, tokenizer, cfg, Path(job["audio_root"]), device, job["output"]))
