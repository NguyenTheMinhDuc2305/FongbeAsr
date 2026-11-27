"""Full held-out validation with per-domain and pooled metrics."""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from src.common import atomic_json, load_config, project_path, read_csv, write_csv
from src.dataset import SpeechDataset, make_loader
from src.metrics import score_pairs


@torch.inference_mode()
def evaluate(model, tokenizer, cfg, root, device, output_dir):
    model.eval()
    output_dir = Path(output_dir)
    results, all_pairs, all_loss, all_count = {}, [], 0.0, 0
    for name in ("general", "religious"):
        rows = read_csv(project_path(cfg["data"][f"valid_{name}_csv"]))
        dataset = SpeechDataset(rows, root, tokenizer, cfg["features"])
        size = cfg["evaluation"]["batch_size"]
        batches = [list(range(i, min(i+size, len(rows)))) for i in range(0, len(rows), size)]
        loader = make_loader(dataset, batches, cfg["evaluation"]["num_workers"], 0)
        predictions, pairs, loss_sum, loss_count = [], [], 0.0, 0
        for batch in tqdm(loader, desc=f"Validation {name}", leave=False):
            # Evaluate FP32 for stable comparisons across sessions and CPU.
            probs, lengths = model(batch["features"].to(device), batch["feature_lengths"].to(device))
            raw_losses = torch.nn.functional.ctc_loss(
                probs.transpose(0, 1).contiguous(), batch["labels"], lengths.cpu(),
                batch["label_lengths"], blank=tokenizer.blank_id, reduction="none",
                zero_infinity=False,
            ) / batch["label_lengths"].to(device).clamp_min(1)
            feasible = (lengths.cpu() >= batch["minimum_lengths"]).to(device)
            if not torch.isfinite(raw_losses[feasible]).all():
                raise FloatingPointError("Nonfinite evaluation loss on CTC-feasible samples")
            loss_sum += raw_losses[feasible].sum().item()
            loss_count += int(feasible.sum())
            ids = probs.argmax(-1).cpu()
            for i, (uid, ref) in enumerate(zip(batch["ids"], batch["references"])):
                text = tokenizer.decode(ids[i, :int(lengths[i])].tolist())
                pairs.append((ref, text))
                predictions.append({"id": uid, "transcription": text})
        scores = score_pairs(pairs)
        results[name] = {**scores, "loss": loss_sum / loss_count if loss_count else None,
                         "loss_samples": loss_count, "ctc_infeasible_samples": len(rows)-loss_count}
        all_pairs.extend(pairs)
        all_loss += loss_sum
        all_count += loss_count
        write_csv(output_dir / f"{name}.csv", predictions, ["id", "transcription"])
    results["all"] = {**score_pairs(all_pairs), "loss": all_loss / all_count if all_count else None,
                       "loss_samples": all_count}
    atomic_json(output_dir / "metrics.json", results)
    return results


if __name__ == "__main__":
    from src.checkpoint import load_inference
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="data/configs/evaluation/proposal01.yaml")
    a = p.parse_args()
    with open(project_path(a.config), encoding="utf-8") as f:
        job = yaml.safe_load(f)
    cfg = load_config(job["training_config"])
    device = torch.device(job["device"])
    model, tokenizer, features = load_inference(job["checkpoint"], device)
    if cfg["features"] != features:
        raise ValueError("Evaluation feature config differs from checkpoint")
    print(evaluate(model, tokenizer, cfg, Path(job["audio_root"]), device, job["output"]))
