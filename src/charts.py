"""Create real charts from JSONL logs; never invent missing measurements."""
import argparse
import json
from pathlib import Path


def plot(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    def records(name):
        p = run / "logs" / name
        return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()] if p.exists() else []
    train, valid = records("train.jsonl"), records("eval.jsonl")
    charts = run / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    if train or valid:
        fig, ax = plt.subplots()
        if train:
            ax.plot([r["step"] for r in train], [r["train_loss"] for r in train], label="train (logged window mean)")
        if valid:
            ax.plot([r["step"] for r in valid], [r["all_loss"] for r in valid], "o-", label="validation")
        ax.set(xlabel="Optimizer step", ylabel="Target-normalized CTC loss", title=run.name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "loss.png", dpi=140)
        plt.close(fig)
    if valid:
        fig, ax = plt.subplots()
        for name in ("general", "religious", "all"):
            ax.plot([r["step"] for r in valid], [100*r[name+"_syllable_wer"] for r in valid], "o-", label=name)
        ax.set(xlabel="Optimizer step", ylabel="Syllable-WER (%)", title=run.name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / "metrics.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    plot(p.parse_args().run)
