# Fongbe ASR — Proposal 01

Zipformer-S **CTC**, trained from scratch on all provided label sources with equal
per-utterance loss weight. No external LM, relabeling or pseudo-labeling.

## Start on Kaggle

Upload **[notebook/proposal01_kaggle.ipynb](notebook/proposal01_kaggle.ipynb)**.
It contains exactly three code cells:
1. Clone the repository, checkout a revision, install dependencies and run smoke tests.
2. Download and verify **audio only** from the Hugging Face repo in the supplied starter.
3. Run `bash bash/run.sh`, then show output paths/charts.

Push this code and the saved split/tokenizer artifacts first. Enable Internet and
**GPU T4 x2**. Public datasets do **not** require `HF_TOKEN`; it is optional for
restricted sources. `GITHUB_TOKEN` is only needed if the GitHub repo is private.
Never put tokens in source, URLs or outputs.

Training and validation use **both T4 GPUs with DDP/NCCL**. It preserves Kaggle's matched
torch/torchaudio installation (requires >=2.5; locally tested 2.8 CPU).
Other versions are checked by smoke tests, not assumed tested.
If setup rejects the installed pair, install a matching CUDA-enabled pair compatible
with Kaggle's driver in the setup script before running; do not install CPU wheels
for GPU training. No k2, lhotse or Transformers dependency is required.

## Repository layout

```text
data/
  configs/{training,evaluation,submission}/proposal01.yaml
  splits/all_labels_valspk_jdg_v1/
  tokenizers/fon_char_nfd_v1/tokenizer.json
  Train_*.csv, Test.csv, official scoring utilities
src/
  model.py, dataset.py, tokenizer.py
  train.py, evaluate.py, infer.py, download.py, charts.py
  vendor/icefall/               # actual pinned Zipformer2 + ScaledAdam/Eden
bash/
  run.sh, experiments/proposal01.sh
  setup.sh, download.sh, evaluate.sh, submit.sh
notebook/proposal01_kaggle.ipynb
docs/                          # local only, ignored by Git
```

## Experiment configuration

Edit **data/configs/training/proposal01.yaml before pushing**. The notebook never
rewrites hyperparameters. Select the script in `bash/run.sh` for another experiment.
Use a new `run_name` for a changed experiment; never overwrite a completed run.

Current P01:
- Non-streaming Zipformer2 small encoder, **21,999,288 total parameters** with 37-token head.
- Character tokenizer: NFD letters + separate tone marks, delimiter/UNK/CTC blank.
  It is loaded from disk, not fitted on Kaggle or validation.
- 132,396 train utterances / ~335.36 h before audio preflight; 37 >30s rows excluded.
- Validation: 2,396 recommended unseen-speaker recordings + 1,194 JDG book recordings.
- Uniform sampling over utterances; uncertain has more samples, so sources are not 1:1:1.
- ScaledAdam lr 0.04, Eden; **4 samples/GPU × 2 GPUs × accumulation 4 = 32**, FP16, 30 epochs.
- `torchrun` starts one process per GPU. Gradients synchronize only on the last accumulation
  microbatch (`no_sync` otherwise). Exact global per-utterance normalization includes uneven tails.
- Deterministic disjoint shards cover every training example once without duplicate padding.
  Validation also uses disjoint shards on both GPUs; only rank zero writes artifacts/metrics.
- Dropout and SpecAugment; target-length-normalized CTC averaged equally per utterance.
- Save every 500 successful updates; evaluate every 2,000 and at epoch end.
- Primary metric: pooled corpus syllable-WER, uncapped. Also report each domain
  separately and official tonal diagnostics. The supplied score.py composite is
  **not** renamed to leaderboard WER.
- Session training pause after 9h (download/setup time is additional). For long initial
  downloads, lower `session_hours` in config before launching to fit your Kaggle limit.
- Hyperparameters are initial choices, not validated accuracy or convergence claims.

30 epochs may need many Kaggle sessions. A short pilot is not evidence of final
accuracy for training from scratch.

The vendored encoder is not a toy Transformer renamed Zipformer. See
[src/vendor/icefall/NOTICE.md](src/vendor/icefall/NOTICE.md) for commit, license and
native-PyTorch Swoosh adaptations. This wrapper is not the paper's exact recipe or
CR-CTC reproduction. Code license does not authorize redistribution of challenge data.

## Data and storage

The downloader uses `JaimeMaLangue-Zindi/JML-Hackathon-fon-asr-data` from the official
starter notebook; change the config if the organizers move it. It resolves and
records the HF revision, checks official SHA256SUMS, extracts only required files,
rejects archive traversal/links, and deletes each archive after extraction.

Budget roughly **40 GB for train/validation WAVs plus archive/headroom**. It checks
free disk and fails instead of silently using less data. Audio goes to
`/kaggle/temp/fongbe`, never into the persisted training Output.
Existing complete audio directories can be reused via `FONGBE_AUDIO_ROOT`.
Preflight checks every required WAV's header/duration and CTC feasibility.
CTC-infeasible training rows are excluded with a report; validation remains intact
for metrics and reports the count excluded only from loss. This is a technical
exception, not confidence filtering. Full audio-content deduplication is not yet done.

Split/tokenizer creation is a **local, one-time** operation:
`python -m src.prepare`. Already-created artifacts refuse overwrite.
To explore another split/tokenizer, use a new version and config, not in-place edits.

## Outputs and safe resume

```text
outputs/<run_name>/
  resolved_config.yaml, manifest.json, status.json, console.log
  tokenizer/tokenizer.json
  checkpoints/last.pt
  best/model.pt, best/tokenizer.json, best/config.json
  logs/train.jsonl, logs/eval.jsonl
  charts/loss.png, charts/metrics.png
  predictions/latest/{general.csv,religious.csv,metrics.json}
  reports/*_preflight.json
```

`last.pt` v2 is atomic and self-contained: optimizer/scheduler, **per-rank scaler/RNG**,
global batch cursor, log history and best inference weights. Only the last resume file and
best inference model are retained to limit Kaggle storage. **Save Version / persist
Output before ending the session**; a local save is NOT an upload. No automatic
GitHub push, release upload or competition submission happens.

In a new session, attach the previous Output under `/kaggle/input`, set
`RESUME_CHECKPOINT` in notebook Cell 1, and checkout the **same commit/config**.
Alternatively use config `resume.checkpoint` or `FONGBE_RESUME`.
Paths can change, but code, model, tokenizer, split and hyperparameter fingerprints
and world size must match. Old single-GPU v1 checkpoints are deliberately rejected by
this v2 run, not silently resumed with a changed batch/sampler. Scheduler/warmup are not restarted. Only load trusted own resume files:
full checkpoints use Python pickle. GPU bitwise reproducibility is not promised.

Weights for inference (~88 MB tensor storage, actual file size varies) are separate
from larger optimizer checkpoints. The competition limit concerns deployed weights,
not resume states. Use suitable large-file storage/releases for large checkpoints.
No weights have been trained/uploaded by this implementation.

## Evaluate / submit / measure CPU

After editing the corresponding job configs:

```bash
bash bash/evaluate.sh
bash bash/download.sh --include-test
bash bash/submit.sh
```

Submission writes `ID,Target` after `syllabify()` exactly once, and preserves plain
transcripts alongside it. Inference supports arbitrary test CSV order and verifies
unique IDs. It currently processes complete utterances, not a streaming/chunked
engine; long files can increase RAM/time.

Offline plain transcription for the official measurement:

```bash
python data/measure_frugality.py \
  --weights /path/to/best/model.pt \
  --command "python -m src.infer --checkpoint /path/to/best/model.pt --csv {csv} --root {root} --output {output} --device cpu --threads 1" \
  --csv data/splits/all_labels_valspk_jdg_v1/valid_general.csv \
  --root /path/to/audio_root --limit 300 --report frugality.json
```

Run from repo root on Linux with permissions needed for the network cut.
This checkpoint embeds its model/feature/tokenizer configuration. No online access
is made during inference. **RTF <=1 and peak RSS <=2 GB are not yet verified.**
ONNX/INT8 export is not implemented; PyTorch CPU is a starting measurement path,
not an assertion of prize eligibility.

## Authorized local Kaggle deployment / polling

Install `requirements-ops.txt` locally; put `KAGGLE_API_TOKEN` in ignored `.env`.
Never upload this file or token. After commit/push:

```bash
python -m ops.kaggle_run submit --mode smoke --slug fongbe-p01-2t4-smoke
python -m ops.kaggle_run detach --state .cache/kaggle/fongbe-p01-2t4-smoke/job.json --interval 120
```

Smoke uses private Kaggle T4 x2, runs setup/DDP tests, and does **not** download or train
competition data. `--mode train` first verifies dataset access locally before spending
GPU quota. Deployment pins the Git commit. Polling runs as a detached local process,
updates ignored `docs/progress.md`, downloads terminal outputs to `docs/kaggle_runs/`,
and records a results entry. The PC must remain awake; it is not a cloud scheduler
and does not automatically launch another GPU run or send chat notifications.

**Current data blocker:** both dataset IDs from the supplied starter/README return
401 anonymously. Get the current competition audio URL from Zindi Chat. Do not use
the separate public JML corpus as a replacement.

## Local checks

Install a matched torch/torchaudio pair, then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python data/test_score.py
python data/syllabify.py
```

Tests cover real Zipformer-S forward/backward/optimizer step, tokenizer, split
integrity, metrics, archive safety, small synthetic-WAV training, exact CPU resume,
checkpoint rejection on config changes, charts and inference/submission.
Generated test audio is **only a software fixture**, not a synthetic training corpus.
The CUDA/FP16 test runs on Kaggle and is skipped locally without a GPU.
Setup additionally uses torchrun to perform two DDP optimizer updates on the
**full Small model on both GPUs**, with accumulation and 30-second synthetic features,
before downloading the corpus. On Windows builds without a usable Gloo device, the
2-process CPU integration test skips; it runs on Kaggle/Linux. If this
OOMs, lower batch size and increase accumulation in the repo config before retrying;
do not silently change the experiment in the notebook.
