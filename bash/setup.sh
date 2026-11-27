#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep Kaggle's matched CUDA torch/torchaudio rather than install CPU wheels.
python - <<'PY'
import torch, torchaudio
version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert version >= (2, 5), f"Requires torch >=2.5, found {torch.__version__}"
if version != (2, 8):
    print("NOTE: local validation used torch 2.8 CPU; checking this Kaggle version with smoke tests.")
assert torch.__version__.split("+")[0] == torchaudio.__version__.split("+")[0], "Mismatched torch/torchaudio"
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Select GPU T4 x2 in Kaggle settings"
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
for i in range(2):
    print("GPU", i, torch.cuda.get_device_name(i))
PY
python -m pip install --quiet -r requirements.txt
python -m pytest -q tests
python -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  -m src.smoke --config data/configs/training/proposal01.yaml
