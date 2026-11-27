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
assert torch.cuda.is_available(), "Enable a GPU in Kaggle Notebook settings"
print("torch", torch.__version__, "torchaudio", torchaudio.__version__, "GPU", torch.cuda.get_device_name(0))
PY
python -m pip install --quiet -r requirements.txt
python -m pytest -q tests
python -m src.smoke --config data/configs/training/proposal01.yaml
