#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  -m src.train --config data/configs/training/proposal01.yaml
