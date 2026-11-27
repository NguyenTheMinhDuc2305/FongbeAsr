#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# Variable-length batches fragment the caching allocator (probe: 6.8 GB allocated vs 10.1 GB reserved).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 \
  -m src.train --config data/configs/training/proposal01.yaml
