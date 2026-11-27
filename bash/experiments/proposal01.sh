#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
python -m src.train --config data/configs/training/proposal01.yaml
