#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.download --config data/configs/training/proposal01.yaml "$@"
