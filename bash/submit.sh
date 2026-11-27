#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Download test audio first: bash bash/download.sh --include-test
python -m src.infer --config data/configs/submission/proposal01.yaml
