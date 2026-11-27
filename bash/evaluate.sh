#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.evaluate --config data/configs/evaluation/proposal01.yaml
