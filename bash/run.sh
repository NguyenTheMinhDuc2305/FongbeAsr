#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Edit/select the experiment here BEFORE launching the Kaggle notebook.
bash bash/experiments/proposal01.sh
