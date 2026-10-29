#!/bin/bash
# Generate samples from a trained model with its EMA weights (500 per class, as in the paper),
# on every visible GPU.
#
# Usage: bash scripts/sample.sh configs/ddpm_config_256.yml runs/stage_256_<timestamp>/models/ema_best.pt
set -euo pipefail

CONFIG="$1"
CKPT="$2"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export RESULTS_DIR="${RESULTS_DIR:-runs}"
export RUN_NAME="${RUN_NAME:-samples}"

python ddpm.py --config "$CONFIG" --mode test --ckpt "$CKPT" --n_per_class 500 --sample_bs 5
