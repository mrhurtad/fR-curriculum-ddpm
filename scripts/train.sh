#!/bin/bash
# Train the three curriculum stages in order (64^2 -> 128^2 -> 256^2) on one node, using every visible GPU
# (select them with CUDA_VISIBLE_DEVICES).
# Before stages 128 and 256, set train_params.pretrained_path in the config to the
# models/latest.pt of the previous stage (run folders are runs/<run_name>_<timestamp>/).
#
# Usage: bash scripts/train.sh configs/ddpm_config_64.yml
set -euo pipefail

CONFIG="$1"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
export RESULTS_DIR="${RESULTS_DIR:-runs}"

python ddpm.py --config "$CONFIG" --mode train
