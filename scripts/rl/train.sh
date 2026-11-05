#!/usr/bin/env bash
# Fit the fixed logistic recipe on development while excluding the EP1 holdout.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ep1_root="${RL_EP1_ROOT:-$(dirname -- "$project_root")/EP1}"
exec docker run --rm --pull=never --network=none --cpus=2 \
    --user "$(id -u):$(id -g)" \
    -v "$project_root:/workspace" -v "$ep1_root:/ep1:ro" -w /workspace \
    -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    --entrypoint python3 "${RL_IMAGE:-pln-rocm:6.4.1}" \
    -u scripts/rl/train-development.py --ep1-root /ep1 "$@"
