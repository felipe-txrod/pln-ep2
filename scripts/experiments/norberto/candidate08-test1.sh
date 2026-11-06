#!/usr/bin/env bash
# Evaluate the saved grid-screening adapter; no training or default promotion.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
exec docker run --rm --pull=never --network=none --cpus=2 \
    --user "$(id -u):$(id -g)" --device=/dev/kfd --device=/dev/dri \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    --security-opt label=disable --security-opt seccomp=unconfined \
    -v "$project_root:/workspace" -v "$(dirname "$project_root")/EP1:/ep1:ro" \
    -w /workspace -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
    -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
    --entrypoint python3 "${NORBERTO_IMAGE:-pln-rocm:6.4.1}" \
    -u scripts/experiments/norberto/candidate08-test1.py --ep1-root /ep1 "$@"
