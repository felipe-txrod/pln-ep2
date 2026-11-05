#!/usr/bin/env bash
# Predict with the selected trained adapter, then score saved predictions on CPU.
set -euo pipefail
stage="${1:-}"
if [[ "$stage" != predict && "$stage" != compare ]]; then
    printf '%s\n' 'Usage: bash scripts/norberto/test1.sh predict|compare [arguments]' >&2
    exit 2
fi
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ep1_root="${NORBERTO_EP1_ROOT:-$(dirname -- "$project_root")/EP1}"
image="${NORBERTO_IMAGE:-pln-rocm:6.4.1}"
devices=()
if [[ "$stage" == predict ]]; then
    devices=(--device=/dev/kfd --device=/dev/dri -e HSA_OVERRIDE_GFX_VERSION=10.3.0)
fi
exec docker run --rm --pull=never --network=none \
    --user "$(id -u):$(id -g)" "${devices[@]}" \
    --security-opt label=disable --security-opt seccomp=unconfined \
    -v "$project_root:/workspace" -v "$ep1_root:/ep1:ro" \
    -v "$ep1_root/.cache/huggingface:/workspace/.cache/huggingface:ro" \
    -w /workspace -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
    --entrypoint python3 "$image" -u scripts/norberto/evaluate-test1.py \
    --stage "$stage" --ep1-root /ep1 "$@"
