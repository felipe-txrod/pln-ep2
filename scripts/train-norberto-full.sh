#!/usr/bin/env bash
# Reuse the original EP1 training implementation with all original training rows.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ep1_root="${NORBERTO_EP1_ROOT:-$(dirname -- "$project_root")/EP1}"
image="${NORBERTO_IMAGE:-pln-rocm:6.4.1}"
if [[ ! -f "$ep1_root/src/clarity/finetune.py" || ! -d "$ep1_root/.cache/huggingface" ]]; then
    printf '%s\n' 'ERROR: The EP1 source tree and its pinned Hugging Face cache are required.' >&2
    exit 1
fi
exec docker run --rm --pull=never --network=none \
    --user "$(id -u):$(id -g)" --device=/dev/kfd --device=/dev/dri \
    --security-opt label=disable --security-opt seccomp=unconfined \
    -v "$project_root:/workspace" -v "$ep1_root:/ep1:ro" \
    -v "$ep1_root/.cache/huggingface:/workspace/.cache/huggingface:ro" \
    -w /workspace -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e HF_HUB_OFFLINE=1 \
    -e HF_HUB_DISABLE_TELEMETRY=1 -e TOKENIZERS_PARALLELISM=false \
    --entrypoint python3 "$image" -u scripts/train-norberto-full.py --ep1-root /ep1 "$@"
