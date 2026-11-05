#!/usr/bin/env bash
# Run the classifier with a mounted workspace and explicit GPU architecture policy.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime="${CONTAINER_RUNTIME:-docker}"
image="${QWEN_IMAGE:-ep2-qwen:rocm6.4}"
environment_args=()
if [[ -n "${QWEN_RUNTIME_PATH:-}" ]]; then
    environment_args+=(-e "PYTHONPATH=${QWEN_RUNTIME_PATH}:/workspace/src")
fi
architecture_command=(env -u HSA_OVERRIDE_GFX_VERSION)
gfx_override="${QWEN_GFX_OVERRIDE:-10.3.0}"
if [[ "$gfx_override" != "native" ]]; then
    architecture_command+=("HSA_OVERRIDE_GFX_VERSION=${gfx_override}")
fi
exec "$runtime" run --rm --pull=never --user "$(id -u):$(id -g)" \
    --device=/dev/kfd --device=/dev/dri \
    --security-opt label=disable --security-opt seccomp=unconfined \
    -v "${project_root}:/workspace" -w /workspace \
    -e HF_HOME=/workspace/.cache/huggingface \
    -e TOKENIZERS_PARALLELISM=false -e HF_HUB_DISABLE_TELEMETRY=1 \
    "${environment_args[@]}" --entrypoint /usr/bin/env "$image" \
    "${architecture_command[@]:1}" python3 -m qwen_classifier "$@"
