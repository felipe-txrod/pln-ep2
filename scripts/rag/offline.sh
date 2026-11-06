#!/usr/bin/env bash
# Run offline diagnostics without exposing a GPT classification command.
set -euo pipefail
stage="${1:-}"
if [[ "$stage" != encode && "$stage" != retrieval && "$stage" != interventions && "$stage" != report ]]; then
    printf '%s\n' 'Usage: bash scripts/rag/offline.sh encode|retrieval|interventions|report [arguments]' >&2
    exit 2
fi
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
research_config="${RAG_RESEARCH_CONFIG:-configs/rag-research.json}"
if [[ "$stage" == encode ]]; then
    gpu_args=()
    research_device="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("device","gpu"))' "$research_config")"
    if [[ "$research_device" == gpu ]]; then
        gpu_args=(--device=/dev/kfd --device=/dev/dri -e HSA_OVERRIDE_GFX_VERSION=10.3.0)
    fi
    exec docker run --rm --pull=never --network=none --cpus=2 \
        --user "$(id -u):$(id -g)" "${gpu_args[@]}" \
        --security-opt label=disable --security-opt seccomp=unconfined \
        -v "$project_root:/workspace" -w /workspace \
        -e PYTHONPATH=/workspace/src -e HF_HUB_OFFLINE=1 \
        -e TOKENIZERS_PARALLELISM=false -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
        "${NORBERTO_IMAGE:-pln-rocm:6.4.1}" python3 -u -m rag_research \
        --config "$research_config" "$stage" "$@"
fi
# These commands only read local encoders and historical replies. No remote backend exists.
export PYTHONPATH="${RAG_RESEARCH_RUNTIME:-.cache/cascade-runtime:.cache/rag-runtime}:src"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec "${RAG_RESEARCH_PYTHON:-python3.12}" -m rag_research --config "$research_config" "$stage" "$@"
