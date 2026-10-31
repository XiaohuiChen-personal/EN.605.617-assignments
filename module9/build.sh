#!/usr/bin/env bash
# Linux build script for the course runner. Extra args go to make.
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH (also looked in /usr/local/cuda/bin)" >&2
    exit 1
fi
cd "$(dirname "$0")"
if [[ -z "${CUDNN_ROOT:-}" ]]; then
    for d in "$HOME"/tmp/en605617/m9/venv/lib/python3.*/site-packages/nvidia/cudnn; do
        if [[ -f "$d/include/cudnn.h" ]]; then
            CUDNN_ROOT=$d
            break
        fi
    done
fi
if [[ ! -f "${CUDNN_ROOT:-}/include/cudnn.h" ]]; then
    echo "cuDNN headers not found. Set CUDNN_ROOT to the directory with include/cudnn.h and lib/libcudnn.so.9" >&2
    exit 1
fi
make CUDNN_ROOT="$CUDNN_ROOT" "$@"
