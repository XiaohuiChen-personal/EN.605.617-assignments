#!/usr/bin/env bash
# Linux build script for the course runner. Extra args go to make.
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH:-}"
if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found on PATH (also looked in /usr/local/cuda/bin)" >&2
    exit 1
fi
cd "$(dirname "$0")"
make "$@"

