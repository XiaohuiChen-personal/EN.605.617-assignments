#!/usr/bin/env bash
# Build assignment.exe. Extra arguments are forwarded to make.
set -euo pipefail
export PATH="/usr/local/cuda/bin:${PATH}"
cd "$(dirname "$0")"
make "$@"
