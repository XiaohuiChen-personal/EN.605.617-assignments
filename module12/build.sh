#!/usr/bin/env bash
# Linux build for the course runner. Extra args go to make.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v gcc >/dev/null 2>&1; then
    echo "gcc not found" >&2
    exit 1
fi
if [[ ! -f /usr/include/CL/cl.h \
    && ! -f "${HOME}/.local/include/CL/cl.h" ]]; then
    echo "OpenCL headers not found" >&2
    exit 1
fi
make "$@"
