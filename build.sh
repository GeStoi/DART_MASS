#!/bin/bash
# ==============================================================================
# Build the MASS Docker image
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo " Building MASS Docker image"
echo " Platform: linux/amd64 (emulated on ARM Mac)"
echo " This will take 15-30 minutes on first build"
echo "=============================================="
echo ""

# Build with docker compose
docker compose build --progress=plain 2>&1 | tee build.log

echo ""
echo "=============================================="
echo " Build complete!"
echo " Run: ./run.sh to start the container"
echo "=============================================="
