#!/bin/bash
# ==============================================================================
# Run the MASS Docker container
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create output directory if not exists
mkdir -p nn_output

echo "=============================================="
echo " Starting MASS container"
echo "=============================================="

# Start container
docker compose up -d

echo ""
echo "Container is running. Enter with:"
echo "  docker compose exec mass bash"
echo ""
echo "Quick commands inside the container:"
echo "  # Train the model"
echo "  cd /workspace/MASS/python && python3 main.py -d ../data/metadata.txt"
echo ""
echo "  # Run the render UI (headless with Xvfb)"
  echo "  cd /workspace/MASS && ./build/render/render ./data/metadata.txt"
echo ""
echo "  # Run with trained model"
  echo "  cd /workspace/MASS && ./build/render/render ./data/metadata.txt ./nn/xxx.pt ./nn/xxx_muscle.pt"
echo ""
echo "To stop:  docker compose down"
echo "=============================================="
