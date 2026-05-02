#!/usr/bin/env bash
# =============================================================================
# run.sh  –  Build and run the CUDA Image Processing pipeline
# =============================================================================
set -euo pipefail

BINARY="./build/image_processing"
INPUT_DIR="./input"
OUTPUT_DIR="./output"

usage() {
  echo "Usage:"
  echo "  $0                         # build + run all PPMs in ./input/"
  echo "  $0 <image.ppm>             # build + run a single image"
  echo "  $0 --generate-test         # generate a test image, then run"
  echo "  $0 --build-only            # only compile"
  exit 1
}

# --------------------------------------------------------------------------
# 1. Build
# --------------------------------------------------------------------------
echo "=== Building ==="
make all
echo ""

# --------------------------------------------------------------------------
# 2. Handle arguments
# --------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
  # No args – process entire input directory
  mkdir -p "$OUTPUT_DIR"
  if compgen -G "${INPUT_DIR}/*.ppm" > /dev/null 2>&1; then
    echo "=== Running on all PPMs in ${INPUT_DIR}/ ==="
    "$BINARY" --dir "$INPUT_DIR" "$OUTPUT_DIR"
  else
    echo "No .ppm files found in ${INPUT_DIR}/."
    echo "Run:  $0 --generate-test   to create a synthetic test image first."
    exit 1
  fi

elif [[ "$1" == "--generate-test" ]]; then
  mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
  echo "=== Generating synthetic test images ==="
  python3 scripts/gen_test_image.py "${INPUT_DIR}/test_512.ppm"   512  512
  python3 scripts/gen_test_image.py "${INPUT_DIR}/test_1024.ppm" 1024 1024
  echo "=== Running pipeline ==="
  "$BINARY" --dir "$INPUT_DIR" "$OUTPUT_DIR"

elif [[ "$1" == "--build-only" ]]; then
  echo "Build complete."
  exit 0

elif [[ -f "$1" ]]; then
  mkdir -p "$OUTPUT_DIR"
  echo "=== Running on single image: $1 ==="
  "$BINARY" "$1" "$OUTPUT_DIR"

else
  usage
fi

echo ""
echo "Done. Results are in ${OUTPUT_DIR}/"
