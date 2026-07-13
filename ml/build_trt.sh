#!/usr/bin/env bash
# Build a TensorRT engine from the exported ONNX model.
#
# TODO(hardware): This script requires an NVIDIA GPU machine (e.g. a T4
# instance) with TensorRT installed -- it cannot run on Apple Silicon.
# Run `python -m ml.export_onnx ...` first, copy the .onnx file over,
# then run this script on the GPU box.
#
# Usage:
#   ./ml/build_trt.sh artifacts/sr_model.onnx artifacts/sr_model_fp16.engine
#
# The engine is built with FP16 enabled (T4 tensor cores) and dynamic
# input shapes covering tiled inference (512x512 tiles) up to full 480p frames.

set -euo pipefail

ONNX_PATH="${1:-artifacts/sr_model.onnx}"
ENGINE_PATH="${2:-artifacts/sr_model_fp16.engine}"

if ! command -v trtexec >/dev/null 2>&1; then
    echo "ERROR: trtexec not found. Install TensorRT (or use the NGC container:" >&2
    echo "  docker run --gpus all -v \$PWD:/work -w /work nvcr.io/nvidia/tensorrt:24.05-py3 \\" >&2
    echo "    ./ml/build_trt.sh $ONNX_PATH $ENGINE_PATH )" >&2
    exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --minShapes=lr:1x3x64x64 \
    --optShapes=lr:1x3x512x512 \
    --maxShapes=lr:1x3x854x854 \
    --memPoolSize=workspace:2048

echo "Engine written to $ENGINE_PATH"
echo "Benchmark it with:"
echo "  python -m ml.benchmark --RANDOM_INIT --ONNX_PATH $ONNX_PATH --TRT_ENGINE_PATH $ENGINE_PATH"
