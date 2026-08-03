#!/usr/bin/env bash
# Build the exact-wheel vLLM image for Apertus 1.5 multimodal serving.
#
# Usage:
#   ./build.sh          # onpremai/vllm-apertus-1p5:p20
#   ./build.sh my-tag   # custom local tag

set -euo pipefail

TAG="${1:-p20}"
IMAGE="onpremai/vllm-apertus-1p5:${TAG}"

cd "$(dirname "$0")"

echo "Building ${IMAGE} from exact upstream wheels and pinned sources..."
docker build -t "${IMAGE}" .
echo "Done: ${IMAGE}"
