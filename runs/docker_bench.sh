#!/usr/bin/env bash
# Build the image and run the benchmark pinned to one CPU; reports land in runs/docker/.
# Usage: runs/docker_bench.sh [extra benchmark.py args, e.g. --repeats 3]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/docker
docker build -t differometor-bench .
docker run --rm -v "$PWD/runs/docker:/out" differometor-bench --save-arrays /out/benchmark-docker.npz "$@"
