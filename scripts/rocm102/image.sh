#!/usr/bin/env bash
# Lane rocm102 (ON tatooine, CPU only): build the ROCm 10.2.0 dev image from deploy/tatooine/Dockerfile.hip-dev-rocm10.
set -euo pipefail
cd "$HOME/intarsiaml-rocm102/deploy/tatooine"
docker build -q -t intarsia-hip-dev:rocm10.2.0-rust1.90.0 -f Dockerfile.hip-dev-rocm10 .
docker run --rm intarsia-hip-dev:rocm10.2.0-rust1.90.0 bash -c 'cat /opt/rocm/.info/version; hipcc --version | head -2; rustc --version'
