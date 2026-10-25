#!/usr/bin/env bash
# on tatooine: compile the harness (+ resource usage) in the dev image. usage: mrbuild.sh [extra hipcc flags]
set -eo pipefail
cd ~/mr-hip
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p ~/mr-hip/.mrbin
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG \
  hipcc -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I kernels/hip "$@" scripts/mr/mxbench.hip -o .mrbin/mxbench 2>&1
