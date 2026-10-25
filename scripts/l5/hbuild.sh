#!/usr/bin/env bash
# Lane l5, ON tatooine (CPU slot, no GPU): compile the harness in the dev image. usage: hbuild.sh <name> [extra hipcc flags]
#   FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --load cpu --expect 5 --label "l5: build f8bench" -- \
#     ssh tatooine 'bash -s' < scripts/l5/hbuild.sh <name>     (arguments after the script name via env: NAME=<name>)
set -eo pipefail
name=${NAME:-${1:?}}
cd ~/intarsiaml-l5-$name
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p l5-out
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/intarsiaml-l5-$name:/work -w /work $IMG \
  hipcc -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I kernels/hip -I kernels/abi ${HIPFLAGS:-} \
  scripts/l5/f8bench.hip kernels/hip/ik_runtime.hip -ldl -o l5-out/f8bench 2>&1 < /dev/null
