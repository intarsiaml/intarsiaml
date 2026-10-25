#!/usr/bin/env bash
# Lane l5, ON tatooine (CPU slot, no GPU): compile the harness with --save-temps and the kernel resource report, to read the ISA of the kernels.
#   FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --load cpu --expect 5 --label "l5: isa of f8bench (CPU)" -- \
#     bash -c "ssh tatooine 'NAME=<snapshot> bash -s' < scripts/l5/hasm.sh"
# Output: ~/intarsiaml-l5-<name>/l5-out/asm/f8bench-hip-amdgcn-amd-amdhsa-gfx1201.s and asm/resource.txt
set -eo pipefail
name=${NAME:?}
cd ~/intarsiaml-l5-$name
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p l5-out/asm
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/intarsiaml-l5-$name:/work -w /work/l5-out/asm $IMG \
  hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I /work/kernels/hip -I /work/kernels/abi -save-temps \
  -Rpass-analysis=kernel-resource-usage /work/scripts/l5/f8bench.hip /work/kernels/hip/ik_runtime.hip -ldl -o /work/l5-out/asm/f8bench-isa > resource.txt 2>&1 < /dev/null
grep -E 'Function Name|VGPRs:|Spill|LDS Size|Occupancy|SGPRs:' resource.txt | grep -A6 'moe_f8_kernel\|moe_wmma_kernel_r7' | head -80
