#!/usr/bin/env bash
# ON tatooine (cpu job): compile scripts/mr/lgbench.hip (logits rows harness) in the dev image, and print VGPR / spills of the
# row kernels. usage: lgbuild.sh [extra hipcc flags]
set -eo pipefail
cd ~/mr-hip
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p .mrbin
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG \
  hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I kernels/hip "$@" scripts/mr/lgbench.hip -o .mrbin/lgbench 2>&1
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG \
  hipcc -c -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage -I kernels/hip "$@" kernels/hip/matmul.hip -o .mrbin/matmul.o > .mrbin/matmul-res.log 2>&1
python3 - <<'PY'
import re
t = open('/home/josef/mr-hip/.mrbin/matmul-res.log').read()
for b in re.split(r'\n(?=\S*matmul.hip:\d+:\d+: remark: Function Name)', t):
    m = re.search(r'Function Name: (\S+)', b)
    if not m or 'gemv_kernel' not in m.group(1) and 'rows_wide' not in m.group(1):
        continue
    g = lambda k: (re.search(k + r': (\d+)', b) or [0, '?'])[1]
    mm = re.search(r'ILi(\d+)ELi(\d+)ELi(\d+)E(?:Lb(\d))?', m.group(1))
    print('wide' if 'wide' in m.group(1) else 'old ', mm.groups() if mm else m.group(1)[:60], 'VGPR', g('VGPRs'), 'spill', g('VGPRs Spill'), 'scratch', g('ScratchSize \\[bytes/lane\\]'), 'occ', g('Occupancy \\[waves/SIMD\\]'))
PY
