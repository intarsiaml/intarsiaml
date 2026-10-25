#!/usr/bin/env bash
# Lane mr, ON tatooine inside a fleetq job (--load cpu): ISA of mxfp4_gemv_rows_kernel<M> (gfx1201), instruction mix per M
# (VALU count, v_fma/v_fmac, VOPD dual-issue ops). Needs ~/mr-hip (scripts/mr/sync.sh dev).
cd ~/mr-hip
mkdir -p .mrbin
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG \
  hipcc -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 --cuda-device-only -S -I kernels/hip scripts/mr/mxbench.hip -o /work/.mrbin/rows.s 2>&1 | tail -3
ls -la .mrbin/rows.s
python3 - <<'PY'
import re
t = open('/home/josef/mr-hip/.mrbin/rows.s').read()
funcs = re.split(r'\n(?=_Z\S+:\s*(?:;.*)?\n)', t)
for f in funcs:
    m = re.match(r'(_Z\S+):', f)
    if not m or 'rows_kernel' not in m.group(1):
        continue
    name = m.group(1)
    mm = re.search(r'kernelILi(\d+)E(\w+?)EEvP', name)
    ops = {}
    for l in f.split('\n'):
        l = l.strip()
        if not l or l.startswith(('.', ';')):
            continue
        o = l.split()[0]
        ops[o] = ops.get(o, 0) + 1
    valu = sum(v for k, v in ops.items() if k.startswith('v_'))
    fma = sum(v for k, v in ops.items() if k in ('v_fma_f32', 'v_fmac_f32'))
    dual = {k: v for k, v in ops.items() if k.startswith('v_dual')}
    ds = sum(v for k, v in ops.items() if k.startswith('ds_'))
    gl = sum(v for k, v in ops.items() if k.startswith('global_load') or k.startswith('buffer_load'))
    print(mm.groups() if mm else name[:50], 'valu', valu, 'fma/fmac', fma, 'dual', sum(dual.values()), dict(list(dual.items())[:4]), 'ds', ds, 'gload', gl)
PY
# lm_head rows (matmul.hip gemv_dispatch<M>): registers and spills per M, to see whether the M > 8 cliff is a spill
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v ~/mr-hip:/work -w /work $IMG \
  hipcc -c -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 -Rpass-analysis=kernel-resource-usage -I kernels/hip kernels/hip/matmul.hip -o /work/.mrbin/matmul.o > .mrbin/matmul-res.log 2>&1
python3 - <<'PY'
import re
t = open('/home/josef/mr-hip/.mrbin/matmul-res.log').read()
for b in re.split(r'\n(?=\S*matmul.hip:\d+:\d+: remark: Function Name)', t):
    m = re.search(r'Function Name: (\S+)', b)
    if not m or 'gemv' not in m.group(1): continue
    g = lambda k: (re.search(k + r': (\d+)', b) or [0, '?'])[1]
    print(m.group(1)[:96], 'VGPR', g('VGPRs'), 'spill', g('VGPRs Spill'), 'scratch', g('ScratchSize \\[bytes/lane\\]'), 'occ', g('Occupancy \\[waves/SIMD\\]'))
PY
