#!/usr/bin/env bash
# VGPR counts and code size of the expert kernels (device asm, no GPU): ON tatooine in a fleetq --load cpu job.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-moe"
mkdir -p logs
INK_CONTAINER=ink-hip-moe-asm bash .scratch/run.sh bash -c 'hipcc -S -O3 -std=c++17 -fPIC --offload-arch=gfx1201 --cuda-device-only -I kernels/hip -o /work/logs/ink_moe.s kernels/hip/ink_moe.hip' > logs/asm.log 2>&1
rc=$?
tail -5 logs/asm.log
python3 - <<'PY'
import re,subprocess
s=open('logs/ink_moe.s').read()
rows=re.findall(r'\.set (_Z\S+)\.num_vgpr, (\d+)',s)
dem=subprocess.run(['c++filt'],input='\n'.join(n for n,_ in rows),capture_output=True,text=True).stdout.strip().split('\n')
st=dict(re.findall(r'\.set (_Z\S+)\.private_seg_size, (\d+)',s))
for (n,v),d in zip(rows,dem):
    d=d.replace('(anonymous namespace)::','').replace('void ','').replace('((anonymous namespace)::MoeArgs)','')
    print(f"{d:<58} vgpr {v:>4} stack {st.get(n,'?')}")
PY
exit $rc
