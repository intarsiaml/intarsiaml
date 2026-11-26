#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-batch): does the RT = false instantiation of ink_attn_dec_k compile to main's
# ISA when the TU holds no RT instantiation? Variant C = the branch's ink_attn.hip without the row-table launcher.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
rm -rf .scratch/isa/varC && mkdir -p .scratch/isa/varC logs && cp -r kernels .scratch/isa/varC/
F=.scratch/isa/varC/kernels/hip/ink_attn.hip
n=$(grep -n '^/\* InkAttnDecode row-table form' $F | cut -d: -f1)
head -n $((n-1)) $F > $F.t && mv $F.t $F
grep -c ink_attn_decode_bf16_rows $F
deploy/tatooine/dev.sh bash -c '
cd /work/.scratch/isa
[ -s main-ink_attn.s ] || /opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I /work/.scratch/main-tree/kernels --cuda-device-only -S /work/.scratch/main-tree/kernels/hip/ink_attn.hip -o main-ink_attn.s
[ -s branch-ink_attn.s ] || /opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I /work/kernels --cuda-device-only -S /work/kernels/hip/ink_attn.hip -o branch-ink_attn.s
/opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wno-unused-function --offload-arch=gfx1201 -I varC/kernels --cuda-device-only -S varC/kernels/hip/ink_attn.hip -o varC-ink_attn.s || echo COMPILE FAIL
' < /dev/null > logs/isa2.log 2>&1
tail -3 logs/isa2.log
python3 .scratch/isa_cmp.py .scratch/isa/main-ink_attn.s .scratch/isa/varC-ink_attn.s
python3 .scratch/isa_cmp.py .scratch/isa/main-ink_attn.s .scratch/isa/branch-ink_attn.s | tail -2
