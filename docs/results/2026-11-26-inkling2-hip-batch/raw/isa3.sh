#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-batch): after the TU split, the solo modules ink_mix.hip / ink_attn.hip of the
# branch against main's, kernel by kernel (instructions and descriptor); the row-table TUs must compile on their own.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
mkdir -p .scratch/isa logs
deploy/tatooine/dev.sh bash -c '
cd /work/.scratch/isa
C="/opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 --cuda-device-only -S"
for f in ink_mix ink_attn; do
  [ -s main-$f.s ] || $C -I /work/.scratch/main-tree/kernels /work/.scratch/main-tree/kernels/hip/$f.hip -o main-$f.s || echo "COMPILE FAIL main $f"
  $C -I /work/kernels /work/kernels/hip/$f.hip -o split-$f.s || echo "COMPILE FAIL split $f"
  $C -I /work/kernels /work/kernels/hip/${f}_rows.hip -o split-${f}_rows.s || echo "COMPILE FAIL split ${f}_rows"
done
' < /dev/null > logs/isa3.log 2>&1
grep -a "COMPILE FAIL\|error" logs/isa3.log | head
for f in ink_mix ink_attn; do python3 .scratch/isa_cmp.py .scratch/isa/main-$f.s .scratch/isa/split-$f.s; done
grep -c '^_Z.*:' .scratch/isa/split-ink_mix_rows.s .scratch/isa/split-ink_attn_rows.s
