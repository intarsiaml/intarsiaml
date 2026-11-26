#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-batch, branch ink/hip-batch-mixmode): (1) device ISA of ink_mix.hip and
# ink_mix_rows.hip against the PR head's (the fix is host code only); (2) host objects: ink_mix.o defines ik_inkmix::g_mix_mask /
# g_mix_dec, ink_mix_rows.o only references them (one variable after linking). The full build runs after this (build.sh b9).
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-batch"
export INTARSIA_REPO=$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-batch INTARSIA_CONTAINER_NAME=ink-hip-batch-build
trap 'docker rm -f ink-hip-batch-build >/dev/null 2>&1' EXIT
mkdir -p .scratch/isa logs
deploy/tatooine/dev.sh bash -c '
cd /work/.scratch/isa
F="-O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I /work/kernels"
for f in ink_mix ink_mix_rows; do
  [ -s head-$f.s ] || /opt/rocm/bin/hipcc $F --cuda-device-only -S /work/.scratch/head/kernels/hip/$f.hip -I /work/.scratch/head/kernels -o head-$f.s || echo "COMPILE FAIL head $f"
  /opt/rocm/bin/hipcc $F --cuda-device-only -S /work/kernels/hip/$f.hip -o fix-$f.s || echo "COMPILE FAIL device $f"
  /opt/rocm/bin/hipcc $F -c /work/kernels/hip/$f.hip -o fix-$f.o || echo "COMPILE FAIL host $f"
  echo "== nm $f.o (g_mix)"; nm -C fix-$f.o | grep -E "g_mix_(mask|dec)" || echo "(none)"
done
' < /dev/null > logs/mixmode.log 2>&1
grep -aE "COMPILE FAIL|== nm|g_mix|none" logs/mixmode.log
for f in ink_mix ink_mix_rows; do python3 .scratch/isa_cmp.py .scratch/isa/head-$f.s .scratch/isa/fix-$f.s same | tail -2; done
