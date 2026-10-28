#!/usr/bin/env bash
# Lane ars (from the Mac): compile the two all-reduce kernel files with the build's hipcc flags in the HIP dev image on tatooine (no GPU) and
# write their gfx1201 assembly next to them, to read the store / load instructions. Kernels are copied to ~/intarsiaml-ars/.ars/bk first:
#   fleetq run --host tatooine --load cpu --expect 3 --label "ars: hipcc check" -- scripts/ar/hipcheck.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; K="$HERE/../../kernels"
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
ssh -o BatchMode=yes tatooine 'rm -rf ~/intarsiaml-ars/.ars/bk && mkdir -p ~/intarsiaml-ars/.ars/bk' < /dev/null
scp -q -r "$K" tatooine:intarsiaml-ars/.ars/bk/
ssh -o BatchMode=yes tatooine "docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v \$HOME/intarsiaml-ars/.ars/bk/kernels:/b -w /b/hip $IMG sh -c '
for f in allreduce ar_norm; do
  hipcc -c -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I /b/hip -o /tmp/\$f.o \$f.hip && echo \"compiled \$f\" || echo \"FAILED \$f\"
  hipcc -S --cuda-device-only -O3 -std=c++17 -Wno-unused-function --offload-arch=gfx1201 -I /b/hip -o \$f.s \$f.hip
done'" < /dev/null
mkdir -p "$HERE/../../.worktrees/ars/asm" 2>/dev/null
scp -q tatooine:intarsiaml-ars/.ars/bk/kernels/hip/allreduce.s tatooine:intarsiaml-ars/.ars/bk/kernels/hip/ar_norm.s "$HERE/../../.worktrees/ars/asm/" 2>/dev/null
ls -la "$HERE/../../.worktrees/ars/asm/" 2>/dev/null
