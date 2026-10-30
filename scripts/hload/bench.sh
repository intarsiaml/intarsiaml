#!/usr/bin/env bash
# Lane hload (from scripts/k4fp8/bench.sh), ON tatooine: compile a standalone harness of scripts/hload (bench_moe.hip, bench_fp8.hip, ...)
# that #includes a kernels/hip file, in the HIP dev image, with the VGPR / scratch report in hload-out/compile-<tag>.log.
#   bench.sh <tree> <harness name> <tag> [extra hipcc flags]
# KTREE=<dir> takes kernels/hip from that tree instead (e.g. the main arm ~/intarsiaml-tp2-k4fp8) and the harness from <tree>.
# INC=<dir> (relative to <tree>) puts a variant copy of the kernel file first. CPU only (a --load cpu --mem 0 fleetq job);
# running hload-out/<harness>-<tag> needs a GPU window (scripts/hload/mbwin.sh).
set -uo pipefail
tree=${1:?tree dir}; h=${2:?harness}; tag=${3:?tag}; shift 3
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p "$tree/hload-out"
kt=/work; mnt=""
if [[ -n "${KTREE:-}" ]]; then kt=/ktree; mnt="-v $KTREE:/ktree:ro"; fi
rc=0
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$tree":/work $mnt -w /work "$IMG" bash -c \
  "hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-function -Rpass-analysis=kernel-resource-usage ${INC:+-I $INC} -I $kt/kernels/hip -I $kt/kernels/abi $* scripts/hload/$h.hip -o hload-out/$h-$tag" \
  > "$tree/hload-out/compile-$h-$tag.log" 2>&1 < /dev/null || rc=$?
echo "compile $h-$tag exit $rc"
ls -la "$tree/hload-out/$h-$tag"
exit $rc
