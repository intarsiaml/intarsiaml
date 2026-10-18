#!/usr/bin/env bash
# Compare the SASS of every object the CUDA build produces for the base tree (origin/main export, nv-base) and the dev tree
# (nv-dev): the kernels this branch must not change must be identical (vLLM UP, no GPU use):
#   fleetq run --host spark1 --mem 2 --label "nvfp4-mma sass diff" -- scripts/cuda-moe-nvfp4/sass.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh '
cd ~/intarsia-decfuse
od() { ls -td $1/target/acc/release/build/intarsia-backend-cuda-*/out | head -1; }
B=$(od nv-base); D=$(od nv-dev); echo "base: $B"; echo "dev:  $D"
norm() { cuobjdump -sass "$1" 2>/dev/null | grep -vE "^\s*//|^\s*\.headerflags|\.nv\.|Fatbin|arch =|code version|producer|host =|compile_size|identifier" | sed -E "s/_GLOBAL__N__[0-9a-f]+_[0-9]+_[A-Za-z0-9_]*_cu_[0-9a-f]+/NS/g"; }
for o in $B/*.o; do
  n=$(basename $o)
  [ -f $D/$n ] || { echo "$n: only in base"; continue; }
  norm $o > /tmp/sass_a_$FLEETQ_JOB.txt; norm $D/$n > /tmp/sass_b_$FLEETQ_JOB.txt
  if cmp -s /tmp/sass_a_$FLEETQ_JOB.txt /tmp/sass_b_$FLEETQ_JOB.txt; then echo "$n: IDENTICAL ($(wc -l < /tmp/sass_a_$FLEETQ_JOB.txt) lines)"
  else echo "$n: DIFFERENT ($(diff /tmp/sass_a_$FLEETQ_JOB.txt /tmp/sass_b_$FLEETQ_JOB.txt | grep -c "^[<>]") differing lines)"; diff /tmp/sass_a_$FLEETQ_JOB.txt /tmp/sass_b_$FLEETQ_JOB.txt | head -6 | cut -c1-150; fi
done
for o in $D/*.o; do [ -f $B/$(basename $o) ] || echo "$(basename $o): only in dev"; done
rm -f /tmp/sass_a_$FLEETQ_JOB.txt /tmp/sass_b_$FLEETQ_JOB.txt
'
