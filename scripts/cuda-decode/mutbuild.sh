#!/usr/bin/env bash
# Build one `fused_decode` test binary per mutation (CPU work only, vLLM may stay up), so that the GPU run can happen inside ONE
# short vLLM-down window (w-mutrun.sh) instead of one build + run cycle per mutation:
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-decode/mutbuild.sh [mutation ...]      (env TREE, default dev)
# Per mutation: restore the kernel sources of ~/intarsia-decfuse/$TREE into ~/intarsia-decfuse/mut, apply mutate.py, build
# `--test fused_decode --no-run`, copy the test executable to ~/intarsia-decfuse/mutbins/<name>. `unmutated` is built too.
# The binaries are statically linked against the kernels (build.rs), so they run from any directory.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
TREE=${TREE:-dev}
ALL="norm-skip-sum norm-skip-normed norm-sum-round norm-woff norm-order norm-pair-block norm-nb0 norm-sync norm-read-x gn-round gn-woff gn-order gn-chunk-order gn-group gn-weight gn-silu gn-actq gn-tail"
MUTS=${*:-$ALL}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
rsh "mkdir -p $D/mut $D/mutbins && cd $D/mut && { test -d target || cp -a $D/$TREE/target target; }; rsync -a --delete --exclude target $D/$TREE/ $D/mut/" >/dev/null
build() { # build <name>  -> copies the executable
  rsh "cd $D/mut && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test fused_decode --no-run --message-format=json 2>/dev/null | grep -o '\"executable\":\"[^\"]*fused_decode-[^\"]*\"' | tail -1 | cut -d'\"' -f4 > $D/mutbins/$1.path; test -s $D/mutbins/$1.path && cp \$(cat $D/mutbins/$1.path) $D/mutbins/$1 && echo built \$(basename \$(cat $D/mutbins/$1.path))"
}
restore() { rsh "cd $D/mut && cp $D/$TREE/kernels/cuda/{matmul,fp8}.cu kernels/cuda/"; }
bad=0
restore; echo "== unmutated"; build unmutated || bad=1
for m in $MUTS; do
  restore
  if ! rsh "cd $D/mut && python3 -I scripts/cuda-decode/mutate.py . $m"; then echo "HARNESS: $m did not apply"; bad=1; continue; fi
  echo "== $m"; build "$m" || { echo "HARNESS: $m did not build"; bad=1; }
done
restore
exit $bad
