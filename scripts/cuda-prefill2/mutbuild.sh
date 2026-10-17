#!/usr/bin/env bash
# Build one `gdn_chunked` test binary per mutation of kernels/cuda/gdn_chunked.cu (CPU work only, vLLM may stay up), so the GPU
# run happens inside one short vLLM-down window (mutcheck.sh):
#   fleetq run --host spark1 --mem 8 --label "..." -- scripts/cuda-prefill2/mutbuild.sh [mutation ...]      (env TREE, default dev)
# Per mutation: restore the kernel source of ~/intarsia-decfuse/$TREE into ~/intarsia-decfuse/mut, apply mutate.py, build
# `--test gdn_chunked --no-run`, copy the test executable to ~/intarsia-decfuse/mutbins/gc-<name>. `unmutated` is built too.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
TREE=${TREE:-dev}
ALL="no-interdecay no-kd mask-strict no-solve no-carry tail-decay no-intra-decay no-qe no-qscale"
MUTS=${*:-$ALL}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D='~/intarsia-decfuse'
rsh "mkdir -p $D/mut $D/mutbins && cd $D/mut && { test -d target || cp -a $D/$TREE/target target; }; rsync -a --delete --exclude target $D/$TREE/ $D/mut/" >/dev/null
build() { # build <name>
  rsh "cd $D/mut && CARGO_BUILD_JOBS=6 CARGO_TARGET_DIR=target/acc cargo test --release -p intarsia-backend-cuda --test gdn_chunked --no-run --message-format=json 2>/dev/null | grep -o '\"executable\":\"[^\"]*gdn_chunked[^\"]*\"' | tail -1 | cut -d'\"' -f4 | xargs -I{} cp {} $D/mutbins/gc-$1 && ls -la $D/mutbins/gc-$1"
}
restore() { rsh "cd $D/mut && cp $D/$TREE/kernels/cuda/gdn_chunked.cu kernels/cuda/"; }
bad=0
restore; echo "== unmutated $(date +%T)"; build unmutated || bad=1
for m in $MUTS; do
  restore
  if ! rsh "cd $D/mut && python3 -I scripts/cuda-prefill2/mutate.py . $m"; then echo "HARNESS: $m did not apply"; bad=1; continue; fi
  echo "== $m $(date +%T)"; build "$m" || { echo "HARNESS: $m did not build"; bad=1; }
done
restore
exit $bad
