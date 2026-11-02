#!/usr/bin/env bash
# Lane dsv41/tp2: sync this worktree to spark2, build the CUDA CLI and the dsv41_runtime test binary there, copy both to
# $SPARK_DIR/bin on both Sparks. Runs INSIDE a fleetq slot (vLLM stays up):
#   fleetq run --host spark2 --load cpu --mem 16 --expect 25 --label "dsv41 tp2 build" -- scripts/dsv41-tp2/build.sh
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H=${BUILD_HOST:-spark2}
STAMP="$(git -C "$REPO" rev-parse --short=9 HEAD) dirty=$(git -C "$REPO" diff --quiet HEAD -- crates kernels && echo no || echo yes) $(date +%m%d-%H%M)"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh 'mkdir -p '"$SPARK_DIR"'/src '"$SPARK_DIR"'/bin; test -d '"$SPARK_DIR"'/src/target || { test -d ~/intarsia-dsv41-tp2/src/target && cp -a ~/intarsia-dsv41-tp2/src/target '"$SPARK_DIR"'/src/target; true; }'
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .scratch "$REPO/" "$H:$SPARK_REL/src/"
echo "stamp: $STAMP"
rsh "cd $SPARK_DIR/src && INTARSIA_BUILD_STAMP='$STAMP' cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3"
T=$(rsh "cd $SPARK_DIR/src && cargo test --release -p intarsia-models --test dsv41_runtime --no-run 2>&1 | grep -o 'target/release/deps/dsv41_runtime-[0-9a-f]*' | tail -1")
rsh "cd $SPARK_DIR/src && cp target/release/intarsia $SPARK_DIR/bin/intarsia && cp $T $SPARK_DIR/bin/dsv41_runtime && echo '$STAMP' > $SPARK_DIR/bin/STAMP && ls -la $SPARK_DIR/bin"
O=$([[ $H == spark2 ]] && echo spark1 || echo spark2)
rsh "ssh -o BatchMode=yes $O 'mkdir -p $SPARK_DIR/bin' && scp -q $SPARK_DIR/bin/intarsia $SPARK_DIR/bin/dsv41_runtime $SPARK_DIR/bin/STAMP $O:$SPARK_REL/bin/"
for h in spark1 spark2; do echo "$h: $(ssh -n -o BatchMode=yes $h "cat $SPARK_DIR/bin/STAMP; md5sum $SPARK_DIR/bin/intarsia | cut -c1-12")"; done
