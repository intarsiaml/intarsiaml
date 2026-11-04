#!/usr/bin/env bash
# fleetq job on spark2 (--load cpu): intarsia.base = main before failpaths (03f8dc2a3, as the failpaths run), intarsia = this head. Both Sparks.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
export SPARK_DIR=intarsia-dsv41-attnwire
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/dsv41-tp2/spark-dir.sh"
H=spark2
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
BASE=$(git -C "$REPO" rev-parse 03f8dc2a3)
rsh "mkdir -p $SPARK_DIR/srcbase $SPARK_DIR/bin $SPARK_DIR/logs"
log "base = $BASE"
rsh "test -d $SPARK_DIR/srcbase/target || cp -a ~/intarsia-dsv41-tp2/src/target $SPARK_DIR/srcbase/target"
git -C "$REPO" archive "$BASE" | ssh -o BatchMode=yes $H "tar -x -m -C $SPARK_REL/srcbase"
rsh "find $SPARK_DIR/srcbase/crates $SPARK_DIR/srcbase/kernels -type f -exec touch {} +"
log "== base build (nvcc, MAX_JOBS=4)"
rsh "set -o pipefail; cd $SPARK_DIR/srcbase && INTARSIA_BUILD_STAMP='base ${BASE:0:9}' cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3"
rsh "cp $SPARK_DIR/srcbase/target/release/intarsia $SPARK_DIR/bin/intarsia.base && ssh -o BatchMode=yes spark1 'mkdir -p $SPARK_DIR/bin' && scp -q $SPARK_DIR/bin/intarsia.base spark1:$SPARK_REL/bin/"
for h in spark1 spark2; do echo "$h: intarsia.base $(ssh -n -o BatchMode=yes $h "md5sum $SPARK_DIR/bin/intarsia.base | cut -c1-12")"; done
log "== head build"
TOUCH_ALL=1 "$REPO/scripts/dsv41-tp2/build.sh"
log "done"
