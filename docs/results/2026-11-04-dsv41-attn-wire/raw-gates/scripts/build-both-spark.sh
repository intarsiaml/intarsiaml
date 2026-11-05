#!/usr/bin/env bash
# Window build, ONE fleetq job on spark2 (--load cpu): the merged base dbe5581ae (dsv41/failpaths) as intarsia.base, then this branch as intarsia, both on both Sparks.
# Everything under ~/intarsia-dsv41-attnwire on spark2 / spark1 (bin/, logs/); removed after the window (logs fetched first).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
export SPARK_DIR=intarsia-dsv41-attnwire
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/dsv41-tp2/spark-dir.sh"
H=spark2
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
BASE=$(git -C "$REPO" rev-parse dbe5581ae)
rsh "mkdir -p $SPARK_DIR/srcbase $SPARK_DIR/bin $SPARK_DIR/logs"
scp -q -o BatchMode=yes "$REPO/.scratch/killmine.sh" "$REPO/.scratch/memwatch.sh" "$H:$SPARK_REL/"
ssh -n -o BatchMode=yes $H "nohup $SPARK_REL/memwatch.sh > /dev/null 2>&1 &"
trap 'ssh -n -o BatchMode=yes $H "pkill -f \"intarsia-dsv41-attnwire/memwatch[.]sh\"; true"' EXIT
log "base = dbe5581ae $BASE"
rsh "test -d $SPARK_DIR/srcbase/target || cp -a ~/intarsia-dsv41-tp2/src/target $SPARK_DIR/srcbase/target"
git -C "$REPO" archive $BASE | ssh -o BatchMode=yes $H "tar -x -m -C $SPARK_REL/srcbase"
rsh "find $SPARK_DIR/srcbase/crates $SPARK_DIR/srcbase/kernels -type f -exec touch {} +"
log "== base build (nvcc, MAX_JOBS=4)"
rsh "set -o pipefail; cd $SPARK_DIR/srcbase && INTARSIA_BUILD_STAMP='base failpaths ${BASE:0:9}' cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3"
rsh "cp $SPARK_DIR/srcbase/target/release/intarsia $SPARK_DIR/bin/intarsia.base && ssh -o BatchMode=yes spark1 'mkdir -p $SPARK_DIR/bin' && scp -q $SPARK_DIR/bin/intarsia.base spark1:$SPARK_REL/bin/"
for h in spark1 spark2; do echo "$h: intarsia.base $(ssh -n -o BatchMode=yes $h "md5sum $SPARK_DIR/bin/intarsia.base | cut -c1-12")"; done
log "== head build"
TOUCH_ALL=1 "$REPO/scripts/dsv41-tp2/build.sh"
log "done"
