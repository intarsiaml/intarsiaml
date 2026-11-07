#!/usr/bin/env bash
# Lane mt2 (docs/results/2026-11-07-mtp-moe-tp.md): build the CUDA CLI on spark1 for origin/main (`base`) or this working
# tree (`dev`) in ~/intarsia-mt2/<name>, then copy the binary to ~/intarsia-mt2/bin/<name>-intarsia on both Sparks.
# Runs INSIDE a fleetq slot (vLLM stays up):
#   fleetq run --host spark1,spark2 --mem spark1=20,spark2=1 --load cpu --expect 40 --label "mt2 build" -- scripts/mt2/build.sh base|dev
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
NAME=${1:?base or dev}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H=spark1; R='~/intarsia-mt2'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$H" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1; $*"; }
if [[ $NAME == base ]]; then
  SRC=$(mktemp -d "$REPO/.mt2/base.XXXX"); git -C "$REPO" archive origin/main | tar -x -C "$SRC"
  STAMP="head=$(git -C "$REPO" rev-parse --short=7 origin/main) code-tree=$(git -C "$REPO" rev-parse 'origin/main^{tree}' | cut -c1-12) dirty=no"
else
  SRC=$REPO; STAMP=$("$REPO/scripts/tree-stamp.sh")
fi
rsh "mkdir -p $R/$NAME $R/bin"
rsh "test -d $R/$NAME/target || { test -d ~/intarsia-acc35b/src/target && cp -a ~/intarsia-acc35b/src/target $R/$NAME/target; true; }"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .mt2 --exclude docs/results "$SRC/" "$H:intarsia-mt2/$NAME/"
[[ $NAME == base ]] && rm -rf "$SRC"
echo "stamp: $STAMP"
rsh "cd $R/$NAME && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -5"
rsh "cp $R/$NAME/target/acc/release/intarsia $R/bin/$NAME-intarsia && ssh spark2 mkdir -p intarsia-mt2/bin && scp -q $R/bin/$NAME-intarsia spark2:intarsia-mt2/bin/$NAME-intarsia.new && ssh spark2 mv intarsia-mt2/bin/$NAME-intarsia.new intarsia-mt2/bin/$NAME-intarsia"
for h in spark1 spark2; do ssh -n -o BatchMode=yes "$h" "echo $h: \$(~/intarsia-mt2/bin/$NAME-intarsia --version 2>&1 | tr '\n' ' ')"; done
