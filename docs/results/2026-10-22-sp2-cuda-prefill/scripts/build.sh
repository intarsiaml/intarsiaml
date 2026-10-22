#!/usr/bin/env bash
# Lane S1 (docs/results/2026-10-22-s1-spark-prefill.md): build the dev tree (this worktree) and the base tree
# (origin/main export, BASE_REF, default 06fba3a9) on spark2, CLI + CUDA test binaries, inside a fleetq slot (vLLM up):
#   fleetq run --host spark2 --mem 16 --label "..." -- docs/results/2026-10-22-sp2-cuda-prefill/scripts/build.sh
# Trees: ~/intarsia-decfuse/sp2-dev, ~/intarsia-decfuse/sp2-base (lane-unique names). Every synced source file is touched
# after the rsync so a seeded target dir can never look fresh (round 7's base-build pitfall). Checks that the base binary
# has none of the new symbols and the dev binary has them.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
export HOST=${HOST:-spark2}
BASE_REF=${BASE_REF:-f9087322}
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STEPS=${STEPS:-base,dev}
has() { [[ ",$STEPS," == *",$1,"* ]]; }
seed() {  # seed the registry-dependency cache from an existing tree on the box
  rsh "mkdir -p $R/$1; test -d $R/$1/target || cp -a $R/${SEED_TREE:-tp2-base}/target $R/$1/target"
}
touch_tree() { rsh "cd $R/$1 && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"; }
build() {  # build <tree> <src dir> <stamp>
  local t=$1 src=$2 stamp=$3
  seed "$t"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude "$src/" "$HOST:intarsia-decfuse/$t/"
  touch_tree "$t"
  rsh "cd $R/$t && env INTARSIA_BUILD_STAMP='$stamp' CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/acc cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^(error|warning: unused)|Finished' | tail -5"
  rsh "ls -la $R/$t/target/acc/release/intarsia"
}
if has base; then
  TMP=$(mktemp -d)
  git -C "$REPO" archive "$BASE_REF" | tar -x -C "$TMP"
  build sp2-base "$TMP" "base-$BASE_REF"
  rm -rf "$TMP"
fi
if has dev; then
  build sp2-dev "$REPO" "$("$REPO/scripts/tree-stamp.sh")"
  rsh "cd $R/sp2-dev && { test -d target/test || cp -a target/acc target/test; }"  # tests in their own target dir (lane rules)
  rsh "cd $R/sp2-dev && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8} CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-backend-cuda --no-run 2>&1 | grep -E '^error|Finished' | tail -3" || true
fi
for t in sp2-base sp2-dev; do
  rsh "test -x $R/$t/target/acc/release/intarsia && echo \"$t: pair=\$(grep -c ik_matmul_fp8_pair_gemm_reg $R/$t/target/acc/release/intarsia) nvfp4reg=\$(grep -c ik_matmul_nvfp4_glu_gemm $R/$t/target/acc/release/intarsia) \$($R/$t/target/acc/release/intarsia --version 2>&1 | head -1)\"" || true
done
