#!/usr/bin/env bash
# Lane G1c (CUDA graphs under tensor parallelism): build the CUDA CLI on spark1 for one or more trees and copy each binary
# to spark2 (no Rust toolchain there; same CUDA 13.0.3, same paths). Runs INSIDE a small fleetq slot (vLLM stays up):
#   fleetq run --host spark1 --mem 4 --label "G1c build" -- scripts/tp2-graphs/build.sh <tree>...
# Trees:
#   dev            this working tree (stamp from scripts/tree-stamp.sh)
#   base           $BASE_REF (default origin/main), exported with git archive
#   mut-<name>     this working tree with mutant <name> applied by scripts/tp2-graphs/mutate.py (the tree is not touched)
# Binaries: spark{1,2}:~/intarsia-tp2/bin/g1c-<tree>-intarsia.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
BASE_REF=${BASE_REF:-origin/main}
export HOST=spark1 CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-6}
for t in "$@"; do
  case $t in
    dev) "$REPO/scripts/cuda-decode/build.sh" g1c-dev ;;
    base)
      tmp=$(mktemp -d)
      git -C "$REPO" archive "$BASE_REF" | tar -x -C "$tmp"
      SRC_DIR=$tmp INTARSIA_STAMP="head=$(git -C "$REPO" rev-parse --short=7 "$BASE_REF") code-tree=$(git -C "$REPO" rev-parse "$BASE_REF^{tree}" | cut -c1-12) dirty=no" \
        "$REPO/scripts/cuda-decode/build.sh" g1c-base
      rm -rf "$tmp" ;;
    mut-*)
      tmp=$(mktemp -d)
      rsync -a --exclude target --exclude .git --exclude .claude --exclude docs "$REPO/" "$tmp/"
      python3 "$S/mutate.py" "$tmp" "${t#mut-}"
      SRC_DIR=$tmp INTARSIA_STAMP="$("$REPO/scripts/tree-stamp.sh") dirty=yes mutant=${t#mut-}" "$REPO/scripts/cuda-decode/build.sh" g1c-$t
      rm -rf "$tmp" ;;
    *) echo "unknown tree $t"; exit 2 ;;
  esac
  ssh -n -o BatchMode=yes spark1 "mkdir -p ~/intarsia-tp2/bin && cp ~/intarsia-decfuse/g1c-$t/target/acc/release/intarsia ~/intarsia-tp2/bin/g1c-$t-intarsia && scp -q ~/intarsia-tp2/bin/g1c-$t-intarsia spark2:intarsia-tp2/bin/g1c-$t-intarsia.new && ssh spark2 mv intarsia-tp2/bin/g1c-$t-intarsia.new intarsia-tp2/bin/g1c-$t-intarsia"
  for h in spark1 spark2; do ssh -n -o BatchMode=yes "$h" "echo $h: \$(~/intarsia-tp2/bin/g1c-$t-intarsia --version 2>&1 | tr '\n' ' ')"; done
done
