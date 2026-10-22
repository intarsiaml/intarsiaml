#!/usr/bin/env bash
# Lane fx12 (acknowledged tp_remote Truncate / Snapshot): build the CUDA CLI on spark1 for two trees and copy each binary to
# spark2 (no Rust toolchain there; same CUDA, same paths). Runs INSIDE a small fleetq slot (vLLM stays up):
#   fleetq run --host spark1 --mem 4 --label "fx12 build" -- scripts/tp2-acks/build.sh dev base
# Trees: dev = this working tree (commit first: the stamp names the code), base = $BASE_REF (default origin/main).
# Binaries: spark{1,2}:~/intarsia-tp2/bin/fx12-<tree>-intarsia.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
BASE_REF=${BASE_REF:-origin/main}
export HOST=spark1 CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-6}
for t in "$@"; do
  case $t in
    dev) "$REPO/scripts/cuda-decode/build.sh" fx12-dev ;;
    base)
      tmp=$(mktemp -d)
      git -C "$REPO" archive "$BASE_REF" | tar -x -C "$tmp"
      SRC_DIR=$tmp INTARSIA_STAMP="head=$(git -C "$REPO" rev-parse --short=7 "$BASE_REF") code-tree=$(git -C "$REPO" rev-parse "$BASE_REF^{tree}" | cut -c1-12) dirty=no" \
        "$REPO/scripts/cuda-decode/build.sh" fx12-base
      rm -rf "$tmp" ;;
    *) echo "unknown tree $t"; exit 2 ;;
  esac
  ssh -n -o BatchMode=yes spark1 "mkdir -p ~/intarsia-tp2/bin && cp ~/intarsia-decfuse/fx12-$t/target/acc/release/intarsia ~/intarsia-tp2/bin/fx12-$t-intarsia && scp -q ~/intarsia-tp2/bin/fx12-$t-intarsia spark2:intarsia-tp2/bin/fx12-$t-intarsia.new && ssh spark2 mv intarsia-tp2/bin/fx12-$t-intarsia.new intarsia-tp2/bin/fx12-$t-intarsia"
  for h in spark1 spark2; do ssh -n -o BatchMode=yes "$h" "echo $h: \$(~/intarsia-tp2/bin/fx12-$t-intarsia --version 2>&1 | tr '\n' ' ')"; done
done
