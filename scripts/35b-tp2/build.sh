#!/usr/bin/env bash
# Lane G1b (docs/design/35b-tp2.md): build the CUDA CLI on spark1 for the base tree (origin/main, exported) and/or the dev
# tree (this working tree), then copy each binary to spark2 (which has no Rust toolchain; same CUDA 13.0.3, same paths).
# Runs INSIDE a small fleetq slot on the build host (BUILD_HOST, default spark1; spark2 has a copy of spark1's toolchain):
#   fleetq run --host spark1 --mem 4 --label "G1b build" -- scripts/35b-tp2/build.sh [base] [dev]
# Binaries: spark{1,2}:~/intarsia-tp2/bin/<tree>-intarsia.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
TREES=("$@"); [[ ${#TREES[@]} -eq 0 ]] && TREES=(dev)
BASE_REF=${BASE_REF:-origin/main}
B=${BUILD_HOST:-spark1}; export HOST=$B CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-6}
for t in "${TREES[@]}"; do
  case $t in
    base)
      tmp=$(mktemp -d)
      git -C "$REPO" archive "$BASE_REF" | tar -x -C "$tmp"
      SRC_DIR=$tmp INTARSIA_STAMP="head=$(git -C "$REPO" rev-parse --short=7 "$BASE_REF") code-tree=$(git -C "$REPO" rev-parse "$BASE_REF^{tree}" | cut -c1-12) dirty=no" "$REPO/scripts/cuda-decode/build.sh" tp2-base
      rm -rf "$tmp" ;;
    dev) "$REPO/scripts/cuda-decode/build.sh" tp2-dev ;;
    xdev|ydev|zdev) "$REPO/scripts/cuda-decode/build.sh" tp2-$t ;; # the working tree under another name (while dev is in use)
    *) echo "unknown tree $t"; exit 2 ;;
  esac
  # the binary goes to ~/intarsia-tp2/bin/<tree>-intarsia on both Sparks (same CUDA 13.0.3, same paths)
  OTHER=$([[ $B == spark1 ]] && echo spark2 || echo spark1)
  ssh -n -o BatchMode=yes "$B" "mkdir -p ~/intarsia-tp2/bin && cp ~/intarsia-decfuse/tp2-$t/target/acc/release/intarsia ~/intarsia-tp2/bin/$t-intarsia && ssh $OTHER mkdir -p intarsia-tp2/bin && scp -q ~/intarsia-tp2/bin/$t-intarsia $OTHER:intarsia-tp2/bin/$t-intarsia.new && ssh $OTHER mv intarsia-tp2/bin/$t-intarsia.new intarsia-tp2/bin/$t-intarsia"
  for h in spark1 spark2; do ssh -n -o BatchMode=yes "$h" "echo $h: \$(~/intarsia-tp2/bin/$t-intarsia --version 2>&1 | tr '\n' ' ')"; done
done
