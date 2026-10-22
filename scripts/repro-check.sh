#!/usr/bin/env bash
# Prove the release binary is reproducible: export HEAD (or --ref) twice into different directories, build each with
# scripts/repro-build.sh, compare sha256. Exit 0 if identical, 1 if not.
# Usage: scripts/repro-check.sh [--ref <git ref>] [--features <list>] [--keep]
# Default is the CPU-only build (no GPU toolchain needed). On a fleet box run it inside a fleetq slot.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
REF=HEAD FEATURES="" KEEP=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --ref) REF=$2; shift 2 ;;
    --features) FEATURES=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "repro-check: unknown arg $1" >&2; exit 2 ;;
  esac
done
SHA=$(git -C "$REPO" rev-parse "$REF")
SDE=$(git -C "$REPO" log -1 --format=%ct "$SHA")
W=$(mktemp -d); [[ $KEEP == 1 ]] || trap 'rm -rf "$W"' EXIT
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
declare -a SUMS
for d in first/a second/much/deeper/b; do
  mkdir -p "$W/$d"
  git -C "$REPO" archive "$SHA" | tar -x -C "$W/$d"
  echo "repro-check: building $SHA in $W/$d ${FEATURES:+(features: $FEATURES)}"
  # Same inputs except the directory; separate target dir per tree, no shared cargo target.
  (cd "$W/$d" && env -u CARGO_TARGET_DIR SOURCE_DATE_EPOCH="$SDE" INTARSIA_BUILD_STAMP="head=${SHA:0:7} repro-check" \
     scripts/repro-build.sh ${FEATURES:+--features "$FEATURES"} >build.log 2>&1) || { tail -30 "$W/$d/build.log"; echo "repro-check: BUILD FAILED"; exit 2; }
  SUMS+=("$(sha "$W/$d/target/release/intarsia")")
done
echo "repro-check: ${SUMS[0]}  $W/first"
echo "repro-check: ${SUMS[1]}  $W/second"
if [[ ${SUMS[0]} == "${SUMS[1]}" ]]; then echo "repro-check: IDENTICAL"; else echo "repro-check: DIFFER"; [[ $KEEP == 1 ]] || echo "(rerun with --keep to inspect the trees)"; exit 1; fi
