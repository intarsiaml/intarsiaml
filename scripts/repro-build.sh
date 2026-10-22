#!/usr/bin/env bash
# Build the release `intarsia` binary so the same git sha gives the same bytes in any directory (and, with the same
# toolchain, on any host). Usage: scripts/repro-build.sh [cargo build args...]   e.g.  --features metal
# Adds to `cargo build --release --locked -p intarsia-cli`:
#   * --remap-path-prefix for the source tree, the cargo target dir, CARGO_HOME and the rustc sysroot (the paths rustc and
#     build scripts would otherwise embed in panic messages, debug info and `file!()`),
#   * SOURCE_DATE_EPOCH (commit time of HEAD, or the value you export; 0 when neither is known) and ZERO_AR_DATE for the
#     archivers/linkers that honour them,
#   * a fixed locale and timezone.
# Provenance stays: pass INTARSIA_BUILD_STAMP (deploy.sh does) to record git sha / tree hash; it is an input, not a clock.
# Print the result with `scripts/repro-build.sh --print-sha` after a build (sha256 of target/release/intarsia).
# docs/reproducing-results.md has the details and what is still host specific.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
TARGET=${CARGO_TARGET_DIR:-$ROOT/target}
if [[ ${1:-} == --print-sha ]]; then sha "$TARGET/release/intarsia"; exit 0; fi
mkdir -p "$TARGET"; TARGET=$(cd "$TARGET" && pwd -P)

if [[ -z ${SOURCE_DATE_EPOCH:-} ]]; then
  SOURCE_DATE_EPOCH=$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || true)
  SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-0}
fi
export SOURCE_DATE_EPOCH ZERO_AR_DATE=1 LC_ALL=C TZ=UTC

CARGO_HOME_DIR=${CARGO_HOME:-$HOME/.cargo}
SYSROOT=$(rustc --print sysroot)
# Later mappings win, so the most specific prefix goes last (the target dir usually sits inside the tree).
MAPS=("--remap-path-prefix=$ROOT=/build/intarsiaml")
[[ $TARGET == "$ROOT"/* ]] || MAPS+=("--remap-path-prefix=$TARGET=/build/target")
MAPS+=("--remap-path-prefix=$CARGO_HOME_DIR=/build/cargo" "--remap-path-prefix=$SYSROOT=/build/rustc")
# RUSTFLAGS from the caller are kept (cargo ignores [build].rustflags when RUSTFLAGS is set, so we always set it here).
export RUSTFLAGS="${RUSTFLAGS:-} ${MAPS[*]}"
# Nothing in this build should depend on the incremental cache or a developer's own cargo config.
export CARGO_INCREMENTAL=0

cargo build --release --locked -p intarsia-cli "$@"
echo "repro-build: sha256 $(sha "$TARGET/release/intarsia")  SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH  rustc=$(rustc --version)"
