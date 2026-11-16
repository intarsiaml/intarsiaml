#!/usr/bin/env bash
# Lane hip-reuse (ON tatooine, CPU only): the stamped release CLI and the hybrid_reuse_real test binary.
set -u
. "$(dirname "$0")/lib.sh"
"$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
"$D" cargo test --release -p intarsia-models --test hybrid_reuse_real --no-run < /dev/null; echo "BUILD hybrid_reuse_real exit $?"
