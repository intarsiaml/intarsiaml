#!/usr/bin/env bash
# Lane a4k-restore (ON tatooine, CPU only): the stamped release CLI and the HIP / models test binaries.
set -u
. "$(dirname "$0")/lib.sh"
"$D" env INTARSIA_BUILD_STAMP="$STAMP" cargo build --release -p intarsia-cli --features hip < /dev/null; echo "BUILD cli exit $?"
"$D" cargo test --release -p intarsia-backend-hip --tests --no-run < /dev/null; echo "BUILD hip tests exit $?"
"$D" cargo test --release -p intarsia-models --test batch_recurrent --test mtp --test batch_cuda_real --test hybrid_reuse_real --no-run < /dev/null; echo "BUILD models tests exit $?"
