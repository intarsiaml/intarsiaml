#!/usr/bin/env bash
# Lane a4k-restore (ON tatooine, CPU only): the origin/main tree ~/intarsiaml-a4k-restore-base (copied by the Mac from `git archive`),
# its hybrid_reuse_real test binary, in its own cargo target volume.
set -u
. "$(dirname "$0")/lib.sh"
export INTARSIA_REPO=$HOME/intarsiaml-a4k-restore-base INTARSIA_TARGET_VOL=intarsia-cargo-target-a4k-restore-base
"$INTARSIA_REPO/deploy/tatooine/dev.sh" cargo test --release -p intarsia-models --test hybrid_reuse_real --no-run < /dev/null; echo "BUILD base hybrid_reuse_real exit $?"
