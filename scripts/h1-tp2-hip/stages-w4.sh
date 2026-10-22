#!/usr/bin/env bash
# Lane H1 window 4 (ON tatooine, radiance down, called by window.sh): window 3's HIP GPU test run stopped at the comm
# binary (a cross-group deadlock of the parallel test harness under GPU_MAX_HW_QUEUES=1, now serialised) and so never ran
# the later test files. The whole HIP crate again with --no-fail-fast, then the comm tests three more times.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
ht() { # ht <tag> [cargo test args]
  local tag=$1; shift
  INTARSIA_REPO=$(T dev) INTARSIA_TARGET_VOL=intarsia-cargo-target-h1-dev-hiptest timeout 600 "$(T dev)/deploy/tatooine/dev.sh" \
    cargo test --release -p intarsia-backend-hip --no-fail-fast "$@" > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result' "$O/hiptest-$tag.log" | awk '{p+=$4; f+=$6; i+=$8} END {print p " passed, " f " failed, " i " ignored"}') $(grep -aE 'FAILED|panicked|HSA_STATUS' "$O/hiptest-$tag.log" | head -4 | tr '\n' '|')"
}
ht all
for i in 1 2 3; do ht comm-$i --test comm; done
log "done"
