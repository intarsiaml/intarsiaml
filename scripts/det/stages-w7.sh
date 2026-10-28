#!/usr/bin/env bash
# Lane det w7 (ON tatooine, radiance down): is it the P2P all-reduce? w5: every event is a small logit shift (it flips the run's lowest-margin
# tokens: 7, 17, 20, 27, 38), tp=2 only (tp=1 0/200), unchanged by poisoned allocations or the capture and peer syncs. A one-sided stale read
# in a non-last row would still leave rank 0 == rank 1 on the last row (the next all-reduce sums it into both), so the per-row tap can tell.
# Arms, all --spec mtp fresh processes: per-row tap x200 (16 tokens), INTARSIA_HIP_AR_DRAIN=5 (~25 us before the peer's data is read) x200,
# --comm host (host-staged all-reduce, no P2P kernel) x150.
set -uo pipefail
export O=${1:?out dir}
source "$(dirname "$0")/lib.sh"
clear w7 || exit 2
M="--spec mtp --spec-k 2"
TAP=1 MT=16 loop tap 0,1 200 --tp 2 --devices 0,1 $M
E="INTARSIA_HIP_AR_DRAIN=5" loop drain5 0,1 200 --tp 2 --devices 0,1 $M
loop host 0,1 150 --tp 2 --devices 0,1 --comm host $M
log "w7 done"
