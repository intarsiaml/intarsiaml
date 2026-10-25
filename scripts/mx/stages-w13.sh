#!/usr/bin/env bash
# Lane mx window 13 (card 0): 35B batch_cuda_real on HIP with --include-ignored, main 5241377b vs the PR head (build10),
# so the 3 failures of w6 (CUDA-only expectations) get a valid main result: base runs first, on its own test volume.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
log "w13 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
need 700 "mtest base" && mtest base batch35-base --test batch_cuda_real -- --test-threads=1 --include-ignored
need 700 "mtest dev" && mtest dev batch35-dev --test batch_cuda_real -- --test-threads=1 --include-ignored
for t in base dev; do grep -aE '^test .* \.\.\. ' "$O/mtest-batch35-$t.txt" | sort > "$O/batch35-$t.list"; done
same batch35-base.list batch35-dev.list "35B batch_cuda_real per-test outcomes main vs head"
log "w13 done; $(left) s left"
