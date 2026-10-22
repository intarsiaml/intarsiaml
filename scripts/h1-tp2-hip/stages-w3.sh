#!/usr/bin/env bash
# Lane H1 window 3 (ON tatooine, radiance down, called by window.sh): gates on the dev build that need the GPUs but no
# timing, and the mutants.
#  1. the HIP crate's GPU tests (prebuilt by build.sh HIPTEST=1; incl. the graph-captured collectives and the tp=2 shard
#     shapes) and `kernels test --backend hip` on the 35B config: 0 FAIL;
#  2. mutants (trees ~/intarsiaml-h1-mut-<name>, scripts/h1-tp2-hip/mutate.py): 256 greedy ids of the 35B at tp=2 with
#     graphs against base tp=2 eager (window 1), plus rank 1's graph counters; tp-fold-outproj runs with fusion on.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
INTARSIA_REPO=$(T dev) INTARSIA_TARGET_VOL=intarsia-cargo-target-h1-dev-hiptest timeout 420 "$(T dev)/deploy/tatooine/dev.sh" \
  cargo test --release -p intarsia-backend-hip > "$O/hiptest-all.log" 2>&1 < /dev/null
log "hiptest all exit $?: $(grep -aE '^test result' "$O/hiptest-all.log" | awk '{p+=$4; f+=$6} END {print p " passed, " f " failed"}') $(grep -aE 'FAILED|panicked' "$O/hiptest-all.log" | head -4 | tr '\n' '|')"
run dev 0 420 kernels-test kernels test --backend hip --device 0 --model $M35
log "kernels test: $(grep -aiE 'fail|pass' "$O/kernels-test.out" | tail -3 | tr '\n' '|')"
REF=$HOME/h1/out/w1/gen-base-35b-tp2-short.ids
for m in no-peer-upload drop-captured-allreduce peer-ignores-flag tp-fold-outproj; do
  e=""; [[ $m == tp-fold-outproj ]] && e="INTARSIA_TP_FUSION=1"
  E="$e" gen mut-$m 0,1 mut-$m-35b-tp2-short $M35 short $TP2
  if cmp -s "$O/gen-mut-$m-35b-tp2-short.ids" "$REF"; then v="ids SAME as base tp=2 eager"; else v="ids DIFFER from base tp=2 eager"; fi
  log "MUTANT $m: $v; $(grep -ahE 'tensor-parallel rank 1: decode graph|^Error|trap|Aborted|core dumped' "$O/gen-mut-$m-35b-tp2-short.err" | head -2 | tr '\n' '|')"
done
log "done"
