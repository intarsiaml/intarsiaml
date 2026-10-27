#!/usr/bin/env bash
# Lane mpf: the gate sequence after gate 1 (ids), one fleetq job per step, from the Mac, in the order of risk:
#   speed (long context, 3 rounds), soak (40 requests), per-category speed (3 rounds), G0 on CUDA, CUDA mutants (build + ids).
# Run from the repo root: scripts/mpf/chain.sh   (logs under docs/results/2026-10-27-mpf-macro-prefill-mtp/raw/)
set -u
R=docs/results/2026-10-27-mpf-macro-prefill-mtp/raw; mkdir -p $R
step() { echo "== $* $(date +%T)"; }
step longctx; fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 --label "mpf: gate4 long context off vs mtp x3" -- env REPS=5 OUT=$PWD/$R/longctx scripts/mpf/longctx.sh 3 > $R/longctx.log 2>&1
step soak; fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "mpf: gate2 soak 40 (off vs mtp2)" -- env W=soak CATS=none ARMS="off mtp2" ROUNDS=1 SOAK=40 SOAK_ARMS=mtp2 SOAK_REF=4 CAP=2400 scripts/mpf/window.sh > $R/window-soak.log 2>&1
step window; fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 75 --label "mpf: gate4 categories off vs mtp2 x3" -- env W=s1 ARMS="off mtp2" ROUNDS=3 CAP=5400 scripts/mpf/window.sh > $R/window-s1.log 2>&1
step g0; fleetq run --host spark1 --quiet gpu,cpu --mem 60 --expect 50 --label "mpf: G0 cuda" -- ssh spark1 'bash -s' < scripts/mpf/cuda-g0.sh > $R/cuda-g0.log 2>&1
for m in rollback-skip accept-off-by-one cuda-perrow-off macro-head-skip head-kvonly-pos; do
  step "mutant build $m"; fleetq run --host spark1 --load cpu --expect 12 --label "mpf: CUDA mutant build $m" -- ssh spark1 "MODE=build bash -s $m" < scripts/mpf/mutant.sh > $R/mutant-build-$m.log 2>&1
  case $m in cuda-perrow-off) P="16k-0 16k-1 16k-2 16k-3 16k-4 16k-5" ;; macro-head-skip|head-kvonly-pos) P="4k-0 4k-1 16k-0 34k-0" ;; *) P="short-0 short-1 4k-0 4k-1 16k-0" ;; esac
  step "mutant ids $m"; fleetq run --host spark1 --mem 50 --expect 25 --label "mpf: CUDA mutant ids $m" -- ssh spark1 "MODE=ids OFF_FROM=\$HOME/intarsiaml-mpf/out bash -s $m $P" < scripts/mpf/mutant.sh > $R/mutant-ids-$m.log 2>&1
done
step done
