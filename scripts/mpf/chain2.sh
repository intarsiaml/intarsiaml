#!/usr/bin/env bash
# Lane mpf: gate sequence after the category window (scripts/mpf/chain.sh minus the window, with a trimmed CUDA mutant list so everything
# fits before the 23:00 fleet outage): G0 on CUDA, then the CUDA mutants (build + ids on the prompts that expose them).
set -u
R=docs/results/2026-10-27-mpf-macro-prefill-mtp/raw
step() { echo "== $* $(date +%T)"; }
until grep -q "done, time left" $R/s1/steps.log 2>/dev/null; do sleep 20; done
step g0; fleetq run --host spark1 --quiet gpu,cpu --mem 60 --expect 50 --label "mpf: G0 cuda" -- ssh spark1 'bash -s' < scripts/mpf/cuda-g0.sh > $R/cuda-g0.log 2>&1
for m in rollback-skip accept-off-by-one cuda-perrow-off macro-head-skip; do
  step "mutant build $m"; fleetq run --host spark1 --load cpu --expect 12 --label "mpf: CUDA mutant build $m" -- ssh spark1 "MODE=build bash -s $m" < scripts/mpf/mutant.sh > $R/mutant-build-$m.log 2>&1
  case $m in cuda-perrow-off) P="16k-0 16k-1 16k-2" ;; macro-head-skip) P="4k-0 34k-0" ;; *) P="short-0 4k-0 16k-0" ;; esac
  step "mutant ids $m"; fleetq run --host spark1 --mem 50 --expect 20 --label "mpf: CUDA mutant ids $m" -- ssh spark1 "MODE=ids OFF_FROM=\$HOME/intarsiaml-mpf/out bash -s $m $P" < scripts/mpf/mutant.sh > $R/mutant-ids-$m.log 2>&1
done
step done
