#!/usr/bin/env bash
# Lane hgv: identity windows (tp1, tp2; main, hgv, main again) then three interleaved perf rounds, each its own fleetq job (run FROM THE MAC).
set -u
R=$PWD/docs/results/2026-10-30-hgv-hip-small-gemv-grid/raw
H="--host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30"
for tp in tp1 tp2; do
  MODE=ident ROUND=$tp CAP=950 ARMS="$tp:base $tp:hgv $tp:base" OUT=$R/ident-$tp \
    fleetq run $H --expect 16 --label "HGV identity $tp (main, hgv, main)" -- scripts/hgv/window.sh > $R/fleetq-ident-$tp.log 2>&1
done
for r in 1 2 3; do
  case $r in 2) arms="tp2:base tp2:hgv tp1:hgv tp1:base";; *) arms="tp1:base tp1:hgv tp2:hgv tp2:base";; esac
  MODE=perf ROUND=$r CAP=1000 ARMS="$arms" OUT=$R/perf-r$r \
    fleetq run $H --expect 18 --label "HGV A/B perf round $r" -- scripts/hgv/window.sh > $R/fleetq-perf-r$r.log 2>&1
done
echo chain done
