#!/usr/bin/env bash
# Lane mpf: the shipped default end to end. Build the tree with the recipe key set (target-flip), then ids for all 21 prompts, off arm
# reused from the gate-1 run, auto arm = no --spec (must print the default line and the spec: stats).
set -u
R=docs/results/2026-10-27-mpf-macro-prefill-mtp/raw
N="short-0 short-1 short-2 short-3 short-4 short-5 4k-0 4k-1 4k-2 4k-3 4k-4 4k-5 16k-0 16k-1 16k-2 16k-3 16k-4 16k-5 34k-0 34k-1 34k-2"
echo "== build flip $(date +%T)"; fleetq run --host spark1 --load cpu --expect 12 --label "mpf: build flip (default on)" -- ssh spark1 'TREES=flip bash -s' < scripts/mpf/spark1.sh > $R/build-flip.log 2>&1
echo "== ids auto $(date +%T)"; fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 50 --label "mpf: gate1 ids 21 prompts off vs default (no flag)" -- ssh spark1 "BIN=\$HOME/intarsiaml-mpf/target-flip/release/intarsia O=\$HOME/intarsiaml-mpf/out-auto OFF_FROM=\$HOME/intarsiaml-mpf/out ARMS='off auto' bash -s $N" < scripts/mpf/ids.sh > $R/ids-auto.log 2>&1
echo "== done $(date +%T)"
