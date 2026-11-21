#!/usr/bin/env bash
# ab35e: run all windows in order, each under its own fleetq slot.
cd "$(dirname "$0")/../.." || exit 1
export FLEETQ_OWNER=claude-overnight-ab35e
R=docs/results/2026-11-20-ab35e-cutover-scoreboard/raw
run() { n=$1; e=$2; shift 2; echo "$(date +%T) start $n" >> $R/chain.log
  fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect $e --label "ab35e $n" -- "$@" >> $R/chain.log 2>&1; echo "$(date +%T) end $n rc=$?" >> $R/chain.log; }
run w1 25 bash scripts/ab35e/window.sh w1 tools replay
run w2 30 bash scripts/ab35e/window.sh w2 speed sampled
run w3 20 bash scripts/ab35e/window.sh w3 gsm8k
BARGS="--kv-dtype fp8_e4m3" BLABEL=fp8 run w4 10 bash scripts/ab35e/window.sh w4 gsm8kb
CTL_SERVE="--max-batch 2 --prefix-store on" run w5 20 bash scripts/ab35e/window-ctl.sh w5 replay1
echo "$(date +%T) chain done" >> $R/chain.log
