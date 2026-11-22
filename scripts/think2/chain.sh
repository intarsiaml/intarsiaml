#!/usr/bin/env bash
# think2: the six measurement windows, each its own fleetq slot (so lane drill can interleave): BBH subset A1 B A2, then GSM8K A1 B A2.
cd "$(dirname "$0")/../.." || exit 1
export FLEETQ_OWNER=claude-overnight-think2
R=docs/results/2026-11-21-think2/raw
run() { n=$1; mem=$2; e=$3; shift 3; echo "$(date +%T) start $n" >> $R/chain.log
  fleetq run --host spark2 --quiet gpu,cpu --mem $mem --expect $e --label "think2 $n" -- "$@" >> $R/chain.log 2>&1; echo "$(date +%T) end $n rc=$?" >> $R/chain.log; }
run bbh-A1 2 90 bash scripts/think2/window.sh A bbh bbh-A1
run bbh-B 50 80 bash scripts/think2/window.sh B bbh bbh-B
run bbh-A2 2 90 bash scripts/think2/window.sh A bbh bbh-A2
run gsm8k-A1 2 140 bash scripts/think2/window.sh A gsm8k gsm8k-A1
run gsm8k-B 50 110 bash scripts/think2/window.sh B gsm8k gsm8k-B
run gsm8k-A2 2 140 bash scripts/think2/window.sh A gsm8k gsm8k-A2
echo "$(date +%T) chain done" >> $R/chain.log
