#!/usr/bin/env bash
# Stage: first speed numbers of Inkling-2-Mini on one R9700 (`intarsia generate`, greedy, spec off, graphs as the default).
# Usage: stage_speed.sh [decode] [prefill] [optable] [rocprof]
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip
R=$REPO/docs/results/2026-11-25-inkling2-hip/raw
M=/models/Inkling-2-Mini
I=/target/release/intarsia
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos"
line() { grep -aE "finish=|steps=|prefill chunk|Error" | cut -c1-400; }
for s in "$@"; do
  echo "==== $s ($(date '+%T'))"
  case $s in
    decode)
      for rep in 1 2; do
        echo "-- short, 256 tokens, rep $rep"; bash "$R/run.sh" $I $G --profile-every 0 --max-tokens 256 -p "$PS" 2>&1 | line
        echo "-- 4k prompt, 256 tokens, rep $rep"; bash "$R/run.sh" $I $G --profile-every 0 --max-tokens 256 -p "$P4" 2>&1 | line
      done ;;
    prefill)
      for ch in 256 512 1024 2048; do
        for rep in 1 2; do
          echo "-- 4k prompt, chunk $ch, rep $rep"; bash "$R/run.sh" $I $G --profile-every 0 --max-tokens 8 --prefill-chunk $ch -p "$P4" 2>&1 | line
        done
      done ;;
    optable)
      echo "-- eager decode op table (short, 64 tokens, every step profiled)"
      bash "$R/run.sh" $I $G --no-graphs --profile-every 1 --op-table --max-tokens 64 -p "$PS" 2>&1 | tail -45 | cut -c1-200
      echo "-- prefill op table (4k prompt, chunk 256, 1 token)"
      bash "$R/run.sh" $I $G --no-graphs --profile-every 1 --op-table --max-tokens 1 -p "$P4" 2>&1 | tail -45 | cut -c1-200 ;;
    rocprof)
      rm -rf "$REPO/logs/rocprof"; mkdir -p "$REPO/logs/rocprof"
      bash "$R/run.sh" rocprofv3 --kernel-trace --output-format csv -d /work/logs/rocprof -o run -- $I $G --profile-every 0 --max-tokens 24 -p "$PS" 2>&1 | tail -6 | cut -c1-300
      csv=$(find "$REPO/logs/rocprof" -name '*kernel_trace.csv' | head -1)
      echo "kernel trace: ${csv:-none}"
      [ -n "$csv" ] && gzip -kf "$csv" && python3 "$R/ktrace.py" "$csv" ;;
    *) echo "unknown stage $s" ;;
  esac
done
exit 0
