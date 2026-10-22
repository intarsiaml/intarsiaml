#!/usr/bin/env bash
# fx6: queue passes 2..4 of the R9700 interleaved comparison (pass 1 was ABCD). One radiance window each, run back to back.
set -u
cd "$(dirname "$0")/../.."
R=$PWD/docs/results/2026-10-22-fx6-standings/raw/hip
run() { # <pass> <arms> <mmlu>
  mkdir -p $R/$1
  fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --expect 20 --label "fx6 R9700 pass $1 ($2)" -- \
    env OUT=$R/$1 ARMS="$2" PASS=$1 MMLU=$3 bash scripts/standings-fx6/hip-window.sh >$R/$1.fleetq.out 2>&1
}
run p2 "llama-hip hipfire llama-vk intarsia" 0
run p3 "llama-vk intarsia llama-hip hipfire llama-vk4" 0
run p4 "intarsia llama-vk" 1
