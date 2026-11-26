#!/usr/bin/env bash
# GPU stage (lane ink-hip-attn): where the +19 ms of short-prompt prefill on #518 (81 -> 100 ms) goes. Prebuilt binaries:
# main 4fc9901d4 (~/intarsiaml-ink-hip-speed-main) and the #518 tree (~/intarsiaml-ink-hip-speed). Each line: one process.
#   rep1 / rep3: one request vs the third request in one process (generate --repeat 3 prints the last request's timing)
#   nograph: INTARSIA_SOLO_GRAPHS=0 (eager decode, no capture)
#   nolazy: HIP_ENABLE_DEFERRED_LOADING=0 (all code objects loaded at start-up, not on a kernel's first launch)
#   revert: branch with INTARSIA_INK_MIX_FAST=0 INTARSIA_INK_ATTN_FUSED=0 (attn + mix fast forms off)
set -uo pipefail
A=$HOME/intarsiaml-ink-hip-attn/docs/results/2026-11-25-inkling2-hip-attn/raw
B=$HOME/intarsiaml-ink-hip-speed
M0=$HOME/intarsiaml-ink-hip-speed-main
I=/target/release/intarsia
PS="$(cat "$B/scripts/cmx/ids-prompts/short-0.txt")"
G="generate -m /models/Inkling-2-Mini --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0 --max-tokens 32"
run() {  # label build extra-env... -- extra-args...
  local lab=$1 b=$2; shift 2
  local envs=() args=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  args=("$@")
  local tree vol
  if [ "$b" = main ]; then tree=$M0; vol=intarsia-cargo-target-ink-hip-speed-main; else tree=$B; vol=intarsia-cargo-target-ink-hip-attn; fi
  local t0; t0=$(date +%s.%N)
  local o; o=$(bash "$A/run_on.sh" "$tree" "$vol" ink-hip-attn-run env "${envs[@]}" $I $G "${args[@]}" -p "$PS" 2>&1)
  local t1; t1=$(date +%s.%N)
  echo "PF $lab $b | $(echo "$o" | grep -aoE 'prefill [0-9]+ ms' | tail -1) | $(echo "$o" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | tail -1) | $(echo "$o" | grep -aoE 'graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | tail -1) | process $(echo "$t1 - $t0" | bc) s"
  echo "$o" | grep -aE 'panicked|Error' | head -2
}
for b in main branch; do run rep1 $b X=1; done
for b in main branch; do run rep3 $b X=1 -- --repeat 3; done
for b in main branch; do run nograph $b INTARSIA_SOLO_GRAPHS=0; done
for b in main branch; do run nolazy $b HIP_ENABLE_DEFERRED_LOADING=0; done
run revert branch INTARSIA_INK_MIX_FAST=0 INTARSIA_INK_ATTN_FUSED=0
run rep1b main X=1
run rep1b branch X=1
echo "[stage] done $(date '+%T')"
