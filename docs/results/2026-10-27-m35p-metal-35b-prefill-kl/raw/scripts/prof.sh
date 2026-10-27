#!/bin/bash
# Lane M35P: 35B Metal prefill profile (plain / kernel-mode / cb-mode) at 4k and 16k, in one fleetq slot.
# Usage: prof.sh BIN OUT_DIR PROMPT_DIR
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; OUT=$2; P=$3; mkdir -p "$OUT"; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
echo "job $FLEETQ_JOB $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l)" > "$OUT/meta.txt"
gen() { # stem ctx [env...]
  local stem=$1 ctx=$2; shift 2
  env "$@" $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos \
    --profile-every 0 --prompt "$(cat $P/prompt-$ctx.txt)" > "$OUT/$stem.out" 2> "$OUT/$stem.err"
  echo "$stem: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' "$OUT/$stem.err" | tail -1) rustc=$(pgrep -x rustc | wc -l)"
}
gen warm-4k 4096
gen plain-4k-a 4096
gen plain-4k-b 4096
gen kernel-4k 4096 INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-4k.jsonl
gen cb-4k 4096 INTARSIA_METAL_PROFILE=cb:$OUT/cb-4k.jsonl
gen plain-16k-a 16384
gen kernel-16k 16384 INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-16k.jsonl
gen cb-16k 16384 INTARSIA_METAL_PROFILE=cb:$OUT/cb-16k.jsonl
gen plain-16k-b 16384
