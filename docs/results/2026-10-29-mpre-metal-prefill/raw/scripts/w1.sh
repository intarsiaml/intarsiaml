#!/bin/bash
# MPRE window 1: attribution. Plain prefill (wall), kernel-mode trace (per-op GPU time), cb-mode trace (GPU idle gaps),
# NAX GEMM peak probe and flash attention probe, macmon 200 ms throughout. Inside ONE fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
L=/opt/code/intarsiaml/.worktrees/mpre/.lane; BIN=$L/intarsia-base; OUT=$L/w1; mkdir -p $OUT
P=/opt/code/intarsiaml/.worktrees/mpre/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
P1K=/opt/code/intarsiaml/.worktrees/mpre/docs/results/raw/2026-10-14-metal-phase-c/prompt-1k.txt
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/Models/Qwen3-8B
/Users/josef/bin/macmon pipe -i 200 > $OUT/macmon.jsonl 2> /dev/null &
MM=$!
echo "start $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)" > $OUT/meta.txt
pf() { case $1 in 1k) echo $P1K;; 4k) echo $P/prompt-4096.txt;; 16k) echo $P/prompt-16384.txt;; esac; }
gen() { local stem=$1 model=$2 ctx=$3; shift 3
  echo "$(date +%T.%N | cut -c1-12) begin $stem" >> $OUT/marks.txt
  env "$@" $BIN generate --model $model --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 \
    --prompt "$(cat $(pf $ctx))" > $OUT/$stem.out 2> $OUT/$stem.err
  echo "$(date +%T.%N | cut -c1-12) end $stem" >> $OUT/marks.txt
  echo "$stem: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $OUT/$stem.err | tail -1) $(grep -o 'prompt [0-9]* tokens' $OUT/$stem.err | tail -1)"; }
for m in 35 8; do
  eval MODEL=\$M$m
  gen m$m-warm $MODEL 4k A=1
  for c in 1k 4k 16k; do gen m$m-plain-$c-a $MODEL $c A=1; gen m$m-plain-$c-b $MODEL $c A=1; done
  for c in 1k 4k 16k; do gen m$m-cb-$c $MODEL $c INTARSIA_METAL_PROFILE=cb:$OUT/m$m-cb-$c.jsonl; done
  for c in 1k 4k 16k; do gen m$m-kernel-$c $MODEL $c INTARSIA_METAL_PROFILE=kernel:$OUT/m$m-kernel-$c.jsonl; done
done
echo "$(date +%T) gemm peak" >> $OUT/marks.txt
NAX_M=2048,4096 NAX_ROUNDS=3 $L/target/release/examples/gemm_nax bench > $OUT/gemm-peak.txt 2>&1
echo "$(date +%T) flash" >> $OUT/marks.txt
$L/target/release/examples/flash_bench --heads 32 --kv-heads 8 --dim 128 --chunk 2048 flash nax 0 2048 14336 > $OUT/flash-8b.txt 2>&1
$L/target/release/examples/flash_bench --heads 16 --kv-heads 2 --dim 256 --chunk 2048 flash 0 2048 14336 > $OUT/flash-35b.txt 2>&1
echo "end $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)" >> $OUT/meta.txt
kill $MM; gzip -f $OUT/*.jsonl
