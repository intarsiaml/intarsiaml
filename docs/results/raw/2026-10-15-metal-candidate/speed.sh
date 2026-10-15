#!/bin/zsh
# metal-candidate plain-prefill timing, ONE fleetq slot. generate --max-tokens 1, prefill ms from stderr.
# Arms: base = spike/metal 6271eec as shipped (chunk 256); cand = metal-candidate as shipped (chunk 2048);
#       cand256 = candidate at --prefill-chunk 256 (F1 without the P3 chunk); base2048 = spike/metal at --prefill-chunk 2048 (P3 chunk without F1).
# Order per prompt: warm-up (cand), then two palindromic rounds base cand cand256 base2048 | base2048 cand256 cand base  (4 runs per arm).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238
R=$W/docs/results/raw/2026-10-15-metal-candidate/speed
RAW=$W/docs/results/raw
P1K=$RAW/2026-10-14-metal-phase-c/prompt-1k.txt
P4K=$RAW/2026-10-14-metal-phase-c/prompt-4k.txt
P16K=$RAW/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
BASE=$S/intarsia-base
CAND=$S/intarsia-cand
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 40 --label "metal-candidate plain prefill ABBA (base vs candidate)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $R/zoom.txt 2>&1; echo "zoom: $(cat $R/zoom.txt)"
echo "idle start: $(idle3)" | tee $R/idle.txt
run() { # arm prompt-file maxseq tag
  local BIN X
  case $1 in base) BIN=$BASE; X=();; cand) BIN=$CAND; X=();; cand256) BIN=$CAND; X=(--prefill-chunk 256);; base2048) BIN=$BASE; X=(--prefill-chunk 2048);; esac
  $BIN generate --model $M --backend metal --max-seq $3 --temperature 0 --max-tokens 1 --profile-every 0 $X --prompt "$(cat $2)" > $R/$4.out 2> $R/$4.err
  echo "$4: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$4.err | tail -1)"
}
run cand $P4K 8192 warm
for P in 1k 4k 16k; do
  case $P in 1k) F=$P1K; MS=8192;; 4k) F=$P4K; MS=8192;; 16k) F=$P16K; MS=18432;; esac
  echo "idle $P: $(idle3)" >> $R/idle.txt
  n=0
  for ORDER in "base cand cand256 base2048" "base2048 cand256 cand base" "base cand cand256 base2048" "base2048 cand256 cand base"; do
    n=$((n+1))
    for A in ${=ORDER}; do run $A $F $MS $P-$A-o$n; done
  done
done
echo "idle end: $(idle3)" | tee -a $R/idle.txt
echo done
