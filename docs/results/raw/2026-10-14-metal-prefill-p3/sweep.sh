#!/bin/zsh
# P3 chunk sweep: plain prefill (generate --max-tokens 1) at --prefill-chunk 256/512/1024/2048, prompts 1k/4k/16k, 3 reps in
# order fwd, rev, fwd (drift shows up as spread, not as a chunk effect). One discarded warm-up. Peak memory from /usr/bin/time -l.
# Binary = spike/metal 2e1fdb1 build (default chunk 256, no [tuning.metal]); the sweep sets the chunk by flag.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9
R=$W/docs/results/raw/2026-10-14-metal-prefill-p3/sweep
RAW=/opt/code/intarsiaml/docs/results/raw
P1K=$RAW/2026-10-14-metal-phase-c/prompt-1k.txt
P4K=$RAW/2026-10-14-metal-phase-c/prompt-4k.txt
P16K=$RAW/2026-10-14-metal-phase-b/b2a/prompt-16k.txt
BIN=$S/p3/intarsia-base
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 30 --label "metal P3: chunk sweep (plain prefill 256/512/1024/2048)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
pgrep -fl zoom.us > $R/zoom.txt 2>&1; echo "zoom: $(cat $R/zoom.txt)"
echo "idle start: $(idle3)" | tee $R/idle.txt
run() { # prompt-file maxseq chunk tag
  FLEETQ_JOB=$J /usr/bin/time -l $BIN generate --model $M --backend metal --max-seq $2 --temperature 0 --max-tokens 1 --profile-every 0 \
    --prefill-chunk $3 --prompt "$(cat $1)" > $R/$4.out 2> $R/$4.err
  echo "$4: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $R/$4.err | tail -1) | $(grep -E 'peak memory footprint|maximum resident' $R/$4.err | tr -s ' ' | tr '\n' ';')"
}
run $P4K 8192 256 warm
for P in 1k 4k 16k; do
  case $P in 1k) F=$P1K; MS=8192;; 4k) F=$P4K; MS=8192;; 16k) F=$P16K; MS=18432;; esac
  for rep in 1 2 3; do
    if [ $rep = 2 ]; then ORDER=(2048 1024 512 256); else ORDER=(256 512 1024 2048); fi
    echo "idle $P rep$rep: $(idle3)" >> $R/idle.txt
    for C in $ORDER; do run $F $MS $C $P-c$C-r$rep; done
  done
done
# memory-vs-chunk view at the long max-seq with the 4k prompt
for C in 256 2048; do run $P4K 18432 $C 4k-ms18432-c$C; done
echo "idle end: $(idle3)" | tee -a $R/idle.txt
