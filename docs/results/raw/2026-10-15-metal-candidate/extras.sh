#!/bin/zsh
# metal-candidate extras, ONE fleetq slot: (1) baseline self-check at chunk 256 with --allow-config-mismatch (the recorded
# baselines are chunk 2048, the tool refuses otherwise); (2) decode ms/step cross-check of pre (intarsia-pre = spike/metal 766c723
# build, same decode as a710f35) vs base (6271eec) vs cand, 4k and short, palindromic order, to see whether the slower 4k decode in
# the gates run (29.6 ms vs 28.0 ms earlier today) is drift or code.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-a0a61a602516c3238
R=$W/docs/results/raw/2026-10-15-metal-candidate/extras
PB=$W/docs/results/raw/2026-10-14-metal-phase-b
BASE=$S/intarsia-base
CAND=$S/intarsia-cand
PRE=$S/intarsia-pre
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
SB=$W/golden/baselines
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 20 --label "metal-candidate extras (c256 selfcheck, decode cross-check)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
echo "== 1. selfcheck at chunk 256, --allow-config-mismatch"
$CAND accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 --baseline $SB/qwen3-8b-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 --allow-config-mismatch --prefill-chunk 256 > $R/selfcheck-short-c256.out 2> $R/selfcheck-short-c256.err; echo "short exit $?"
grep -E "verdict" $R/selfcheck-short-c256.out | cut -c1-300
$CAND accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 64 --max-seq 8192 --baseline $SB/qwen3-8b-long-fp32.metal.baseline.json --noise-envelope $SB/qwen3-8b-long-fp32.metal.envelope.json --allow-same-code --pre-change e8be186 --allow-config-mismatch --prefill-chunk 256 > $R/selfcheck-long-c256.out 2> $R/selfcheck-long-c256.err; echo "long exit $?"
grep -E "verdict" $R/selfcheck-long-c256.out | cut -c1-300
echo "== 2. decode cross-check"
for P in 4k short; do
  for B in pre base cand cand base pre; do
    case $B in pre) BIN=$PRE;; base) BIN=$BASE;; cand) BIN=$CAND;; esac
    N=$(ls $R | grep -c "^dec-$B-$P-.*jsonl"); F=$R/dec-$B-$P-$((N+1))
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: period/busy/idle/disp $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1)"
  done
done
echo done
