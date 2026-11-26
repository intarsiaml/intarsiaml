#!/bin/bash
# lane ink-decroof job X (exclusive): the shipped tip binary against main's binary, across processes. ABBA x 8 per arm, each
# process prompts6.json (143 / 1k / 4k twice), --spec off, 256 greedy tokens; then per-dispatch traces at 143 and 4k, interleaved,
# 3 per arm.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/X; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
echo "jobX start $(date +%T) base $(md5 -q $S/bin/intarsia-base) tip $(md5 -q $S/bin/intarsia-tip2)"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
bin() { [ $1 = base ] && echo $S/bin/intarsia-base || echo $S/bin/intarsia-tip2; }
G="--model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --max-tokens 256 --ignore-eos"
STEP=cross; i=0
for r in 1 2 3 4 5 6 7 8; do
  if [ $((r % 2)) = 1 ]; then order="base tip"; else order="tip base"; fi
  for a in $order; do
    i=$((i + 1)); t0=$(date +%s)
    env X=1 $(bin $a) generate $G --prompt-ids $S/prompts6.json > $O/q${i}_$a.out 2> $O/q${i}_$a.err
    echo "q$i $a exit $? $(( $(date +%s) - t0 )) s | $(grep -h '^decode\[' $O/q${i}_$a.err | sed 's/decode//;s/ tokens, [0-9]* ms//' | tr '\n' ' ')"
  done
done
python3 -I $S/ab_proc2.py $O
STEP=traces
for r in 1 2 3; do
  if [ $((r % 2)) = 1 ]; then order="base tip"; else order="tip base"; fi
  for a in $order; do
    for k in 143 4k; do
      env X=1 INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/disp_${a}_${k}_$r.jsonl $(bin $a) generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --ignore-eos --max-tokens 33 --prompt-ids $S/p$k.json > /dev/null 2> $O/disp_${a}_${k}_$r.err
      echo "trace $a $k r$r: $(python3 -I $S/disp2.py $O/disp_${a}_${k}_$r.jsonl | grep -E 'decode steps|ink_attn_decode_bf16 |rel_sink_attn|ink_kv_prologue' | tr '\n' ' ')"
    done
  done
done
echo "jobX done $(date +%T)"
