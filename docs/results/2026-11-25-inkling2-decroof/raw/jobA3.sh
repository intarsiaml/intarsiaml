#!/bin/bash
# lane ink-decroof job A3 (exclusive): the fused decode attention above its 4096-key rule. Diag3 binary, within-process arms
# C (defaults: CONV_PRE + BIAS_PRE, the rule) / F (C with InkAttnDecode forced on every visit), 4k / 8k / 16k, 5 cycles, 2 processes.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/A3; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
STEP=build
(cd $S/diag3 && CARGO_TARGET_DIR=$S/diag3/target cargo build --release -p intarsia-cli --features metal) > $O/build.log 2>&1 || { echo "BUILD FAILED"; tail -20 $O/build.log; exit 1; }
cp $S/diag3/target/release/intarsia $S/bin/intarsia-diag3; DIAG=$S/bin/intarsia-diag3
echo "jobA3 start $(date +%T) diag3 $(md5 -q $DIAG) tree $(git -C $S/diag3 rev-parse --short HEAD) dirty $(git -C $S/diag3 status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
G="--model $CK --backend metal --max-seq 20480 --temperature 0 --profile-every 0 --spec off --max-tokens 256 --ignore-eos --show-ids"
STEP=within
for p in 1 2; do
  t0=$(date +%s)
  env INTARSIA_DIAG_ARMS=$(cat $S/arms_ab3.txt) $DIAG generate $G --prompt-ids $S/prompts_ab3.json > $O/p$p.out 2> $O/p$p.err
  echo "within p$p exit $? $(( $(date +%s) - t0 )) s, launch counts: $(grep -h '^cpre\[' $O/p$p.err | awk '{print $2 "/" $3 "/" $4}' | sort | uniq -c | tr '\n' ' ') distinct id lists: $(grep -h '^ids\[' $O/p$p.err | sed 's/^ids\[[^]]*\]: //' | sort -u | wc -l | tr -d ' ')"
done
META=$S/meta_ab3.json python3 -I $S/ab_within.py $O
echo "jobA3 done $(date +%T)"
