#!/bin/bash
# lane ink-decroof job A2 (exclusive): does CONV_PRE add anything on top of BIAS_PRE? Rebuild the diag binary (arm E = BIAS_PRE
# alone), then within-process arms A (all off) / E (BIAS_PRE) / C (CONV_PRE + BIAS_PRE), 7 cycles AEC / CEA, 3 processes.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/A2; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
STEP=build
(cd $S/diag && CARGO_TARGET_DIR=$S/diag/target cargo build --release -p intarsia-cli --features metal) > $O/build.log 2>&1 || { echo "BUILD FAILED"; tail -20 $O/build.log; exit 1; }
cp $S/diag/target/release/intarsia $S/bin/intarsia-diag2; DIAG=$S/bin/intarsia-diag2
echo "jobA2 start $(date +%T) diag2 $(md5 -q $DIAG) tree $(git -C $S/diag rev-parse --short HEAD) dirty $(git -C $S/diag status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
G="--model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --max-tokens 256 --ignore-eos --show-ids"
STEP=within
for p in 1 2 3; do
  t0=$(date +%s)
  env INTARSIA_DIAG_ARMS=$(cat $S/arms_ab2.txt) $DIAG generate $G --prompt-ids $S/prompts_ab2.json > $O/p$p.out 2> $O/p$p.err
  echo "within p$p exit $? $(( $(date +%s) - t0 )) s, launch counts: $(grep -h '^cpre\[' $O/p$p.err | awk '{print $2 "/" $3 "/" $4}' | sort | uniq -c | tr '\n' ' ') distinct id lists: $(grep -h '^ids\[' $O/p$p.err | sed 's/^ids\[[^]]*\]: //' | sort -u | wc -l | tr -d ' ')"
done
META=$S/meta_ab2.json python3 -I $S/ab_within.py $O
STEP=tip
(cd $W && cargo build --release -p intarsia-cli --features metal) > $O/build_tip.log 2>&1 || { echo "TIP BUILD FAILED"; tail -20 $O/build_tip.log; exit 1; }
cp $W/target/release/intarsia $S/bin/intarsia-tip; TIP=$S/bin/intarsia-tip
echo "tip $(git -C $W rev-parse --short HEAD) dirty $(git -C $W status --porcelain --untracked-files=no | wc -l | tr -d ' ') md5 $(md5 -q $TIP)"
G2="--model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --max-tokens 256 --ignore-eos --show-ids --prompt-ids $S/prompts.json"
for m in off spec; do
  if [ $m = off ]; then x="--spec off"; else x=""; fi
  env X=1 $TIP generate $G2 $x > $O/tip_$m.out 2> $O/tip_$m.err
  for i in 1 2 3; do
    a=$(grep -h '^ids\[' $S/G/base_$m.err | sed -n "${i}p" | sed 's/^ids\[[^]]*\]: //' | md5 -q); b=$(grep -h '^ids\[' $O/tip_$m.err | sed -n "${i}p" | sed 's/^ids\[[^]]*\]: //' | md5 -q)
    echo "TIP IDS $m prompt $i: base (#11936) $a tip $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER ) | $(grep -h '^decode\[' $O/tip_$m.err | sed -n "${i}p")"
  done
done
(cd $W && cargo test --release -p intarsia-cli --test docs_env_vars) > $O/docs_env_vars.log 2>&1; echo "docs_env_vars exit $?: $(grep -h 'test result' $O/docs_env_vars.log)"
(cd $W && cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact) > $O/t_attn.log 2>&1; echo "tip inkling2_attn + rows_exact exit $?: $(grep -h 'test result' $O/t_attn.log | tr '\n' ' ')"
echo "jobA2 done $(date +%T)"
