#!/bin/bash
# lane ink-decroof job T (exclusive): tip gates after the fused-threshold change. Build the tip; greedy ids vs main's base binary
# at 143 / 1k / 4k (--spec off and spec default) and at 8k / 16k (--spec off, base binary run here); Metal + models Inkling tests;
# docs_env_vars.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/T; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BASE=$S/bin/intarsia-base
STEP=build
(cd $W && cargo build --release -p intarsia-cli --features metal) > $O/build.log 2>&1 || { echo "TIP BUILD FAILED"; tail -30 $O/build.log; exit 1; }
cp $W/target/release/intarsia $S/bin/intarsia-tip2; TIP=$S/bin/intarsia-tip2
echo "jobT start $(date +%T) tip $(git -C $W rev-parse --short HEAD) dirty $(git -C $W status --porcelain --untracked-files=no | wc -l | tr -d ' ') md5 $(md5 -q $TIP) base $(md5 -q $BASE)"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
idl() { grep -h '^ids\[' $1 | sed -n "$2p" | sed 's/^ids\[[^]]*\]: //' | md5 -q; }
G="--model $CK --backend metal --max-seq 20480 --temperature 0 --profile-every 0 --max-tokens 256 --ignore-eos --show-ids"
STEP=ids
for m in off spec; do
  if [ $m = off ]; then x="--spec off"; else x=""; fi
  env X=1 $TIP generate $G $x --prompt-ids $S/prompts.json > $O/tip_$m.out 2> $O/tip_$m.err
  for i in 1 2 3; do a=$(idl $S/G/base_$m.err $i); b=$(idl $O/tip_$m.err $i)
    echo "IDS $m prompt $i: base $a tip $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER ) | $(grep -h '^decode\[' $O/tip_$m.err | sed -n "${i}p")"; done
done
env X=1 $BASE generate $G --spec off --prompt-ids $S/prompts_long.json > $O/base_long.out 2> $O/base_long.err
env X=1 $TIP generate $G --spec off --prompt-ids $S/prompts_long.json > $O/tip_long.out 2> $O/tip_long.err
for i in 1 2; do a=$(idl $O/base_long.err $i); b=$(idl $O/tip_long.err $i)
  echo "IDS long prompt $i ($( [ $i = 1 ] && echo 8216 || echo 16432 ) tokens): base $a tip $b: $( [ $a = $b ] && echo EQUAL || echo DIFFER ) | base $(grep -h '^decode\[' $O/base_long.err | sed -n "${i}p") | tip $(grep -h '^decode\[' $O/tip_long.err | sed -n "${i}p")"; done
STEP=tests
t() { tag=$1; shift; local t0=$(date +%s); env "$@" > $O/t_$tag.log 2>&1; rc=$?
  echo "== tests $tag exit $rc $(( $(date +%s) - t0 )) s: $(grep -h 'test result' $O/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
  grep -h "panicked\|FAILED" $O/t_$tag.log | head -8; }
cd $W
t metal_ink X=1 cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact --test inkmix --test inkctrl --test inkload
t models_metal INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch -- --test-threads 1
t core_lib X=1 cargo test --release -p intarsia-core --lib inkling2
t docs_env_vars X=1 cargo test --release -p intarsia-cli --test docs_env_vars
echo "jobT done $(date +%T)"
