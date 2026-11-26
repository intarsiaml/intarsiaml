#!/bin/bash
# lane ink-decroof job G (exclusive): CONV_PRE / BIAS_PRE / BIAS_EARLY gates + fresh decode profile of main.
#  1. greedy ids, base vs cand, 143 / 1021 / 4108-token prompts, 256 tokens, --ignore-eos: --spec off and the spec default;
#     diag binary arms per prompt (A all off, B CONV_PRE, C + BIAS_PRE, D + BIAS_EARLY) against both, with launch counts
#  2. Metal tests on cand (backend-metal Inkling tests, models inkling2 = G1 tiny, inkling2_batch)
#  3. G2 short (decode mode) self-golden, base vs cand, byte compare
#  4. profile, --spec off: per-dispatch (COMMIT_EVERY=1, TOKEN_FEEDBACK=0) short and 4k for base and cand; normal-step cb
#     trace (token feedback) short and 4k for base
# Every step prints its result as it finishes; TERM kills the tree.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
kill_tree() { local c; for c in $(pgrep -P $1 2>/dev/null); do kill_tree $c; done; kill $1 2>/dev/null; }
cleanup() { for c in $(pgrep -P $$ 2>/dev/null); do kill_tree $c; done; }
trap cleanup EXIT; trap 'echo "CUT (TERM) at step ${STEP:-?}"; cleanup; exit 143' INT TERM
for v in $(env | grep -oE '^INTARSIA_[A-Z0-9_]+'); do unset $v; done
W=/opt/code/intarsiaml/.worktrees/ink-decroof; S=$W/.scratch; O=$S/G; mkdir -p $O
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
BASE=$S/bin/intarsia-base; CAND=$S/bin/intarsia-cand; DIAG=$S/bin/intarsia-diag
for f in $BASE $CAND $DIAG; do [ -x $f ] || { echo "ABORT: missing $f"; exit 6; }; done
cd $W
echo "jobG start $(date +%T) tree $(git rev-parse --short HEAD) base $(md5 -q $BASE) cand $(md5 -q $CAND) diag $(md5 -q $DIAG)"
others=$(pgrep -fl 'intarsia[^ ]* (serve|generate|accuracy)' || true)
[ -n "$others" ] && { echo "ABORT: other engine processes on the box:"; echo "$others"; exit 5; }
G="--model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --max-tokens 256 --ignore-eos --show-ids"
run() { # tag binary [env...] -- [generate args]
  local tag=$1 bin=$2; shift 2
  local envs=(X=1); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local t0=$(date +%s)
  env "${envs[@]}" $bin generate $G "$@" > $O/$tag.out 2> $O/$tag.err
  echo "== $tag exit $? $(( $(date +%s) - t0 )) s | $(grep -h '^decode\[' $O/$tag.err | sed 's/decode//;s/ tokens, [0-9]* ms//' | tr '\n' ' ') | $(grep -h '^cpre\[' $O/$tag.err | tr '\n' ' ')"
}
ids() { grep -h '^ids\[' $O/$1.err | sed -n "$2p" | sed 's/^ids\[[^]]*\]: //' | md5 -q; }
STEP=ids
{
run base_off $BASE -- --spec off --prompt-ids $S/prompts.json
run cand_off $CAND -- --spec off --prompt-ids $S/prompts.json
run base_spec $BASE -- --prompt-ids $S/prompts.json
run cand_spec $CAND -- --prompt-ids $S/prompts.json
run diagAB_off $DIAG INTARSIA_DIAG_ARMS=A,A,A,B,B,B -- --spec off --prompt-ids $S/prompts6.json
run diagCD_off $DIAG INTARSIA_DIAG_ARMS=C,C,C,D,D,D -- --spec off --prompt-ids $S/prompts6.json
run diagBA_spec $DIAG INTARSIA_DIAG_ARMS=B,B,B,A,A,A -- --prompt-ids $S/prompts6.json
run diagDC_spec $DIAG INTARSIA_DIAG_ARMS=D,D,D,C,C,C -- --prompt-ids $S/prompts6.json
for m in off spec; do
  for i in 1 2 3; do
    b=$(ids base_$m $i); c=$(ids cand_$m $i)
    if [ $m = off ]; then dA=$(ids diagAB_$m $i); dB=$(ids diagAB_$m $((i + 3))); dC=$(ids diagCD_$m $i); dD=$(ids diagCD_$m $((i + 3)))
    else dB=$(ids diagBA_$m $i); dA=$(ids diagBA_$m $((i + 3))); dD=$(ids diagDC_$m $i); dC=$(ids diagDC_$m $((i + 3))); fi
    v=EQUAL; for x in $c $dA $dB $dC $dD; do [ $x = $b ] || v=DIFFER; done
    echo "IDS $m prompt $i: base $b cand $c diag A $dA B $dB C $dC D $dD: $v"
  done
done
grep -h '^spec:' $O/*_spec.err | sort | uniq -c
} 2>&1 | tee $O/ids.txt
STEP=tests
t() { tag=$1; shift; local t0=$(date +%s); env "$@" > $O/t_$tag.log 2>&1; rc=$?
  echo "== tests $tag exit $rc $(( $(date +%s) - t0 )) s: $(grep -h 'test result' $O/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
  grep -h "panicked\|FAILED" $O/t_$tag.log | head -8; }
{
t metal_ink X=1 cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact --test inkmix --test inkctrl --test inkload
t attn_bpre INTARSIA_INK_BIAS_PRE=1 cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact
t attn_bearly INTARSIA_INK_BIAS_PRE=1 INTARSIA_INK_BIAS_EARLY=1 cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact
t attn_cpre_off INTARSIA_INK_CONV_PRE=0 cargo test --release -p intarsia-backend-metal --test inkling2_attn --test rows_exact
t models_metal INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch -- --test-threads 1
t batch_bearly INTARSIA_INKLING2_BACKEND=metal INTARSIA_INK_BIAS_PRE=1 INTARSIA_INK_BIAS_EARLY=1 cargo test --release -p intarsia-models --test inkling2_batch -- --test-threads 1
} 2>&1 | tee $O/tests.txt
STEP=g2
{
for a in base cand cand_bearly; do
  bin=$S/bin/intarsia-${a%_bearly}; t0=$(date +%s)
  e=X=1; [ $a = cand_bearly ] && e="INTARSIA_INK_BIAS_PRE=1 INTARSIA_INK_BIAS_EARLY=1"
  env $e $bin accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors --max-mean-kl 4e-3 --no-ref-noise --allow-dirty --write-self-golden $O/g2-short-$a.safetensors > $O/g2-short-$a.log 2>&1
  echo "== g2 short $a exit $? $(( $(date +%s) - t0 )) s: $(grep -h '^OVERALL' $O/g2-short-$a.log | tr -s ' ' | cut -c1-120 | tr '\n' '|')"
done
echo "G2 short self-golden base vs cand: $(python3 -I $S/stcmp.py $O/g2-short-base.safetensors $O/g2-short-cand.safetensors 2>&1)"
echo "G2 short self-golden base vs cand + BIAS_PRE + BIAS_EARLY: $(python3 -I $S/stcmp.py $O/g2-short-base.safetensors $O/g2-short-cand_bearly.safetensors 2>&1)"
} 2>&1 | tee $O/g2.txt
STEP=profile
{
for a in base cand; do
  bin=$S/bin/intarsia-$a
  for k in 143 4k; do
    t0=$(date +%s)
    env X=1 INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/disp_${a}_$k.jsonl $bin generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --ignore-eos --max-tokens 33 --prompt-ids $S/p$k.json > $O/disp_${a}_$k.out 2> $O/disp_${a}_$k.err
    echo "== per-dispatch $a $k exit $? $(( $(date +%s) - t0 )) s"
  done
done
for k in 143 4k; do
  t0=$(date +%s)
  env X=1 INTARSIA_METAL_PROFILE=cb:$O/cb_base_$k.jsonl $BASE generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --spec off --ignore-eos --max-tokens 257 --prompt-ids $S/p$k.json > $O/cb_base_$k.out 2> $O/cb_base_$k.err
  echo "== normal-step base $k exit $? $(( $(date +%s) - t0 )) s: $(python3 -I $S/cbstat.py $O/cb_base_$k.jsonl fb 2>&1 | grep -E 'steps|span' | tr -s ' ' | tr '\n' ' ') $(grep -h '^decode\[' $O/cb_base_$k.err)"
done
} 2>&1 | tee $O/profile.txt
echo "jobG done $(date +%T)"
