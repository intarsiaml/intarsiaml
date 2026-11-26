#!/usr/bin/env bash
# Window 8 (lane ink-hip-vrow): the gates of vrow2 (main 12c6c78ac + blend fuse 8:2 + op 75 fused for 1-row calls only). Logs in logs/w8/.
#   card 0: (1) bitwise verify probe rows 2..8 at 17 / 1024 / 4000 + 16 greedy windows (the defaults); (2) greedy md5, --spec off and
#           the spec default, short-0 / 4k-0 x 64 / 256 (main2's from window 4); (3) G2 short / long; (4) the timing probe at 1k, the
#           defaults and main's rules (INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16).
#   card 1: (5) hip tests ink_mix (rows-exact + reference), ink_attn rows_exact_real_shapes, ink_rows; (6) admission of loop_blend,
#           conv_mix, attn_res_mix, ink_attn_decode; (7) hip tiny suites.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w8
rm -rf "$O"; mkdir -p "$O/md5"
M=/models/Inkling-2-Mini
BR=${BR:-vrow2}
now() { date '+%T'; }
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; }
ex() { grep -aoE "/target/release/deps/$1-[0-9a-f]+" "$REPO/logs/$2" | tail -1; }
VEXE=$(ex verify_rows_real build-mtest-verify_rows_real.log)
[ -x "$S/bin/intarsia-$BR" ] && [ -n "$VEXE" ] || { echo "[stage] guard: $BR or the probe missing"; exit 3; }
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
gen() {  # binary mode(off|dflt) prompt-name tokens
  local f=$O/md5/$1-$2-$3-$4 X=""
  [ "$2" = off ] && X="--spec off"
  local PR="$PS"; [ "$3" = 4k ] && PR="$P4"
  r0 /work/.scratch/bin/intarsia-$1 generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens "$4" $X -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $2 $3 $4 $md5 rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-90)"
}
card0() {
  echo "[c0] (1) bitwise verify probe rows 2..8 $(now) ($VEXE)"
  r0 env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_LENS=17,1024,4000 $VEXE --nocapture > $O/probe-bitwise.log 2>&1
  echo "rc $?"; grep -aE "PROBE_RESULT|PROBE_GRAPHS|panicked|test result" $O/probe-bitwise.log | cut -c1-300
  echo "[c0] (2) md5 $BR $(now)"
  for p in short 4k; do for n in 64 256; do for m in off dflt; do gen $BR $m $p $n; done; done; done
  for g in short long; do
    echo "[c0] (3) G2 $g $BR $(now)"
    r0 /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors > $O/g2-$g.log 2>&1
    echo "rc $?"; grep -aE "KL [0-9.e-]+ (PASS|FAIL)|top-1 misses" $O/g2-$g.log | cut -c1-240 | head -3
  done
  PT="INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=8 INTARSIA_VERIFY_TIME_ONLY=1 INTARSIA_VERIFY_PROMPT=/work/.scratch/prompts/len1k.json"
  for arm in main dflt; do
    E=""; [ $arm = main ] && E="INTARSIA_HIP_INK_BLEND_FUSE=2 INTARSIA_HIP_INK_DEC_FUSED_ROWS=16"
    echo "[c0] (4) timing probe 1k rows 1..8, $arm rules $(now)"
    r0 env $E $PT $VEXE --nocapture > $O/probe-$arm.log 2>&1
    echo "rc $?"; grep -aE "PROBE_TIME |panicked" $O/probe-$arm.log | cut -c1-400
  done
  echo "[c0] done $(now)"
}
card1() {
  XEXE=$(ex ink_mix build-htest-ink_mix.log); AEXE=$(ex ink_attn build-htest-ink_attn.log); REXE=$(ex ink_rows build-htest-ink_rows.log)
  echo "[c1] (5) hip tests $(now): $XEXE $AEXE $REXE"
  r1 $XEXE ink_attn_res_mix_and_blend_rows_exact ink_conv_mix_rows_exact ink_mix_ops_match_the_reference --test-threads=1 > $O/test-ink_mix.log 2>&1
  echo "ink_mix rc $? | $(grep -aE '^test result' $O/test-ink_mix.log | head -1)"; grep -aE "^test .*FAILED|panicked" $O/test-ink_mix.log | head -3
  r1 $AEXE rows_exact_real_shapes --test-threads=1 --nocapture > $O/test-ink_attn.log 2>&1
  echo "ink_attn rc $? | $(grep -aE '^test result|rows-exact cases' $O/test-ink_attn.log | head -2 | tr '\n' ' ')"; grep -aE "^test .*FAILED|panicked" $O/test-ink_attn.log | head -3
  r1 $REXE --test-threads=1 > $O/test-ink_rows.log 2>&1
  echo "ink_rows rc $? | $(grep -aE '^test result' $O/test-ink_rows.log | head -1)"; grep -aE "^test .*FAILED|panicked" $O/test-ink_rows.log | head -3
  for op in loop_blend conv_mix attn_res_mix ink_attn_decode; do
    echo "[c1] (6) kernels test $op $(now)"
    r1 /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op > $O/adm-$op.log 2>&1
    echo "rc $?"; grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" $O/adm-$op.log | head -6 | cut -c1-200
  done
  echo "[c1] (7) hip tiny suites $(now)"
  for t in inkling2 inkling2_verify_softcap inkling2_dspark inkling2_dspark_cut; do
    TE=$(ex $t build-mtest-$t.log)
    r1 env INTARSIA_INKLING2_BACKEND=hip $TE --test-threads 1 > $O/tiny-$t.log 2>&1
    echo "$t rc $? | $(grep -aE '^test result' $O/tiny-$t.log | head -1)"; grep -aE "^test .*FAILED" $O/tiny-$t.log | head -5
  done
  echo "[c1] done $(now)"
}
echo "[stage] start $(now) BR=$BR"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
