#!/usr/bin/env bash
# Window 4 (lane ink-hip-verify): the gates of the branch (binary $BR = main 80719483a + [tuning.hip] mtp_default_k = $K) against
# main2 (= main 80719483a). Nothing set in the environment unless named. Full logs in logs/w4/.
#   card 0: greedy md5 (generate --ignore-eos, graphs, the text before finish=), short-0 / 4k-0, 64 and 256 tokens, --spec off and the
#           spec default (no --spec), main2 and the branch.
#   card 1: (a) the bitwise verify probe on the branch's test build, rows 2..$((K+1)) at 17 / 1024 / 4000 ids + 16 greedy windows;
#           (b) the hip tiny suites inkling2, inkling2_verify_softcap, inkling2_dspark, inkling2_dspark_cut (--no-fail-fast);
#           (c) G2 short / long (accuracy --mode prefill, oracle W4A16 goldens), main2 and the branch.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-verify
S=$REPO/.scratch
O=$REPO/logs/w4
rm -rf "$O"; mkdir -p "$O/md5"
M=/models/Inkling-2-Mini
BR=${BR:-verify1}; MAIN=${MAIN:-main2}; K=${K:-3}
now() { date '+%T'; }
# guard: the branch build job (build_branch.sh) finished: its binary and its last log exist
if [ ! -x "$S/bin/intarsia-$BR" ] || [ ! -x "$S/bin/intarsia-$MAIN" ] || [ ! -s "$REPO/logs/test-docs-env.log" ]; then
  echo "[stage] guard: $BR / $MAIN binary or the build's last log missing, not running"; exit 3
fi
r0() { HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" "$@"; }
r1() { HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-verify-run2 bash "$S/run.sh" "$@"; }
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
gen() {  # binary mode(off|dflt) prompt-name tokens
  local f=$O/md5/$1-$2-$3-$4 X=""
  [ "$2" = off ] && X="--spec off"
  local PR="$PS"; [ "$3" = 4k ] && PR="$P4"
  r0 /work/.scratch/bin/intarsia-$1 generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens "$4" $X -p "$PR" > "$f.out" 2> "$f.err"
  local rc=$? md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $2 $3 $4 $md5 rc $rc | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -aE '^spec: verify' "$f.err" | head -1 | cut -c1-90) | $(grep -aE '^--spec dflash|DSpark default' "$f.err" | head -1 | cut -c1-120)"
}
card0() {
  for p in short 4k; do for n in 64 256; do for b in $MAIN $BR; do for m in off dflt; do gen $b $m $p $n; done; done; done; done
  echo "[c0] done $(now)"
}
card1() {
  TEXE=$(grep -aoE '/target/release/deps/verify_rows_real-[0-9a-f]+' "$REPO/logs/build-mtest-verify_rows_real.log" | tail -1)
  echo "[c1] (a) bitwise verify probe rows 2..$((K + 1)) $(now) ($TEXE)"
  r1 env INTARSIA_VERIFY_DIR=$M INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_VERIFY_ROWS=$((K + 1)) INTARSIA_VERIFY_LENS=17,1024,4000 $TEXE --nocapture > $O/probe-bitwise.log 2>&1
  echo "rc $?"; grep -aE "PROBE_RESULT|PROBE_TIME |PROBE_GRAPHS|worst|DIFF|panicked|test result" $O/probe-bitwise.log | cut -c1-300 | head -12
  echo "[c1] (b) hip tiny suites $(now)"
  r1 env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --no-fail-fast --test inkling2 --test inkling2_verify_softcap --test inkling2_dspark --test inkling2_dspark_cut -- --test-threads 1 > $O/tiny.log 2>&1
  echo "rc $?"; grep -aE "^test |test result|Running" $O/tiny.log | cut -c1-200 | head -40
  for b in $MAIN $BR; do for g in short long; do
    echo "[c1] (c) G2 $g $b $(now)"
    r1 /work/.scratch/bin/intarsia-$b accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors > $O/g2-$b-$g.log 2>&1
    echo "rc $?"; grep -aE "prefill floor|floor verdict|KL|top-1|Error|panicked" $O/g2-$b-$g.log | cut -c1-300 | head -4
  done; done
  echo "[c1] done $(now)"
}
echo "[stage] start $(now) BR=$BR MAIN=$MAIN K=$K"
card0 > "$O/c0.log" 2>&1 &
card1 > "$O/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$O/c0.log"
echo "---- card 1"; cat "$O/c1.log"
echo "[stage] done $(now)"
