#!/usr/bin/env bash
# Window 1 stage (lane ink-hip-pfflip), two cards at once. Radiance down <= 5 min (window.sh cap 280 s).
#   card 0: greedy md5 at 64 tokens, short and 4k prompts, four arms: M0 main default; M1 main with INTARSIA_HIP_INK_MOE_WMMA=1
#           INTARSIA_HIP_INK_ATTN_FLASH=1 --prefill-chunk 2048; P0 the branch with nothing set; P1 the branch with both knobs 0 and
#           --prefill-chunk 256. Expected: P0 = M1, P1 = M0. Then G2 short / long at the branch defaults, then admission of the
#           expert and attention ops at the branch defaults.
#   card 1: the hip crate's tests at the defaults (ink_moe, ink_attn: default and revert tests, rows-exact pinned, admission sweep).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pfflip
S=$REPO/.scratch
OUT=$REPO/logs/gate
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%T'; }
gen() {  # arm binary prompt-name prompt "env..." extra-flags
  local f=$OUT/$1-$3
  HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env $5 /work/.scratch/bin/intarsia-$2 $G --max-tokens 64 $6 -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $3 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
card0() {
  for p in 4k short; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    gen M0 main $p "$PR" "INK_ARM=M0" ""
    gen M1 main $p "$PR" "INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1" "--prefill-chunk 2048"
    gen P0 pf $p "$PR" "INK_ARM=P0" ""
    gen P1 pf $p "$PR" "INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0" "--prefill-chunk 256"
  done
  for g in short long; do
    echo "G2 $g branch defaults [$(now)] $(HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" /work/.scratch/bin/intarsia-pf accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400)"
  done
  for op in moe_experts moe_experts_mxfp8 rel_sink_attn ink_kv_prologue; do
    echo "ADMIT $op [$(now)]"
    HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" /work/.scratch/bin/intarsia-pf kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -8 | sed "s/^/ADMIT $op: /"
  done
}
card1() {
  echo "TEST ink_moe [$(now)]"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-pfflip-run2 timeout ${TEST_CAP:-110} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|assert|FAIL|error" | head -30
  echo "TEST ink_attn [$(now)]"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-pfflip-run2 timeout ${TEST_CAP:-125} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INK_ATTN_QUICK=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|assert|FAIL|error|rows-exact" | head -30
}
echo "[stage] gate start $(now)"
card0 > "$OUT/c0.log" 2>&1 &
card1 > "$OUT/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$OUT/c0.log"
echo "---- card 1"; cat "$OUT/c1.log"
echo "[stage] gate done $(now)"
