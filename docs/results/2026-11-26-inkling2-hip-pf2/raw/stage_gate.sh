#!/usr/bin/env bash
# Gate stage (lane ink-hip-pf2), two cards at once. Radiance down <= 5 min (window.sh cap).
#   card 0: greedy md5, 64 tokens at short / 4k and 16 tokens at 16k, three arms, all with the new prefill settings
#           (INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1 --prefill-chunk 2048): M main, B the branch default,
#           B0 the branch at INTARSIA_HIP_INK_PF2=0. Expected: one md5 per prompt. Then G2 short / long on B, then admission of
#           the touched ops on B (experts with the tensor-core prefill on, rel_sink_attn with flash on).
#   card 1: the hip crate's ink_attn (quick admission) and ink_moe tests on the branch's test binaries.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pf2
S=$REPO/.scratch
OUT=$REPO/logs/gate
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-pf2a}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
NEW="INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1"
G="generate -m $M --backend hip --max-seq 20480 --spec off --ignore-eos --profile-every 0 --prefill-chunk 2048"
now() { date '+%T'; }
gen() {  # arm binary prompt-name prompt tokens "env..."
  local f=$OUT/$1-$3
  HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env $NEW $6 /work/.scratch/bin/intarsia-$2 $G --max-tokens $5 -p "$4" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $3 $5 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
card0() {
  for p in short 4k 16k; do
    case $p in short) PR="$PS"; N=64 ;; 4k) PR="$P4"; N=64 ;; *) PR="$P16"; N=16 ;; esac
    gen M main $p "$PR" $N "INK_ARM=M"
    gen B $BR $p "$PR" $N "INK_ARM=B"
    gen B0 $BR $p "$PR" $N "INTARSIA_HIP_INK_PF2=0"
  done
  for g in short long; do
    echo "G2 $g branch [$(now)] $(HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env $NEW /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --prefill-chunk 2048 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400)"
  done
  for op in ink_kv_prologue rel_sink_attn moe_experts moe_experts_mxfp8; do
    echo "ADMIT $op [$(now)]"
    HIP_VISIBLE_DEVICES=0 bash "$S/run.sh" env $NEW /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -8 | sed "s/^/ADMIT $op: /"
  done
}
card1() {
  echo "TEST ink_attn [$(now)]"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-pf2-run2 timeout ${TEST_CAP:-150} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 INK_ATTN_QUICK=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|assert|FAIL|error|bitwise" | head -30
  docker rm -f ink-hip-pf2-run2 >/dev/null 2>&1
  echo "TEST ink_moe [$(now)]"
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-pf2-run2 timeout ${TEST_CAP2:-110} bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|panicked|assert|FAIL|error|bitwise" | head -30
  docker rm -f ink-hip-pf2-run2 >/dev/null 2>&1
}
echo "[stage] gate start $(now)"
card0 > "$OUT/c0.log" 2>&1 &
card1 > "$OUT/c1.log" 2>&1 &
wait
echo "---- card 0"; cat "$OUT/c0.log"
echo "---- card 1"; cat "$OUT/c1.log"
echo "[stage] gate done $(now)"
