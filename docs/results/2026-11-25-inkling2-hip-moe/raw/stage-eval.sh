#!/usr/bin/env bash
# S2 evaluation stage of lane ink-hip-moe (inside window.sh; prebuilt binaries). Two streams, one per R9700:
#   card 0: 4k prefill at the default chunk and at chunk 2048 for off / wmma / flash / both (the "off" default-chunk run makes
#           64 tokens: its text md5 is the bitwise gate against ink/hip-speed), then admission of both expert ops and this
#           lane's test with the real-shape T >= 1024 cases, all with the tensor-core prefill on;
#   card 1: G2 short and long for wmma / flash / both, then 16k prefill at chunk 2048 for the four settings.
set -uo pipefail
B=$HOME/intarsiaml-ink-hip-moe
cd "$B"
OUT=$B/logs/eval; rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
I=/target/release/intarsia
P4="$(cat scripts/cmx/ids-prompts/4k-0.txt)"
P16="$P4
$P4
$P4
$P4"
now() { date '+%T'; }
envs() { case $1 in off) echo "INK_EVAL=off" ;; wmma) echo "INTARSIA_HIP_INK_MOE_WMMA=1" ;; flash) echo "INTARSIA_HIP_INK_ATTN_FLASH=1" ;; both) echo "INTARSIA_HIP_INK_MOE_WMMA=1 INTARSIA_HIP_INK_ATTN_FLASH=1" ;; esac; }
gen() {  # card container tag cfg prompt tokens maxseq [extra flags]
  local card=$1 cn=$2 tag=$3 cfg=$4 prompt=$5 tok=$6 ms=$7; shift 7
  local f=$OUT/$tag-$cfg
  HIP_VISIBLE_DEVICES=$card INK_CONTAINER=$cn bash .scratch/run.sh env $(envs $cfg) $I generate -m $M --backend hip --max-seq $ms --spec off --ignore-eos --profile-every 0 --max-tokens $tok "$@" -p "$prompt" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "GEN $tag $cfg md5 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'finish=.*|Error.*' | head -1 | cut -c1-220) [$(now)]"
}
stream0() {
  for cfg in off wmma flash both; do gen 0 ink-hip-moe-run 4k-default $cfg "$P4" 64 8192; done
  for cfg in off wmma flash both; do gen 0 ink-hip-moe-run 4k-c2048 $cfg "$P4" 8 8192 --prefill-chunk 2048; done
  for op in moe_experts moe_experts_mxfp8; do
    echo "ADMIT $op wmma=1 start [$(now)]"
    HIP_VISIBLE_DEVICES=0 INK_CONTAINER=ink-hip-moe-run bash .scratch/run.sh env INTARSIA_HIP_INK_MOE_WMMA=1 $I kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -8 | sed "s/^/ADMIT $op: /"
  done
  echo "BIG start [$(now)]"
  HIP_VISIBLE_DEVICES=0 INK_CONTAINER=ink-hip-moe-run timeout ${BIG_CAP:-70} bash .scratch/run.sh env INTARSIA_REQUIRE_GPU=1 INK_MOE_BIG=1 INTARSIA_HIP_INK_MOE_WMMA=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_matches_cpu_reference 2>&1 | grep -aE "T=(1024|2048|4096)|test result|FAIL|panicked|failures" | cut -c1-200 | sed "s/^/BIG: /"
  echo "BIG exit ${PIPESTATUS[0]} (124 = its cap) [$(now)]"
}
stream1() {
  for cfg in wmma flash both; do
    for g in short long; do
      echo "G2 $g $cfg | $(HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-hip-moe-run2 bash .scratch/run.sh env $(envs $cfg) $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-420 | tr '\n' ' ') [$(now)]"
    done
  done
  for cfg in off wmma flash both; do gen 1 ink-hip-moe-run2 16k-c2048 $cfg "$P16" 8 20480 --prefill-chunk 2048; done
}
echo "[stage] eval start $(now)"
stream0 > "$OUT/s0.log" 2>&1 &
stream1 > "$OUT/s1.log" 2>&1 &
wait
echo "---- card 0"; cat "$OUT/s0.log"
echo "---- card 1"; cat "$OUT/s1.log"
echo "[stage] eval done $(now)"
