#!/usr/bin/env bash
# Window 3 stage (lane ink-hip-dec2): greedy md5 at 64 tokens (short, 4k) for main f569b46a1, the branch (default) and the branch
# with INTARSIA_HIP_INK_DEC_FAST=0; G2 short / long on the branch; admission of the touched ops. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec2
S=$REPO/.scratch
OUT=$REPO/logs/gate
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-dec2b}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # arm binary-name fast prompt-name prompt tokens
  local f=$OUT/$1-$4-$6
  bash "$S/run.sh" env INTARSIA_HIP_INK_DEC_FAST=$3 /work/.scratch/bin/intarsia-$2 $G --max-tokens "$6" -p "$5" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $4 $6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
echo "[stage] ink_moe tests $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_rows_exact ink_moe_wave_grouping_odd_ids 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert" | head -12
echo "[stage] ink_mix tests (rows-exact, route bitwise, admission of the mix ops) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_mix -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|ADMITTED|FAIL" | head -24
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  echo "[stage] md5 $p 64 $(now)"
  gen main main 7 $p "$PR" 64
  gen branch $BR 7 $p "$PR" 64
  gen branch0 $BR 0 $p "$PR" 64
done
for g in short long; do
  echo "[stage] G2 $g branch $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400
done
for op in moe_route moe_experts_mxfp8 loop_blend conv_mix attn_res_mix moe_experts; do
  echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -6
done
echo "[stage] done $(now)"
