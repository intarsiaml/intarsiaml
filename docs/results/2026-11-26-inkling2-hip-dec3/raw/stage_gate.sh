#!/usr/bin/env bash
# Gate stage (lane ink-hip-dec3): greedy md5 at 64 tokens (short, 4k) for main f3a150646, the branch (default) and the branch
# with INTARSIA_HIP_INK_DEC3=0; G2 short / long on the branch; admission of the touched ops. Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
OUT=$REPO/logs/gate
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-dec3c}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # arm binary-name fast prompt-name prompt tokens
  local f=$OUT/$1-$4-$6
  bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$3 /work/.scratch/bin/intarsia-$2 $G --max-tokens "$6" -p "$5" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $4 $6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
echo "[stage] ink_attn rows_exact_real_shapes (op 75 templated on DPP: default vs the unfused pair, and at INTARSIA_HIP_INK_DEC3=0) $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | grep -aE "^test |test result|bitwise|panicked|assert|error\[" | head -12
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  echo "[stage] md5 $p 64 $(now)"
  gen main main 0 $p "$PR" 64
  gen branch $BR 7 $p "$PR" 64
  gen branch0 $BR 0 $p "$PR" 64
done
for g in short long; do
  echo "[stage] G2 $g branch $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400
done
for op in moe_route conv_mix attn_res_mix loop_blend sconv_res ink_attn_decode argmax logits; do
  echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -6
done
echo "[stage] done $(now)"
