#!/usr/bin/env bash
# Follow-up window (lane ink-hip-dec3, coordinator brief ~04:40): a non-Inkling HIP model (Qwen3.6-35B-A3B-NVFP4, 248320 x 2048
# lm_head) with main, the branch default and the branch at INTARSIA_HIP_INK_DEC3=0: 256 greedy tokens, short and ~4k prompt, md5
# of the text; admission of matmul / logits / argmax on the branch; a rocprofv3 trace of the branch's short decode (does the
# lm_head GEMV run in pieces?). Radiance down <= 5 min.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-dec3
S=$REPO/.scratch
OUT=$REPO/logs/x35
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Qwen3.6-35B-A3B-NVFP4
BR=${BR:-dec3c}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --mem-budget 29 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # arm binary-name dec3 prompt-name prompt tokens
  local f=$OUT/$1-$4-$6
  bash "$S/run.sh" env INTARSIA_HIP_INK_DEC3=$3 /work/.scratch/bin/intarsia-$2 $G --max-tokens "$6" -p "$5" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $4 $6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prompt=[0-9]+|decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -2 | tr '\n' ' ') | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1) $(grep -aE 'Error|panicked' "$f.out" "$f.err" | head -2 | cut -c1-200)"
}
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  echo "[stage] 35B md5 $p 256 $(now)"
  gen main main 7 $p "$PR" 256
  gen branch $BR 7 $p "$PR" 256
  gen branch0 $BR 0 $p "$PR" 256
done
for op in matmul logits argmax; do
  echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -40
done
echo "[stage] rocprof 35B short, branch $(now)"
mkdir -p "$OUT/rp"
INTARSIA_IMAGE=intarsia-hip-prof:rocm7.2.4-rust1.90.0 bash "$S/run.sh" rocprofv3 --kernel-trace --output-format csv -d /work/logs/x35/rp -o run -- /work/.scratch/bin/intarsia-$BR $G --max-tokens 12 -p "$PS" 2>&1 | grep -aE "finish=|steps=|Error|panicked" | cut -c1-300
csv=$(find "$OUT/rp" -name '*kernel_trace.csv' | head -1)
[ -n "$csv" ] && gzip -kf "$csv" && python3 "$S/tailtrace.py" "$csv"
echo "[stage] done $(now)"
