#!/usr/bin/env bash
# GPU stage of the ink/hip-speed PR window (lane ink-hip-attn): greedy output md5 + decode, this branch vs main 4fc9901d4; G2;
# admission of the ink ops. Prebuilt binaries only (both built by a --load cpu job before). Budget: radiance down <= 5 min.
set -uo pipefail
B=$HOME/intarsiaml-ink-hip-speed
M0=$HOME/intarsiaml-ink-hip-speed-main
R=$B/docs/results/2026-11-25-inkling2-hip-attn/raw
OUT=$B/logs/md5
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
I=/target/release/intarsia
PS="$(cat "$B/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$B/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
gen() {  # build-name tree volume prompt-name prompt tokens
  local f=$OUT/$1-$4-$6
  bash "$R/run_on.sh" "$2" "$3" ink-hip-attn-run $I $G --max-tokens "$6" -p "$5" > "$f.out" 2> "$f.err"
  # the md5 of the generated text only: everything before "finish=" (timing lines dropped); raw files kept for a re-hash
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "MD5 $1 $4 $6 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'finish=.*' | head -1 | cut -c1-200)"
  echo "GRAPH $1 $4 $6 | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
for tok in 64 256; do
  for p in short 4k; do
    [ $p = short ] && PR="$PS" || PR="$P4"
    echo "[stage] md5 $p $tok $(now)"
    gen main "$M0" intarsia-cargo-target-ink-hip-speed-main $p "$PR" $tok
    gen branch "$B" intarsia-cargo-target-ink-hip-attn $p "$PR" $tok
  done
done
for g in short long; do
  echo "[stage] G2 $g branch $(now)"
  bash "$R/run_on.sh" "$B" intarsia-cargo-target-ink-hip-attn ink-hip-attn-run $I accuracy -m $M --backend hip --max-seq 4096 --mode prefill --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors 2>&1 | grep -aE "prefill floor|floor verdict|Error|panicked" | cut -c1-400
done
for op in ink_kv_prologue rel_sink_attn ink_attn_decode attn_res_mix loop_blend sconv_res conv_mix ink_in_proj moe_experts moe_experts_mxfp8 moe_route; do
  echo "[stage] kernels test $op $(now)"
  bash "$R/run_on.sh" "$B" intarsia-cargo-target-ink-hip-attn ink-hip-attn-run $I kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -6
done
echo "[stage] done $(now)"
