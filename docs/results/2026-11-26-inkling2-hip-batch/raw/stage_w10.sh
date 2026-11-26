#!/usr/bin/env bash
# Window 10 stage (lane ink-hip-batch): b8 = the branch merged onto main 3226af2b9 (#528: HIP prologue / flash split, `qhi` in InkP).
# Ordered so a short CAP keeps what #528 can have changed: (1) the row-table kernels bitwise the solo calls (InkPR inherits InkP's new
# field); (2) one greedy 4k-256 run per arm, main4 vs b8 (prologue / flash prefill + decode); then the parts that can be cut: short-256,
# the two 64-token md5s, rows_exact_real_shapes, the two admissions, the tiny inkling2_batch suite.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-batch
S=$REPO/.scratch
OUT=$REPO/logs/gate10
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
MA=${MA:-main4}
BR=${BR:-b8}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
G="generate -m $M --backend hip --max-seq 8192 --spec off --ignore-eos --profile-every 0"
now() { date '+%F %T'; }
# the window ended (serve_window.sh's finish touches logs/hold-stop): start nothing more after radiance is back
alive() { [ -e "$REPO/logs/hold-stop" ] && { echo "[stage] hold-stop seen: stopping $(now)"; exit 9; }; return 0; }
f() { grep -aE "^test |test result|bitwise|row-table calls|panicked|assert|SKIPPED|error(\[|:)" | head -${1:-30}; }
gen() {  # arm binary-name prompt-name prompt tokens rep
  local o=$OUT/$1-$3-$5-r$6
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$2 $G --max-tokens "$5" -p "$4" > "$o.out" 2> "$o.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$o.out" | md5sum | cut -c1-32)
  echo "SPD $1 $3 $5 r$6 $md5 | $(cat "$o.out" "$o.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$o.out" "$o.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$o.out" "$o.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
}
alive; echo "[stage] unit: ink_rows $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_rows -- --nocapture --test-threads=1 2>&1 | f 12
alive; echo "[stage] md5 4k 256 $(now)"
gen main $MA 4k "$P4" 256 1
alive; gen branch $BR 4k "$P4" 256 1
alive; echo "[stage] md5 short 256 $(now)"
gen branch $BR short "$PS" 256 1
alive; gen main $MA short "$PS" 256 1
alive; echo "[stage] md5 64 $(now)"
for p in short 4k; do
  [ $p = short ] && PR="$PS" || PR="$P4"
  alive; gen main $MA $p "$PR" 64 1
  gen branch $BR $p "$PR" 64 1
done
alive; echo "[stage] unit: ink_attn rows_exact_real_shapes $(now)"
bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_attn -- --nocapture --test-threads=1 rows_exact_real_shapes 2>&1 | f 6
for op in conv_mix ink_attn_decode; do
  alive; echo "[stage] kernels test $op $(now)"
  bash "$S/run.sh" /work/.scratch/bin/intarsia-$BR kernels test --backend hip --op $op 2>&1 | grep -aE "ADMITTED|REJECT|FAIL|NO CASES|Error|panicked" | head -4
done
alive; echo "[stage] inkling2_batch tiny on hip $(now)"
bash "$S/run.sh" env INTARSIA_INKLING2_BACKEND=hip CI=1 cargo test --release -j8 -p intarsia-models --test inkling2_batch -- --nocapture --test-threads=1 2>&1 | f 14
alive; echo "[stage] done $(now)"
