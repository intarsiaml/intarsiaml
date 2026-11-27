#!/usr/bin/env bash
# Window 1 (lane ink-hip-p12): the step-1 microbench (P12 vs bf16 GEMV at the real shapes, rows 1 and 4, launch shapes), then the
# gate-1 bitwise tests (P12 == bf16 twin, rows 1..16, real shapes; small / adversarial), then one quick end-to-end look: greedy
# 128 tokens short, main vs the branch (default) vs the branch at INTARSIA_INK_PACK12=0, and both at the default max-seq (load log).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
OUT=$REPO/logs/w1
mkdir -p "$OUT"
M=/models/Inkling-2-Mini
BR=${BR:-p12a}
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
now() { date '+%F %T'; }
echo "[stage] microbench $(now)"
bash "$R/run.sh" env P12_BENCH=1 P12_MS=1,4,16 INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_p12 p12_microbench -- --nocapture --test-threads=1 2>&1 | grep -aE "^BENCH|test result|panicked|error" | head -80
echo "[stage] bitwise tests $(now)"
bash "$R/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_p12 -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|bitwise|escapes|panicked|differs|error" | head -20
G="generate -m $M --backend hip --spec off --ignore-eos --profile-every 0"
gen() {  # arm binary env tokens maxseq-args
  local f=$OUT/$1-$4
  bash "$R/run.sh" env $3 /work/.scratch/bin/intarsia-$2 $G $5 --max-tokens "$4" -p "$PS" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "GEN $1 $4 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'steps=[0-9]+ graph replays=[0-9]+ \(captures [0-9]+\) eager decode steps=[0-9]+' | head -1)"
  grep -aiE "P12|pack12|max.seq|kv cache|kv budget|weights" "$f.err" | head -6 | cut -c1-300
}
echo "[stage] e2e: branch short 128 (max-seq 8192), then main and branch at the default max-seq (load log) $(now)"
gen branch $BR X=1 128 "--max-seq 8192"
gen main-dms main X=1 16 ""
gen branch-dms $BR X=1 16 ""
echo "[stage] done $(now)"
