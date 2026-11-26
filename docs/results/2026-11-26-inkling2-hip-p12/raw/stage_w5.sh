#!/usr/bin/env bash
# Window 5 (lane ink-hip-p12). Cheap checks first: the merged head (p12c = branch + origin/main 80719483a) against main2 (80719483a):
# greedy md5 at 64 tokens short with the spec default; the tiny `inkling2` suite of main2 and of p12c on hip; `inkling2_batch` on hip
# (p12c). Then plain decode 16k (prefill too) and spec-default 4k, 4 + 4 ABBA each (main 3226af2b9 vs p12b, the base of every speed
# row).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
OUT=$REPO/logs/gate
mkdir -p "$OUT"
now() { date '+%T'; }
M=/models/Inkling-2-Mini
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
for a in main2 p12c; do
  f=$OUT/merged-$a-default
  bash "$R/run.sh" /work/.scratch/bin/intarsia-$a generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens 64 -p "$PS" > "$f.out" 2> "$f.err"
  echo "MD5 $a default-short 64 $(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(grep -acE 'P12 \(' "$f.err") P12 load lines [$(now)]"
done
echo "[stage] tiny inkling2, main2's test binary $(now)"
bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/inkling2-test-main2 --test-threads 1 2>&1 | grep -aE "g1b|FAILED|SKIPPED|test result" | head -10
echo "[stage] tiny inkling2, p12c $(now)"
bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip cargo test --release -p intarsia-models --test inkling2 -- --test-threads 1 2>&1 | grep -aE "g1b|FAILED|SKIPPED|test result" | head -10
echo "[stage] inkling2_batch on hip, p12c $(now)"
bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip CI=1 cargo test --release -p intarsia-models --test inkling2_batch -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked|SKIPPED" | head -12
BR=p12b SPEC=off PROMPTS="16k" bash "$R/stage_speed.sh"
BR=p12b SPEC=default PROMPTS="4k" bash "$R/stage_speed.sh"
echo "[stage] done $(now)"
