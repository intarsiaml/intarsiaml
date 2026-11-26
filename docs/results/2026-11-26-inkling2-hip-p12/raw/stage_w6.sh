#!/usr/bin/env bash
# Window 6 (lane ink-hip-p12, brief 2; radiance down <= 3 min). In order:
#   1. `serve --max-batch 16` at the default --kv-fraction / --max-seq, main2 (80719483a) on card 1 and p12d on card 0 at the same
#      time (each start admits its plan cold, ~85-100 s; HOME on the lane folder): each start's "loaded ... context N tokens" line
#      (the per-sequence context the KV budget allows) and its load notes;
#   2. solo `generate` on p12d (spec default, short, 64 tokens): greedy md5 (main: 04cdb28a) and the P12 load lines;
#   3. `inkling2_tiny_g1b_teacher_forced_bf16` on hip with INTARSIA_HIP_INK_ATTN_FLASH=0, main2's test binary;
#   4. if time is left: round 2 of 1 with the cards swapped (warm admission cache), then `inkling2_batch` on hip (p12d).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
OUT=$REPO/logs/w6
mkdir -p "$OUT" "$REPO/.scratch/home"
now() { date '+%T'; }
M=/models/Inkling-2-Mini
BR=${BR:-p12d}
cleanup() { docker rm -f ink-hip-p12-c0 ink-hip-p12-c1 >/dev/null 2>&1; }
trap cleanup EXIT TERM INT
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
serve_start() {  # binary card tag: start `serve --max-batch 16` in container ink-hip-p12-c<card>, log to $OUT/serve-<tag>.log
  ( export INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-p12 INTARSIA_CONTAINER_NAME=ink-hip-p12-c$2 HIP_VISIBLE_DEVICES=$2
    cd "$REPO" && exec deploy/tatooine/dev.sh bash -c "HOME=/work/.scratch/home exec /work/.scratch/bin/intarsia-$1 serve -m $M --backend hip --max-batch 16 --host 127.0.0.1 --port 8131 --model-id ink" ) \
    > "$OUT/serve-$3.log" 2>&1 < /dev/null &
}
serve_wait() {  # max-seconds tag...: wait until every tag's log has its "loaded" line or an error, then stop both containers
  local lim=$1; shift
  for i in $(seq 1 $((lim * 2))); do
    done_all=1
    for t in "$@"; do grep -aqE "loaded .* context [0-9]+ tokens|^Error" "$OUT/serve-$t.log" 2>/dev/null || done_all=0; done
    [ $done_all = 1 ] && break
    sleep 0.5
  done
  cleanup
  for t in "$@"; do
    echo "SERVE $t | $(grep -aoE 'loaded .* in [0-9.]+s, context [0-9]+ tokens' "$OUT/serve-$t.log" | head -1) | $(grep -aoE 'admission on plan shapes: .*' "$OUT/serve-$t.log" | head -1) [$(now)]"
    grep -aE "pack12|P12 \(|clamped|^Error" "$OUT/serve-$t.log" | head -4 | cut -c1-220
  done
}
echo "[stage] serve --max-batch 16, round 1 (cold admission): main2 card 1, $BR card 0 $(now)"
serve_start main2 1 main2-c1
serve_start $BR 0 $BR-c0
serve_wait 140 main2-c1 $BR-c0
f=$OUT/solo-$BR
bash "$R/run.sh" /work/.scratch/bin/intarsia-$BR generate -m $M --backend hip --max-seq 8192 --ignore-eos --profile-every 0 --max-tokens 64 -p "$PS" > "$f.out" 2> "$f.err"
echo "MD5 $BR default-short 64 $(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
grep -aE "P12 \(|pack12" "$f.err" | cut -c1-200
echo "[stage] g1b, flash off, main2's test binary $(now)"
bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip INTARSIA_HIP_INK_ATTN_FLASH=0 /work/.scratch/bin/inkling2-test-main2 inkling2_tiny_g1b_teacher_forced_bf16 --test-threads 1 2>&1 | grep -aE "^test |SKIPPED|test result|rel err|exceeds|budget" | head -8
if [ $SECONDS -lt 135 ]; then
  echo "[stage] serve --max-batch 16, round 2 (warm, cards swapped): main2 card 0, $BR card 1 $(now)"
  serve_start main2 0 main2-c0
  serve_start $BR 1 $BR-c1
  serve_wait 35 main2-c0 $BR-c1
fi
if [ $SECONDS -lt 165 ]; then
  echo "[stage] inkling2_batch on hip, $BR $(now)"
  bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip CI=1 cargo test --release -p intarsia-models --test inkling2_batch -- --test-threads 1 2>&1 | grep -aE "^test |test result|panicked|SKIPPED" | head -12
fi
echo "[stage] done $(now) (${SECONDS}s)"
