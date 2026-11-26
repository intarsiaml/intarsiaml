#!/usr/bin/env bash
# Window 2 stage (lane ink-hip-pfflip), two cards at once: prefill and decode tok/s, main's default (binary intarsia-main, no flags:
# chunk 256, both knobs off) vs the branch's defaults (intarsia-pf, nothing set), 3 interleaved runs per arm. Radiance down <= 5 min.
#   card 0: 4k prompt (4228 tokens), 64 tokens generated; the short prompt, 256 tokens (decode tok/s); then the rerun of the ink_moe
#   default test. card 1: 16k (the 4k prompt four times over, 16915 tokens, --max-seq 20480), 8 tokens.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-pfflip
S=$REPO/.scratch
OUT=$REPO/logs/speed
rm -rf "$OUT"; mkdir -p "$OUT"
M=/models/Inkling-2-Mini
PS="$(cat "$REPO/scripts/cmx/ids-prompts/short-0.txt")"
P4="$(cat "$REPO/scripts/cmx/ids-prompts/4k-0.txt")"
P16="$P4
$P4
$P4
$P4"
now() { date '+%T'; }
gen() {  # card container arm binary tag prompt tokens maxseq rep   (ONLY=branch: skip the other arm; window 3 reran the branch alone)
  [ -n "${ONLY:-}" ] && [ "$3" != "$ONLY" ] && return 0
  local f=$OUT/$3-$5-r$9
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$2 bash "$S/run.sh" /work/.scratch/bin/intarsia-$4 generate -m $M --backend hip --max-seq $8 --spec off --ignore-eos --profile-every 0 --max-tokens $7 -p "$6" > "$f.out" 2> "$f.err"
  local md5; md5=$(sed -e 's/finish=length.*//' -e '/steps=/d' -e '/tok\/s/d' "$f.out" | md5sum | cut -c1-32)
  echo "SPD $3 $5 r$9 $md5 | $(cat "$f.out" "$f.err" | grep -aoE 'prompt=[0-9]+' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) | $(cat "$f.out" "$f.err" | grep -aoE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' | head -1) [$(now)]"
}
card0() {
  for rep in 1 2 3; do
    if [ $((rep % 2)) = 1 ]; then gen 0 ink-hip-pfflip-run main main 4k "$P4" 64 8192 $rep; gen 0 ink-hip-pfflip-run branch pf 4k "$P4" 64 8192 $rep
    else gen 0 ink-hip-pfflip-run branch pf 4k "$P4" 64 8192 $rep; gen 0 ink-hip-pfflip-run main main 4k "$P4" 64 8192 $rep; fi
  done
  for rep in 1 2 3; do
    if [ $((rep % 2)) = 1 ]; then gen 0 ink-hip-pfflip-run main main short "$PS" 256 8192 $rep; gen 0 ink-hip-pfflip-run branch pf short "$PS" 256 8192 $rep
    else gen 0 ink-hip-pfflip-run branch pf short "$PS" 256 8192 $rep; gen 0 ink-hip-pfflip-run main main short "$PS" 256 8192 $rep; fi
  done
  [ -n "${ONLY:-}" ] && return 0
  echo "TEST ink_moe_wmma_default_and_revert [$(now)]"
  HIP_VISIBLE_DEVICES=0 INK_CONTAINER=ink-hip-pfflip-run timeout 100 bash "$S/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_wmma_default_and_revert 2>&1 | grep -aE "^test |test result|panicked|assert|FAIL|error" | head -10
}
card1() {
  for rep in 1 2 3; do
    if [ $((rep % 2)) = 1 ]; then gen 1 ink-hip-pfflip-run2 branch pf 16k "$P16" 8 20480 $rep; gen 1 ink-hip-pfflip-run2 main main 16k "$P16" 8 20480 $rep
    else gen 1 ink-hip-pfflip-run2 main main 16k "$P16" 8 20480 $rep; gen 1 ink-hip-pfflip-run2 branch pf 16k "$P16" 8 20480 $rep; fi
  done
}
echo "[stage] speed start $(now)"
card0 > "$OUT/c0.log" 2>&1 &
card1 > "$OUT/c1.log" 2>&1 &
wait
echo "---- card 0 (4k, short)"; cat "$OUT/c0.log"
echo "---- card 1 (16k)"; cat "$OUT/c1.log"
echo "[stage] speed done $(now)"
