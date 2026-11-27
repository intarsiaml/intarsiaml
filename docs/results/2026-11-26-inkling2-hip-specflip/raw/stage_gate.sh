#!/usr/bin/env bash
# Lane ink-hip-specflip, gate window (gate 2 of the brief), ON tatooine inside window.sh. Both cards at once, binary $BR (the PR head
# built on merged main), no INTARSIA_* variable set unless named; every process HOME=/work/.scratch/home (one admission cache).
# Card 0: 100 seeded sampled requests at T 1.0 (real-model harness tests/inkling2_sampled_real.rs, 24 tokens, k5): drafting off
#   (INTARSIA_SPEC_SAMPLED_DFLASH=0, the revert) then the default (unset: the new default); their lines must be identical. Then
#   serve --max-batch 16 (16 greedy replies at once, serve_bitwise.py, 128 tokens).
# Card 1: greedy md5, md5.json (golden `code` 101 ids + the 4108-id prompt), 256 tokens, --ignore-eos: --spec off vs the default; then
#   serve solo (the default: DSpark k5), the same 16 replies one at a time; then one default sampled `generate` (T 1.0, seed 1) for
#   the startup lines. b16 vs solo is across the two cards (the confirmation ABBA showed both cards give identical ids); each
#   server's first start runs the plan admission (~100 s, cached per executable).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}
OUT=$REPO/logs/gate-$BR; mkdir -p "$OUT" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
cp "$RAW/md5.json" "$REPO/.scratch/prompts/"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-$BR
now() { date '+%T'; }
echo "[gate] start $(now) $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) commit $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null)"
r0() { env CARD=0 CNAME=ink-hip-specflip-run0 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
r1() { env CARD=1 CNAME=ink-hip-specflip-run1 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
seeds() {  # arm: off | def
  local f=$OUT/seeds-$1 e=""
  rm -f "$f.txt"
  [ "$1" = off ] && e="INTARSIA_SPEC_SAMPLED_DFLASH=0"
  r0 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=5 INTARSIA_SS_BUDGET_S=${SEEDS_BUDGET:-75} INTARSIA_SS_OUT=/work/logs/gate-$BR/seeds-$1.txt \
    /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$f.log" 2>&1
  echo "[gate c0] seeds $1 rc $? $(now) lines $(wc -l < "$f.txt" 2>/dev/null) | $(grep -aE 'INTARSIA_SPEC_SAMPLED_DFLASH=|DFlash-family drafting|draft length per request|requests in|^spec' "$f.log" | tr '\n' ' ' | cut -c1-400)"
}
md5() {  # arm: off | def
  local f=$OUT/md5-$1 x=""
  [ "$1" = off ] && x="--spec off"
  r1 $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 --prompt-ids /work/.scratch/prompts/md5.json $x > "$f.out" 2> "$f.err"
  echo "[gate c1] md5 $1 rc $? $(now) $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^decode overall|^spec: verify' "$f.err" | tr '\n' ' ' | cut -c1-200)"
}
srv() {  # card, name, mode (solo | conc), serve args: start, 16 replies, stop
  local c=$1 n=$2 mode=$3; shift 3
  local C=ink-hip-specflip-run$c ip="" i
  docker rm -f $C > /dev/null 2>&1
  ( r$c bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
  for i in $(seq 1 150); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && break
    ip=""; sleep 1
  done
  if [ -z "$ip" ]; then echo "[gate c$c] $n: no server after 150 s"; docker rm -f $C > /dev/null 2>&1; return 1; fi
  echo "[gate c$c] $n up after ${i}s $(now)"
  SB_FIT=1 timeout 100 python3 "$RAW/serve_bitwise.py" "http://$ip:8132/v1" $mode "$OUT/$n.json" > "$OUT/$n.client" 2>&1
  echo "[gate c$c] $n $mode rc $? $(now) | $(tail -n 1 "$OUT/$n.client")"
  docker rm -f $C > /dev/null 2>&1
  echo "[gate c$c] $n lines: $(grep -aE '^loaded|context|DSpark default|DFlash-family drafting|draft length per request|continuous batching|^--spec' "$OUT/$n.log" | tr '\n' ' ' | cut -c1-600)"
}
(
  seeds off; seeds def
  srv 0 b16 conc --max-batch 16
) > "$OUT/c0.log" 2>&1 &
(
  md5 off; md5 def
  srv 1 solo solo
  f=$OUT/sampled-def
  r1 $B generate -m $M --backend hip --max-tokens 64 --temperature 1.0 --seed 1 --prompt "Write a short story about a lighthouse." > "$f.out" 2> "$f.err"
  echo "[gate c1] sampled default rc $? $(now) | $(grep -aE '^loaded|DSpark default|DFlash-family drafting|draft length per request|^spec: verify|^decode overall' "$f.err" | tr '\n' ' ' | cut -c1-600)"
) > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
python3 "$RAW/serve_bitwise.py" compare "$OUT/solo.json" "$OUT/b16.json" > "$OUT/serve-compare.txt" 2>&1
echo "[gate] serve b16 (card 0) vs solo (card 1): $(tail -n 1 "$OUT/serve-compare.txt")"
if [ -s "$OUT/seeds-off.txt" ] && [ -s "$OUT/seeds-def.txt" ]; then
  a=$(sort "$OUT/seeds-off.txt"); b=$(sort "$OUT/seeds-def.txt")
  echo "[gate] seeds T 1.0: off $(echo "$a" | grep -c .) default $(echo "$b" | grep -c .) shared $(comm -12 <(echo "$a" | cut -d' ' -f1,2) <(echo "$b" | cut -d' ' -f1,2) | grep -c .) identical $(comm -12 <(echo "$a") <(echo "$b") | grep -c .)"
fi
echo "[gate] done $(now)"
