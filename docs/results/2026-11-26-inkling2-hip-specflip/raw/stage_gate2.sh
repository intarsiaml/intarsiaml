#!/usr/bin/env bash
# Lane ink-hip-specflip, gate window v2 (gate 2 of the brief for the final heads), ON tatooine inside window.sh. stage_gate.sh with the
# serves first (each binary's first `serve` runs the plan admission, ~100-125 s cold: D2 / `generate` does not warm it) and knobs:
#   BR       binary (.scratch/bin/intarsia-$BR, test exe t-$BR-inkling2_sampled_real)
#   SEED_DRAFT  the seed harness's --spec-draft (unset: the model dir = DSpark; PR B: /models/Inkling-2-Mini-DFlash2), SEED_K (default 5)
#   MD5_REV=1   a third greedy md5 arm `rev`: INTARSIA_SPEC_DFLASH2_DEFAULT=0 (PR B's revert: the DSpark default), must equal `off`
# Card 0: serve --max-batch 16 (16 greedy replies at once), then 100 seeds at T 1.0: drafting off (=0) vs the default (unset).
# Card 1: serve solo (the default), the same 16 replies one at a time; then greedy md5 off / def [/ rev] (md5.json, 256 tokens); then one
#   default sampled `generate` (T 1.0) for the startup lines. b16 (card 0) vs solo (card 1).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}
OUT=$REPO/logs/gate2-$BR; mkdir -p "$OUT" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
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
  [ -n "${SEED_DRAFT:-}" ] && e="$e INTARSIA_SS_DRAFT=$SEED_DRAFT"
  r0 env $e INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0 \
    INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=${SEED_K:-5} INTARSIA_SS_BUDGET_S=${SEEDS_BUDGET:-75} INTARSIA_SS_OUT=/work/logs/gate2-$BR/seeds-$1.txt \
    /work/.scratch/bin/t-$BR-inkling2_sampled_real --nocapture > "$f.log" 2>&1
  echo "[gate c0] seeds $1 rc $? $(now) lines $(wc -l < "$f.txt" 2>/dev/null) | $(grep -aE 'INTARSIA_SPEC_SAMPLED_DFLASH=|DFlash-family drafting|draft length per request|requests in|draft of|^spec' "$f.log" | tr '\n' ' ' | cut -c1-400)"
}
md5() {  # arm: off | def | rev
  local f=$OUT/md5-$1 x="" e=""
  [ "$1" = off ] && x="--spec off"
  [ "$1" = rev ] && e="INTARSIA_SPEC_DFLASH2_DEFAULT=0"
  r1 env $e $B generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --ignore-eos --max-tokens 256 --prompt-ids /work/.scratch/prompts/md5.json $x > "$f.out" 2> "$f.err"
  echo "[gate c1] md5 $1 rc $? $(now) $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5sum | cut -c1-32) | $(grep -aE '^decode overall|^spec: verify|DFlash2 default|DSpark default|^--spec dflash' "$f.err" | tr '\n' ' ' | cut -c1-420)"
}
srv() {  # card, name, mode (solo | conc), serve args: start, 16 replies, stop
  local c=$1 n=$2 mode=$3; shift 3
  local C=ink-hip-specflip-run$c ip="" i
  docker rm -f $C > /dev/null 2>&1
  ( r$c bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
  for i in $(seq 1 175); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && break
    ip=""; sleep 1
  done
  if [ -z "$ip" ]; then echo "[gate c$c] $n: no server after 175 s"; docker rm -f $C > /dev/null 2>&1; return 1; fi
  echo "[gate c$c] $n up after ${i}s $(now)"
  SB_FIT=1 timeout 100 python3 "$RAW/serve_bitwise.py" "http://$ip:8132/v1" $mode "$OUT/$n.json" > "$OUT/$n.client" 2>&1
  echo "[gate c$c] $n $mode rc $? $(now) | $(tail -n 1 "$OUT/$n.client")"
  docker rm -f $C > /dev/null 2>&1
  echo "[gate c$c] $n lines: $(grep -aE '^admission|^loaded|DSpark default|DFlash2 default|DFlash-family drafting|draft length per request|continuous batching|^--spec' "$OUT/$n.log" | tr '\n' ' ' | cut -c1-600)"
}
(
  srv 0 b16 conc --max-batch 16
  seeds off; seeds def
) > "$OUT/c0.log" 2>&1 &
(
  srv 1 solo solo
  md5 off; md5 def
  [ -n "${MD5_REV:-}" ] && md5 rev
  f=$OUT/sampled-def
  r1 $B generate -m $M --backend hip --max-tokens 64 --temperature 1.0 --seed 1 --prompt "Write a short story about a lighthouse." > "$f.out" 2> "$f.err"
  echo "[gate c1] sampled default rc $? $(now) | $(grep -aE '^loaded|DSpark default|DFlash2 default|^--spec dflash|DFlash-family drafting|draft length per request|^spec: verify|^decode overall' "$f.err" | tr '\n' ' ' | cut -c1-700)"
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
