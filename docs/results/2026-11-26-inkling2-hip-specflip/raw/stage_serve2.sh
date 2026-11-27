#!/usr/bin/env bash
# Lane ink-hip-specflip, PR A serve-only rerun (coordinator 00:27 EST 11-27: one try, 285 s cap), ON tatooine inside window.sh.
# GA2's solo server did not finish its cold first-start admission in 175 s beside the b16 one. Here: solo (card 1) starts FIRST and
# ALONE, up to 240 s for its admission; once it answers, b16 (card 0, admission cached by GA2) starts while solo serves its 16 replies
# one at a time; then b16's 16 replies at once. b16 vs solo (serve_bitwise.py compare). BR = binary (.scratch/bin/intarsia-$BR).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}
OUT=$REPO/logs/serve2-$BR; mkdir -p "$OUT" "$REPO/.scratch/home"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-$BR
now() { date '+%T'; }
echo "[serve2] start $(now) $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) commit $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null)"
r0() { env CARD=0 CNAME=ink-hip-specflip-run0 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
r1() { env CARD=1 CNAME=ink-hip-specflip-run1 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
start() {  # card, name, serve args
  local c=$1 n=$2; shift 2
  docker rm -f ink-hip-specflip-run$c > /dev/null 2>&1
  ( r$c bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
}
up() {  # card, name, wait s -> prints the ip
  local c=$1 n=$2 w=$3 ip="" i
  for i in $(seq 1 "$w"); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ink-hip-specflip-run$c 2>/dev/null)
    [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && { echo "[serve2 c$c] $n up after ${i}s $(now)" >&2; echo "$ip"; return 0; }
    sleep 1
  done
  echo "[serve2 c$c] $n: no server after $w s $(now)" >&2; return 1
}
client() {  # card, name, mode, ip
  SB_FIT=1 timeout 100 python3 "$RAW/serve_bitwise.py" "http://$4:8132/v1" $3 "$OUT/$2.json" > "$OUT/$2.client" 2>&1
  echo "[serve2 c$1] $2 $3 rc $? $(now) | $(tail -n 1 "$OUT/$2.client")"
  docker rm -f ink-hip-specflip-run$1 > /dev/null 2>&1
  echo "[serve2 c$1] $2 lines: $(grep -aE '^admission|^loaded|DSpark default|DFlash-family drafting|draft length per request|continuous batching|^--spec' "$OUT/$2.log" | tr '\n' ' ' | cut -c1-600)"
}
start 1 solo
if ip1=$(up 1 solo 240); then
  start 0 b16 --max-batch 16
  client 1 solo solo "$ip1"
  if ip0=$(up 0 b16 40); then client 0 b16 conc "$ip0"; else docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1; fi
else
  docker rm -f ink-hip-specflip-run1 > /dev/null 2>&1
fi
python3 "$RAW/serve_bitwise.py" compare "$OUT/solo.json" "$OUT/b16.json" > "$OUT/serve-compare.txt" 2>&1
echo "[serve2] b16 (card 0) vs solo (card 1): $(tail -n 1 "$OUT/serve-compare.txt")"
echo "[serve2] done $(now)"
