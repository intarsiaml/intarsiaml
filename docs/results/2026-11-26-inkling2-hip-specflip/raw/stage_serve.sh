#!/usr/bin/env bash
# Lane ink-hip-specflip, PR A gate 2 serve recheck (window G2), ON tatooine inside window.sh: G1's serve part did not run (the client URL
# lacked /v1, and the solo server's first-start admission passed 150 s beside the b16 one). Binary $BR, both cards at once:
# card 0 serve --max-batch 16 (admission cached by G1), card 1 serve solo (the default: DSpark k5; first start runs its admission);
# the same 16 greedy replies (serve_bitwise.py, 128 tokens): b16 must equal solo. The context / KV lines of both loads. Then on card 0
# main's binary $MAIN (main 3fb1b5f8d) serve --max-batch 16 until its `loaded ... context` line (first start: admission), for the
# b16 context before flip A.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}
OUT=$REPO/logs/serve-$BR; mkdir -p "$OUT" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-$BR
now() { date '+%T'; }
echo "[gate] start $(now) $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) commit $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null)"
r0() { env CARD=0 CNAME=ink-hip-specflip-run0 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
r1() { env CARD=1 CNAME=ink-hip-specflip-run1 bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
srv() {  # card, name, mode (solo | conc), serve args: start, 16 replies, stop
  local c=$1 n=$2 mode=$3; shift 3
  local C=ink-hip-specflip-run$c ip="" i
  docker rm -f $C > /dev/null 2>&1
  ( r$c bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
  for i in $(seq 1 200); do
    ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $C 2>/dev/null)
    [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && break
    ip=""; sleep 1
  done
  if [ -z "$ip" ]; then echo "[gate c$c] $n: no server after 200 s"; docker rm -f $C > /dev/null 2>&1; return 1; fi
  echo "[gate c$c] $n up after ${i}s $(now)"
  timeout 100 python3 "$RAW/serve_bitwise.py" "http://$ip:8132/v1" $mode "$OUT/$n.json" > "$OUT/$n.client" 2>&1
  echo "[gate c$c] $n $mode rc $? $(now) | $(tail -n 1 "$OUT/$n.client")"
  docker rm -f $C > /dev/null 2>&1
  echo "[gate c$c] $n lines: $(grep -aE '^loaded|context|DSpark default|DFlash-family drafting|draft length per request|continuous batching|^--spec' "$OUT/$n.log" | tr '\n' ' ' | cut -c1-600)"
}

ctx() {  # card, bin name, name, serve args: start, wait for the `loaded` line (<= 170 s), stop
  local c=$1 b=$2 n=$3 i; shift 3
  local C=ink-hip-specflip-run$c
  docker rm -f $C > /dev/null 2>&1
  ( r$c bash -c "exec /work/.scratch/bin/intarsia-$b serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink $*" > "$OUT/$n.log" 2>&1 ) &
  for i in $(seq 1 170); do grep -aq '^loaded ' "$OUT/$n.log" 2>/dev/null && break; sleep 1; done
  docker rm -f $C > /dev/null 2>&1
  echo "[gate c$c] $n ($b $(cat "$REPO/.scratch/bin/intarsia-$b.commit" 2>/dev/null | cut -c1-9)) after ${i}s: $(grep -aE '^admission|^continuous batching|^loaded ' "$OUT/$n.log" | tr '\n' ' ')"
}
( srv 0 b16 conc --max-batch 16; ctx 0 ${MAIN:-mainA} main-b16 --max-batch 16 ) > "$OUT/c0.log" 2>&1 &
( srv 1 solo solo ) > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
python3 "$RAW/serve_bitwise.py" compare "$OUT/solo.json" "$OUT/b16.json" > "$OUT/serve-compare.txt" 2>&1
cat "$OUT/serve-compare.txt"
echo "[serve] done $(now)"
