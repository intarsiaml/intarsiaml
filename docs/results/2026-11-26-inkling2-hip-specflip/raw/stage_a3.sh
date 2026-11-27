#!/usr/bin/env bash
# Lane ink-hip-specflip, option (a) window WA3, ON tatooine inside window.sh. BR = the (a) binary (both drafts), CTL = PR A (specA2).
# 1. Sampled contention ABBA, both cards at once, D2's protocol (#543's W2: speed-adapt.json, 128 tokens, --ignore-eos --top-k 0
#    --top-p 1.0 --seed 7, no --spec flag = each binary's default): arm off = CTL default (DSpark adaptive alone, flip A), arm pol = BR
#    default (the same DSpark adaptive with DFlash2 loaded beside it). (speed3.py's arm names: its "pol / off" = BR / CTL.) Card 0 T 1.0,
#    card 1 T 0.7, reps r1..r4 in the order off pol / pol off / pol off / off pol. Out: logs/a-sampled/<arm><T>-r<rep>.err.
# 2. After both cards are done: serve --max-batch 16 with BR (card 0; no drafter loads above one sequence), 16 greedy replies at once
#    (serve_bitwise.py conc), compared with PR A's b16 replies of window GS2 (logs/serve2-specA2/b16.json; GS2: = PR A solo 16 / 16).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}; CTL=${CTL:-specA2}
OUT=$REPO/logs/a-sampled; SV=$REPO/logs/a3-serve-$BR; mkdir -p "$OUT" "$SV" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
cp "$RAW/speed-adapt.json" "$REPO/.scratch/prompts/"
M=/models/Inkling-2-Mini
now() { date '+%T'; }
echo "[a3] start $(now) BR $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null) CTL $CTL $(md5sum "$REPO/.scratch/bin/intarsia-$CTL" | cut -c1-32)"
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
SP="--max-tokens 128 --ignore-eos --prompt-ids /work/.scratch/prompts/speed-adapt.json --top-k 0 --top-p 1.0 --seed 7"
arm() { # card arm temp-suffix temp rep
  local f=$OUT/$2$3-r$5 b=$BR
  [ $2 = off ] && b=$CTL
  date +%s.%N > $f.t0
  rc $1 /work/.scratch/bin/intarsia-$b $G $SP --temperature $4 > $f.out 2> $f.err
  local r=$?; date +%s.%N >> $f.t0; mv $f.t0 $f.t
  echo "[a3 c$1] $2$3 r$5 rc $r $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err) | $(grep -aE 'DFlash2 drafter|second draft|DFlash-family drafting' $f.err | tr '\n' ' ' | cut -c1-300)"
}
speed() { # card suffix temp
  arm $1 off $2 $3 1; arm $1 pol $2 $3 1
  arm $1 pol $2 $3 2; arm $1 off $2 $3 2
  arm $1 pol $2 $3 3; arm $1 off $2 $3 3
  arm $1 off $2 $3 4; arm $1 pol $2 $3 4
}
speed 0 1 1.0 > "$OUT/c0.log" 2>&1 &
speed 1 07 0.7 > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
# 2. b16 serve, card 0
B=/work/.scratch/bin/intarsia-$BR
docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1
( rc 0 bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink --max-batch 16" > "$SV/b16.log" 2>&1 ) &
ip=""
for i in $(seq 1 ${WAIT:-140}); do
  ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ink-hip-specflip-run0 2>/dev/null)
  [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && break
  ip=""; sleep 1
done
if [ -n "$ip" ]; then
  echo "[a3 c0] b16 up after ${i}s $(now)"
  SB_FIT=1 timeout 60 python3 "$RAW/serve_bitwise.py" "http://$ip:8132/v1" conc "$SV/b16.json" > "$SV/b16.client" 2>&1
  echo "[a3 c0] b16 conc rc $? $(now) | $(tail -n 1 "$SV/b16.client")"
else
  echo "[a3 c0] b16: no server after ${WAIT:-140} s $(now)"
fi
docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1
echo "[a3 c0] b16 lines: $(grep -aE '^admission|^loaded|DSpark default|DFlash2 drafter|second draft|continuous batching|^--spec|^dflash' "$SV/b16.log" | tr '\n' ' ' | cut -c1-600)"
python3 "$RAW/serve_bitwise.py" compare "$REPO/logs/serve2-specA2/b16.json" "$SV/b16.json" > "$SV/serve-compare.txt" 2>&1
echo "[a3] b16 $BR vs b16 $CTL (GS2): $(tail -n 1 "$SV/serve-compare.txt")"
echo "[a3] done $(now)"
