#!/usr/bin/env bash
# Lane ink-hip-specflip, option (a) window WA3, ON tatooine inside window.sh. BR = the (a) binary (both drafts), CTL = the one-drafter
# control (11-27 morning: main7 = origin/main 4b629184c, the merge base of the (a) head; was PR A specA2 before the main7 merge).
# PART = sampled | serve | all (default all; 11-27: BR = dualM's b16 start is cold, so WA3 runs as two windows, sampled then serve).
# 1. (sampled) Sampled contention ABBA, both cards at once, D2's protocol (#543's W2: speed-adapt.json, 128 tokens, --ignore-eos
#    --top-k 0 --top-p 1.0 --seed 7, no --spec flag = each binary's default): arm off = CTL default (DSpark adaptive alone, flip A), arm
#    pol = BR default (the same DSpark adaptive with DFlash2 loaded beside it). (speed3.py's arm names: its "pol / off" = BR / CTL.)
#    Card 0 T 1.0, card 1 T 0.7, reps r1..r4 in the order off pol / pol off / pol off / off pol. Out: logs/a-sampled/<arm><T>-r<rep>.err.
# 2. (serve) serve --max-batch 16 with BR (card 0; no drafter loads above one sequence), 16 greedy replies at once (serve_bitwise.py
#    conc), compared with REFS (default: main7's b16 replies of window M1, logs/m7a-main7/b16.json, and PR A's of window GS2,
#    logs/serve2-specA2/b16.json; both = their solo 16 / 16). Card 1 meanwhile (untimed): greedy md5.json 256 tokens --ignore-eos, BR
#    `--spec off` vs BR default (the DFlash2 k4 greedy default): ids md5 must be equal, and = 380af8a5 (main7's).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}; CTL=${CTL:-main7}; PART=${PART:-all}
REFS=${REFS:-m7a-main7 serve2-specA2}
OUT=$REPO/logs/a-sampled-$BR; SV=$REPO/logs/a3-serve-$BR; mkdir -p "$OUT" "$SV" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
cp "$RAW/speed-adapt.json" "$RAW/md5.json" "$REPO/.scratch/prompts/"
M=/models/Inkling-2-Mini
now() { date '+%T'; }
echo "[a3] start $(now) PART $PART BR $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null) CTL $CTL $(md5sum "$REPO/.scratch/bin/intarsia-$CTL" | cut -c1-32)"
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
if [ "$PART" != serve ]; then
  speed 0 1 1.0 > "$OUT/c0.log" 2>&1 &
  speed 1 07 0.7 > "$OUT/c1.log" 2>&1 &
  wait
  cat "$OUT/c0.log" "$OUT/c1.log"
fi
[ "$PART" = sampled ] && { echo "[a3] done $(now)"; exit 0; }
# 2. b16 serve, card 0; greedy md5 on card 1 meanwhile
B=/work/.scratch/bin/intarsia-$BR
md5() {  # name extra
  local f=$SV/md5-$1
  rc 1 $B $G --ignore-eos --max-tokens 256 --prompt-ids /work/.scratch/prompts/md5.json $2 > "$f.out" 2> "$f.err"
  echo "[a3 c1] md5 $1 rc $? $(now) $(cat "$f.out" "$f.err" | grep -a '^ids\[' | md5sum | cut -c1-8) | $(grep -aE '^decode overall|^spec: verify|DFlash2 drafter|second draft' "$f.err" | tr '\n' ' ' | cut -c1-300)"
}
( md5 off "--spec off"; md5 def "" ) > "$SV/c1.log" 2>&1 &
MD5_PID=$!
docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1
( rc 0 bash -c "exec $B serve -m $M --backend hip --host 0.0.0.0 --port 8132 --model-id ink --max-batch 16" > "$SV/b16.log" 2>&1 ) &
ip=""
for i in $(seq 1 ${WAIT:-200}); do
  ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ink-hip-specflip-run0 2>/dev/null)
  [ -n "$ip" ] && curl -sf -m 2 "http://$ip:8132/health" > /dev/null 2>&1 && break
  ip=""; sleep 1
done
if [ -n "$ip" ]; then
  echo "[a3 c0] b16 up after ${i}s $(now)"
  SB_FIT=1 timeout 60 python3 "$RAW/serve_bitwise.py" "http://$ip:8132/v1" conc "$SV/b16.json" > "$SV/b16.client" 2>&1
  echo "[a3 c0] b16 conc rc $? $(now) | $(tail -n 1 "$SV/b16.client")"
else
  echo "[a3 c0] b16: no server after ${WAIT:-200} s $(now)"
fi
docker rm -f ink-hip-specflip-run0 > /dev/null 2>&1
echo "[a3 c0] b16 lines: $(grep -aE '^admission|^loaded|DSpark default|DFlash2 drafter|second draft|continuous batching|^--spec|^dflash' "$SV/b16.log" | tr '\n' ' ' | cut -c1-600)"
for r in $REFS; do
  python3 "$RAW/serve_bitwise.py" compare "$REPO/logs/$r/b16.json" "$SV/b16.json" > "$SV/serve-compare-$r.txt" 2>&1
  echo "[a3] b16 $BR vs b16 $r: $(tail -n 1 "$SV/serve-compare-$r.txt")"
done
wait $MD5_PID
cat "$SV/c1.log"
a=$(cat "$SV/md5-off.out" "$SV/md5-off.err" | grep -a '^ids\[' | md5sum | cut -c1-8); b=$(cat "$SV/md5-def.out" "$SV/md5-def.err" | grep -a '^ids\[' | md5sum | cut -c1-8)
echo "[a3] greedy md5 256: off $a default $b: $([ "$a" = "$b" ] && echo EQUAL || echo DIFFER) (main7: 380af8a5)"
echo "[a3] done $(now)"
