#!/bin/zsh
# First 4k request after an idle gap, in a server that is already running; paired arms. serve-idle.sh <outdir> <rounds> <arms...>
# serve-idle2: the final binary (intarsia-new, serve defaults = residency keep-warm). NOPROF=1: no cb trace (decode ITL control); IDLES="3 40".
# arms: base (6d99a6f) | kw (this branch, serve defaults: pre-wire + blit keep-warm) | kwb (blit keep-warm, no pre-wire) |
# rs (residency-set keep-warm, no pre-wire) | off (this binary, both off). R0=<n> starts at round n.
# Per sample: start `intarsia serve` (profile cb trace), wait /health, one warm-up request (4k, 1 token), then for each idle in
# 3 10 40 s: sleep, send one streamed 4k request (33 tokens) with a nonce no earlier request shares.  Same nonce for every arm
# of a round, so the generated text can be compared across arms.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; N=$2; shift 2; ARMS=("$@"); mkdir -p $O
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/resid
R=/opt/code/intarsiaml-residency/docs/results/raw/2026-10-16-metal-weight-residency
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
SPID=""
trap '[ -n "$SPID" ] && kill $SPID 2>/dev/null' EXIT
sample() { # arm round
  local arm=$1 r=$2 bin envv
  case $arm in
    base) bin=$SC/intarsia-base; envv=(A=1);;
    off) bin=$SC/intarsia-new; envv=(INTARSIA_METAL_KEEPWARM=0);;
    blit) bin=$SC/intarsia-new; envv=(INTARSIA_METAL_KEEPWARM_MODE=blit);;
    rs) bin=$SC/intarsia-new; envv=(A=1);;
  esac
  local F=$O/$arm-r$r
  wait_idle || echo "$arm r$r not idle" >> $O/samples.txt
  local prof=(A=1); [ -z "${NOPROF:-}" ] && prof=(INTARSIA_METAL_PROFILE=cb:$F.trace.jsonl)
  env $envv $prof $bin serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --port 8102 > $F.server.log 2>&1 &
  SPID=$!
  local up=0
  for i in {1..240}; do
    curl -sf http://127.0.0.1:8102/health > /dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 0.25
  done
  [ $up = 1 ] || { echo "$arm r$r server failed" >> $O/samples.txt; tail -3 $F.server.log >> $O/samples.txt; SPID=""; return; }
  python3 $R/req.py 8102 "warm-$r" 1 > $F.warm.json 2>&1
  for idle in ${=IDLES:-3 10 40}; do
    sleep $idle
    python3 $R/req.py 8102 "n-$r-$idle" 33 > $F.idle$idle.json 2>&1
  done
  kill $SPID; wait $SPID 2>/dev/null; SPID=""
  echo "$arm r$r done" >> $O/samples.txt
}
for r in $(seq ${R0:-1} $N); do
  # rotate the arm order each round
  k=$(( (r - 1) % ${#ARMS[@]} ))
  ORD=(${ARMS[@]:$k} ${ARMS[@]:0:$k})
  for a in $ORD; do sample $a $r; done
  echo "round $r done $(date +%T)"
done
echo SERVEIDLE_DONE
