#!/bin/zsh
# Cold serve prefill at 4062 tokens, paired: new default (serve pre-wires) vs INTARSIA_METAL_PREWIRE=0 vs base 00abb36. Exclusive slot.
# Each sample: start `intarsia serve` (profile cb trace, chunk 256 default), wait for /health, wait GAP seconds, send ONE 4062-token
# completion (max_tokens 1, temperature 0), time it with curl, stop the server. GAP 0 is the worst case for the pre-wire.
# serve-cold.sh <outdir> <rounds>
set -u
zmodload zsh/datetime
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; N=$2; mkdir -p $O
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
macmon pipe -i 200 > $O/macmon.jsonl 2>/dev/null & MP=$!
SPID=""
cleanup() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; kill $MP 2>/dev/null; }
trap cleanup EXIT
python3 - > $O/body.json <<PY
import json
print(json.dumps({"model": "qwen3", "prompt": open("$R/prompt-4062.txt").read(), "max_tokens": 1, "temperature": 0}))
PY
: > $O/samples.txt
sample() { # arm gap round
  local arm=$1 gap=$2 r=$3 bin envv
  case $arm in
    base) bin=$SC/intarsia-base2; envv=(A=1);;
    new) bin=$SC/intarsia-t1; envv=(A=1);;
    newoff) bin=$SC/intarsia-t1; envv=(INTARSIA_METAL_PREWIRE=0);;
  esac
  local F=$O/$arm-gap$gap-r$r
  wait_idle || echo "$arm gap$gap r$r not idle" >> $O/samples.txt
  env $envv INTARSIA_METAL_PROFILE=cb:$F.trace.jsonl $bin serve --model $M --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --port 8102 > $F.server.log 2>&1 &
  SPID=$!
  local up=0 t_start=$EPOCHREALTIME
  for i in {1..240}; do
    curl -sf http://127.0.0.1:8102/health > /dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 0.25
  done
  [ $up = 1 ] || { echo "$arm gap$gap r$r server failed" >> $O/samples.txt; tail -3 $F.server.log >> $O/samples.txt; SPID=""; return; }
  local t_up=$EPOCHREALTIME
  [ $gap != 0 ] && sleep $gap
  local out=$(curl -s -o $F.resp.json -w "%{time_total} %{http_code}" -H 'content-type: application/json' --data @$O/body.json http://127.0.0.1:8102/v1/completions)
  kill $SPID; wait $SPID 2>/dev/null; SPID=""
  echo "$arm gap$gap r$r load_s $(( t_up - t_start )) curl_s_code $out" >> $O/samples.txt
}
ORDERS=("base new newoff" "new newoff base" "newoff base new" "newoff new base" "base newoff new" "new base newoff")
for r in $(seq 1 $N); do
  ARMS=(${=ORDERS[$(( (r - 1) % 6 + 1 ))]})
  for gap in 0 3; do for a in $ARMS; do sample $a $gap $r; done; done
  echo "round $r done $(date +%T)"
done
echo SERVECOLD_DONE
