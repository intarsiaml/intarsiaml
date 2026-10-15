#!/bin/zsh
# Serve-path speed gap (inside one exclusive endor slot). localeval speed with the G2 args (--sizes 128,1024,4096 --gen 256
# --reps 3 --thinking off; serve --profile-every 0 --no-supervisor --no-ui --max-seq 8192 --served-model-name intarsia-metal
# --spec off), one fresh server per pass, macmon (100 ms) throughout, localeval lines timestamped. Passes (label:binary:variant):
#   a1-c556 a2-56ec a3-cbm0 a4-cbm0 a5-56ec a6-c556   (G2 args, order A B C C B A)
#   o1-56ec o2-56ec  (56ec2d5 with this task's earlier args: default --profile-every, --gen 128), first and last
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
HERE=${0:A:h}; P=$HERE/passes; mkdir -p $P
B=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
M=/Users/josef/models/Qwen3-8B; PORT=8093
export INTARSIA_CPU_FREE_MEM_GB=60 PYTHONUNBUFFERED=1
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; for b in c556008 base-56ec2d5 cbm0-8254311; do shasum -a 256 $B/intarsia-$b 2>/dev/null; done; macmon --version; sw_vers; } > $HERE/versions.txt 2>&1
macmon pipe -i 100 > $HERE/macmon.jsonl 2> $HERE/macmon.err &
MPID=$!
SPID=""
cleanup() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; kill $MPID 2>/dev/null; pkill -f "intarsia-.* serve .*--port $PORT" 2>/dev/null; }
trap cleanup EXIT INT TERM
run_pass() {  # label binary profile_flags gen
  local label=$1 bin=$2 prof=$3 gen=$4
  $bin serve --model $M --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
     ${=prof} --no-supervisor --no-ui --spec off > $P/$label-server.out 2> $P/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  echo "$label start $(date +%s.%N)" >> $HERE/events.log
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen $gen --reps 3 --thinking off --label cbm0-gap-$label 2>&1 | python3 $HERE/ts.py > $P/$label-localeval.ts )
  echo "$label end $(date +%s.%N)" >> $HERE/events.log
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  pkill -f "intarsia-.* serve .*--port $PORT" 2>/dev/null; sleep 8
  echo "$(date +%T) $label done"
}
G2="--profile-every 0"
run_pass o1-56ec $B/intarsia-base-56ec2d5 "" 128
run_pass a1-c556 $B/intarsia-c556008 "$G2" 256
run_pass a2-56ec $B/intarsia-base-56ec2d5 "$G2" 256
run_pass a3-cbm0 $B/intarsia-cbm0-8254311 "$G2 --max-batch 1" 256
run_pass a4-cbm0 $B/intarsia-cbm0-8254311 "$G2 --max-batch 1" 256
run_pass a5-56ec $B/intarsia-base-56ec2d5 "$G2" 256
run_pass a6-c556 $B/intarsia-c556008 "$G2" 256
run_pass o2-56ec $B/intarsia-base-56ec2d5 "" 128
echo ALLDONE
