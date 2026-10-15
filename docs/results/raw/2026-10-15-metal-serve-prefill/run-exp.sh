#!/bin/zsh
# usage: run-exp.sh LABEL BIN ROUNDS [extra env assignments via ENVX="A=1 B=2"] ; runs inside ONE fleetq slot.
# Starts a fresh server, runs exp.py, writes <OUT>/<LABEL>.tsv
LABEL=$1; BIN=$2; ROUNDS=$3
W=/opt/code/intarsiaml/.claude/worktrees/agent-ad08dc8ef0ba4269b
OUT=$W/docs/results/raw/2026-10-15-metal-serve-prefill
MODEL=/Users/josef/models/Qwen3-8B
PROMPT=/opt/code/intarsiaml/docs/results/raw/2026-10-14-metal-phase-b/prompt-4k.txt
[[ -f $PROMPT ]] || PROMPT=$W/docs/results/raw/2026-10-14-metal-phase-b/prompt-4k.txt
export INTARSIA_CPU_FREE_MEM_GB=60
J=${FLEETQ_JOB:-}
OWN=0
if [[ -z $J ]]; then
  J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 30 --label "metal serve-prefill $LABEL" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
  OWN=1
fi
echo "fleetq job $J" | tee -a $OUT/fleetq-jobs.txt
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; (( OWN )) && fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J
pgrep -fl zoom.us && echo "ZOOM RUNNING"
echo "date $(date -Iseconds) bin=$BIN env=$ENVX" | tee $OUT/$LABEL.meta
env ${=ENVX} $BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$LABEL-server.out 2> $OUT/$LABEL-server.err &
SPID=$!
for i in {1..120}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && break; sleep 1; done
grep -h 'prefill chunk' $OUT/$LABEL-server.err | head -1
echo "gpu idle: $(ioreg -r -c IOAccelerator -d 1 | grep -o 'Device Utilization %"=[0-9]*' | head -1)"
(macmon pipe -i 100 | python3 -I $OUT/macmon_log.py > $OUT/$LABEL-macmon.tsv) &
MPID=$!
python3 -I $OUT/exp.py 8093 intarsia-metal $PROMPT $OUT/$LABEL.tsv $ROUNDS ${EXPX:-} 2>&1 | tail -n 200 > $OUT/$LABEL.log
pkill -f "macmon pipe" ; kill $SPID; wait $SPID 2>/dev/null; SPID=""
echo done
