# sourced by the slot scripts: environment asserts, fleetq slot, macmon clock log, temperature guard
: ${DEVELOPER_DIR:?DEVELOPER_DIR must be set}
[ "$DEVELOPER_DIR" = /Applications/Xcode.app/Contents/Developer ] || { echo "DEVELOPER_DIR=$DEVELOPER_DIR, expected the Xcode one"; exit 3; }
: ${FLEETQ_JOB:?run inside a fleetq slot}
W=/opt/code/intarsiaml-longctx
R=$W/docs/results/raw/2026-10-15-metal-longctx-profile
BIN=$W/target/release/intarsia
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
macmon_start() { # $1 = jsonl path
  /Users/josef/bin/macmon pipe -i 100 > $1 2> $1.err &
  MACMON_PID=$!
}
macmon_stop() { kill $MACMON_PID 2>/dev/null; }
# last GPU temperature from the macmon log (deg C), 0 if unknown
gpu_temp() { tail -n 3 $1 2>/dev/null | python3 -c '
import sys, json
t = 0
for l in sys.stdin:
    try: t = json.loads(l)["temp"]["gpu_temp_avg"]
    except Exception: pass
print(int(t))'; }
# wait until the GPU is below 80 C (max 2 min); logs what it did
cooldown() { # $1 = macmon jsonl, $2 = label
  local t=$(gpu_temp $1) n=0
  echo "temp before $2: ${t} C" | tee -a $R/temps.log
  while [ $t -gt ${COOL:-80} ] && [ $n -lt ${COOLN:-12} ]; do sleep 10; t=$(gpu_temp $1); n=$((n+1)); done
  [ $n -gt 0 ] && echo "  cooled to ${t} C after $((n*10)) s" | tee -a $R/temps.log
}
idle3() { for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done | tr '\n' ' '; echo; }
