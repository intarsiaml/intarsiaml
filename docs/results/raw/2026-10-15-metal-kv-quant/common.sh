# sourced by the slot work scripts
: ${DEVELOPER_DIR:?DEVELOPER_DIR must be set}
[ "$DEVELOPER_DIR" = /Applications/Xcode.app/Contents/Developer ] || { echo "DEVELOPER_DIR=$DEVELOPER_DIR, expected the Xcode one"; exit 3; }
: ${FLEETQ_JOB:?run inside a fleetq slot}
W=/opt/code/intarsiaml-kvq
R=$W/docs/results/raw/2026-10-15-metal-kv-quant
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
BASE=$S/intarsia-base          # spike/metal 56ec2d5 release build
NEW=$S/intarsia-new            # this branch, copied before the slot
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden  # fp32 goldens (git-ignored, main checkout); baselines come from the worktree's golden/baselines
PG=$W/docs/results/raw/2026-10-15-metal-spec-g1/prompts
PL=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
export INTARSIA_CPU_FREE_MEM_GB=60
MACMON=/Users/josef/bin/macmon
macmon_start() { $MACMON pipe -i 100 > $1 2> $1.err & MACMON_PID=$!; }
macmon_stop() { kill $MACMON_PID 2>/dev/null; }
gpu_temp() { tail -n 3 $1 2>/dev/null | python3 -c '
import sys, json
t = 0
for l in sys.stdin:
    try: t = json.loads(l)["temp"]["gpu_temp_avg"]
    except Exception: pass
print(int(t))'; }
cooldown() { # $1 = macmon jsonl, $2 = label: wait until the GPU is below ${COOL:-75} C (max ${COOLN:-9} x 10 s)
  local t=$(gpu_temp $1) n=0
  while [ $t -gt ${COOL:-75} ] && [ $n -lt ${COOLN:-9} ]; do sleep 10; t=$(gpu_temp $1); n=$((n+1)); done
  echo "temp before $2: ${t} C (waited $((n*10)) s)"
}
