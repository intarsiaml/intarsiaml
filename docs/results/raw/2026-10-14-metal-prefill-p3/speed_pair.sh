#!/bin/zsh
# Metal P3: localeval speed pre/new, order pre new new pre (ABBA), one exclusive fleetq slot.
# pre = spike/metal 2e1fdb1 build (chunk 256), new = P3 build (chunk 2048 from recipes/qwen3.toml [tuning.metal]).
# Same server flags, localeval args and GPU-idle/zoom recording as docs/results/raw/2026-10-14-metal-phase-b/sameslot.sh.
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9
OUT=$W/docs/results/raw/2026-10-14-metal-prefill-p3/speed
PRE=$S/p3/intarsia-base
NEW=$S/p3/intarsia-p3
MODEL=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 30 --label "metal P3: localeval speed pre/new (ABBA)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
SPID=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; }
trap "cleanup; fleetq release $J" EXIT
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $J"; echo "pre: $(ls -l $PRE)"; echo "new: $(ls -l $NEW)"; sw_vers | tr '\n' ' '; echo; } > $OUT/versions.txt

gpu_check() {
  local f=$OUT/$1-gpu-idle.txt; : > $f
  local tries=0 med
  while true; do
    echo "== sample set $(date -Iseconds)" >> $f
    local vals=()
    for i in 1 2 3; do
      v=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1)
      echo "$v" >> $f; vals+=(${v##*=}); sleep 1
    done
    med=${${(on)vals}[2]}
    echo "median=$med" >> $f
    echo "zoom: $(pgrep -fl zoom.us | tr '\n' ';')" >> $f
    if (( med <= 10 )) || (( tries >= 12 )); then break; fi
    echo "idle util $med > 10, waiting 10s (try $tries)" >> $f; tries=$((tries+1)); sleep 10
  done
  echo "FINAL median=$med tries=$tries" >> $f
  cat $f
}

run() { # label binary
  local label=$1 bin=$2 port=8093
  echo "##### $label"
  $bin serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port $port --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do
    curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  echo "server up after ${i}s; chunk line: $(grep -h 'prefill chunk' $OUT/$label-server.err | head -1)"
  gpu_check $label
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label ) > $OUT/$label-localeval.out 2> $OUT/$label-localeval.err
  echo "localeval exit $?"
  tail -30 $OUT/$label-localeval.out
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null
  SPID=""
  sleep 5
}

run p3speed-pre-1 $PRE
run p3speed-new-1 $NEW
run p3speed-new-2 $NEW
run p3speed-pre-2 $PRE
echo ALLDONE
