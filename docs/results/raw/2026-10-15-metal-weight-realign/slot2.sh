#!/bin/zsh
# metal-weight-realign slot 2 (performance): G0 multi-row bench (aligned vs 8-mod-16, macmon), then serve runs
# BASE(c556008) -> ON -> ON -> BASE with localeval speed and footprint/vmmap snapshots. One exclusive fleetq slot on endor.
S=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad
W=/opt/code/intarsiaml-weight-realign
OUT=$W/docs/results/raw/2026-10-15-metal-weight-realign/slot2
BASE=$S/intarsia-c556008; BIN=$S/intarsia-realign; G0=$S/gemv_g0
G0DIR=$W/docs/results/raw/2026-10-15-metal-g0-multirow
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $OUT/bench $OUT/serve
J=$(fleetq acquire --host endor --exclusive --mem 40 --expect 60 --label "metal-weight-realign: slot2 G0 bench + decode tok/s + footprint" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $OUT/fleetq-job.txt
SPID=""; MM=""
cleanup() { [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MM" ]] && kill $MM 2>/dev/null; fleetq release $J; }
trap cleanup EXIT
export FLEETQ_JOB=$J
{ echo "date: $(date -Iseconds)"; git -C $W rev-parse HEAD; md5 $BASE $BIN $G0; sw_vers | tr '\n' ' '; echo; echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml)"; } > $OUT/versions.txt 2>&1

echo "== A. G0 bench"
sleep 15
/Users/josef/bin/macmon pipe -i 100 > $OUT/bench/macmon.jsonl 2>/dev/null &
MM=$!
sleep 2
export G0_ROWS=2,4,6,8
for run in 1 2; do
  G0_OUT=$OUT/bench/g0-aligned-run$run.csv $G0 qo down kv gu > $OUT/bench/g0-aligned-run$run.log 2>&1; echo "bench aligned run$run exit $?"
done
G0_A8=1 G0_OUT=$OUT/bench/g0-a8.csv $G0 qo down kv gu > $OUT/bench/g0-a8.log 2>&1; echo "bench a8 exit $?"
sleep 2; kill $MM; MM=""
unset G0_ROWS
python3 $G0DIR/analyze_g0.py $OUT/bench > $OUT/bench/analysis.md 2> $OUT/bench/analysis.err; echo "analyze exit $?"

echo "== B. serve runs"
gpu_check() {
  local f=$OUT/serve/$1-gpu-idle.txt; : > $f; local tries=0 med
  while true; do
    local vals=()
    for i in 1 2 3; do v=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1); echo "$v" >> $f; vals+=(${v##*=}); sleep 1; done
    med=${${(on)vals}[2]}; echo "median=$med zoom: $(pgrep -fl zoom.us | tr '\n' ';')" >> $f
    if (( med <= 10 )) || (( tries >= 12 )); then break; fi
    tries=$((tries+1)); sleep 10
  done
  echo "FINAL median=$med tries=$tries" >> $f
}
snap() { # label tag
  local d=$OUT/serve/$1-$2; footprint $SPID > $d-footprint.txt 2>&1; vmmap -summary $SPID > $d-vmmap.txt 2>&1
  ps -o rss=,vsz= -p $SPID > $d-ps.txt; vm_stat > $d-vmstat.txt; memory_pressure 2>/dev/null | tail -3 > $d-pressure.txt
}
run_engine() { # label envflag binary
  local label=$1 flag=$2 bin=$3
  echo "##### $label $(date -Iseconds)"
  local t0=$(python3 -c 'import time; print(time.time())')
  env INTARSIA_METAL_REALIGN=$flag INTARSIA_METAL_DEBUG=1 $bin serve --model $M --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui > $OUT/serve/$label-server.out 2> $OUT/serve/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do curl -sf http://127.0.0.1:8093/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  echo "server up after ${i}s; $(grep -h loaded $OUT/serve/$label-server.err | head -1 | cut -c1-120)"
  sleep 5; snap $label after-load
  gpu_check $label
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8093/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label ) > $OUT/serve/$label-localeval.out 2> $OUT/serve/$label-localeval.err
  echo "localeval exit $?"; tail -25 $OUT/serve/$label-localeval.out
  snap $label after-sweep
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null; SPID=""; sleep 5
}
run_engine base-1 0 $BASE
run_engine on-1 1 $BIN
run_engine on-2 1 $BIN
run_engine base-2 0 $BASE
echo ALLDONE
