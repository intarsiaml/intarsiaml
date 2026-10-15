#!/bin/zsh
# G2 (inside one exclusive slot, FLEETQ_JOB set by slot.sh): localeval speed on a fresh `intarsia serve` per pass,
# order ABBA = spec off, spec on, spec on, spec off. macmon (100 ms) runs throughout; /metrics spec counters are polled
# every second. Same binary for all passes (--spec off|ngram).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
BIN=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/bin/intarsia-gate-stamped
OUT=/opt/code/intarsiaml-spec/docs/results/raw/2026-10-15-metal-spec-g2g3/g2
P=$OUT/passes; mkdir -p $P
MODEL=/Users/josef/models/Qwen3-8B
PORT=8093
export INTARSIA_CPU_FREE_MEM_GB=60 PYTHONUNBUFFERED=1
{ echo "date: $(date -Iseconds)"; echo "fleetq job: $FLEETQ_JOB"; echo "branch head: $(git -C /opt/code/intarsiaml-spec rev-parse HEAD)"
  ls -l $BIN; shasum -a 256 $BIN; echo "localeval: $(grep -m1 '^version' /opt/code/deepeval/pyproject.toml) deepeval head $(git -C /opt/code/deepeval rev-parse HEAD)"
  sw_vers | tr '\n' ' '; echo; /Users/josef/bin/macmon --version 2>&1; } > $OUT/versions.txt 2>&1
/Users/josef/bin/macmon pipe -i 100 > $OUT/macmon.jsonl 2> $OUT/macmon.err &
MPID=$!
SPID=""; QPID=""
cleanup() { [[ -n "$QPID" ]] && kill $QPID 2>/dev/null; [[ -n "$SPID" ]] && kill $SPID 2>/dev/null; kill $MPID 2>/dev/null; }
trap cleanup EXIT INT TERM

gpu_check() {
  local f=$P/$1-gpu-idle.txt; : > $f; local tries=0 med
  echo "zoom before: $(pgrep -fl zoom.us | tr '\n' ';')" >> $f
  while true; do
    echo "== sample set $(date +%s)" >> $f
    local vals=()
    for i in 1 2 3; do v=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1); echo "$v" >> $f; vals+=(${v##*=}); sleep 1; done
    med=${${(on)vals}[2]}; echo "median=$med" >> $f
    if (( med <= 10 )) || (( tries >= 12 )); then break; fi
    tries=$((tries+1)); sleep 10
  done
  echo "FINAL median=$med tries=$tries" >> $f; tail -1 $f
}
wait_port_closed() { for i in {1..30}; do lsof -iTCP:$1 -sTCP:LISTEN -nP >/dev/null 2>&1 || return 0; sleep 1; done; return 1; }

run_pass() { # label mode(off|ngram)
  local label=$1 mode=$2
  echo "##### $label spec=$mode $(date -Iseconds) epoch $(date +%s)"
  echo "$label server_start $(date +%s.%N)" >> $OUT/events.log
  $BIN serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port $PORT --served-model-name intarsia-metal --max-seq 8192 \
     --profile-every 0 --no-supervisor --no-ui --spec $mode > $P/$label-server.out 2> $P/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..180}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; echo "$label SERVER_FAILED" >> $OUT/events.log; SPID=""; return 1; }
  echo "server up after ${i}s; $(grep -h 'speculative\|prefill chunk' $P/$label-server.err | head -2 | cut -c1-120 | tr '\n' ';')"
  gpu_check $label
  ( while true; do echo "$(date +%s.%N) $(curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec' | tr '\n' ' ')"; sleep 1; done ) > $P/$label-metrics.ts 2>/dev/null &
  QPID=$!
  echo "$label localeval_start $(date +%s.%N)" >> $OUT/events.log
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia-metal --sizes 128,1024,4096 --gen 256 --reps 3 --thinking off --label $label 2>&1 ) | python3 $OUT/ts.py > $P/$label-localeval.ts
  echo "$label localeval_end $(date +%s.%N)" >> $OUT/events.log
  kill $QPID 2>/dev/null; QPID=""
  curl -s http://127.0.0.1:$PORT/metrics > $P/$label-metrics-final.txt
  grep -E "rep [0-9]/3|marginal|results/" $P/$label-localeval.ts | tail -12 | cut -c1-200
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { kill -9 $SPID; sleep 1; }
  wait $SPID 2>/dev/null
  { wait_port_closed $PORT && echo "port $PORT closed" || echo "PORT $PORT STILL OPEN"; } | tee -a $P/$label-stop.txt
  SPID=""; sleep 5
}
{ echo "== start $(date +%s.%N)"; pmset -g therm; } >> $P/therm.txt
run_pass p1-off off
run_pass p2-on ngram
run_pass p3-on ngram
run_pass p4-off off
{ echo "== end $(date +%s.%N)"; pmset -g therm; } >> $P/therm.txt
echo ALLDONE
