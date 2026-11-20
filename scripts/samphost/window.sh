#!/usr/bin/env bash
# samphost (copied from scripts/samp64) window: one intarsiaml server on spark2 (vLLM stays up, never touched), single-stream decode cells at several temperatures.
#   FLEETQ_OWNER=claude-samphost fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect N --label "samp64 wX ..." -- \
#       bash scripts/samphost/window.sh <wname> "<extra serve args>" "<env assignments or ->" <cell>...
# cell = <temperature>:<sizes comma list>[:<reps>]   e.g. 0.7:4096,32768,65536   0:32768,65536:5
# Clock guard (lane rules 2026-11-19): SM clock < 2000 MHz at > 90 % util for 3+ samples or SW power capping growth > 1 s aborts the window.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; EXTRA=$2; ENVS=$3; shift 3; CELLS="$*"
[[ $ENVS == "-" ]] && ENVS=""
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-samphost/raw; mkdir -p "$OUT"
S=$REPO/scripts/ab35b; SC=$REPO/scripts/samphost
HOST=spark2; HN=spark-fa00; PORT=8197; MODEL=guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN="${SAMPHOST_BIN:-~/intarsiaml-samphost/bin/intarsia}"; W='~/intarsiaml-samphost/out'; MS=131072; MB=40
UB=http://$HN:$PORT/v1
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 --max-batch 2 --prefix-store on $EXTRA"
SPID=""; WPID=""; CPID=""; T0=$(date +%s); WLOG=$OUT/window-$WN.log; MARKS=$OUT/marks-$WN.tsv
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
capus() { rsh "nvidia-smi -q -d PERFORMANCE | awk '/SW Power Capping/ {print \$(NF-1); exit}'"; }
mark() { printf '%s\t%s\t%s\n' "$(date +%s)" "$1" "$(capus)" >> "$MARKS"; }
stop_all() {
  [[ -n "$CPID" ]] && { rsh "kill $CPID 2>/dev/null"; CPID=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; pkill -9 -P $SPID; kill -9 $SPID 2>/dev/null; sleep 3"; SPID=""; }
}
trap 'stop_all; log "window end"' EXIT; trap 'exit 130' INT TERM
guard_start() { CELLNAME=$1; mark "start:$1"; log "cell $1 start; $(rsh 'nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader')"; }
guard_end() {
  mark "end:$1"
  local a b; a=$(awk -F'\t' -v c="start:$1" '$2==c {print $1, $3}' "$MARKS" | tail -1); b=$(awk -F'\t' -v c="end:$1" '$2==c {print $1, $3}' "$MARKS" | tail -1)
  set -- $a $b; local s0=$1 cap0=$2 e1=$3 cap1=$4 name=$CELLNAME
  local g; g=$(rsh "awk -v s=$s0 -v e=$e1 '\$1>=s && \$1<=e {n++; if (\$4>90) {u++; if (\$2<2000) lo++; if (min==\"\"||\$2<min) min=\$2; sum+=\$2; if (\$3>pm) pm=\$3}} END {printf \"%d %d %d %d %d %d\", n, u, lo, min, (u?sum/u:0), pm}' $W/clk-$WN.tsv")
  set -- $g; local n=$1 u=$2 lo=$3 min=$4 mean=$5 pmax=$6 dcap=$((cap1-cap0))
  log "cell $name end: clock guard: ${n} samples, ${u} at util>90, ${lo} of those SM<2000 MHz, min SM ${min}, mean SM ${mean}, max power ${pmax} W; SW power cap delta ${dcap} us"
  local bad=""; (( dcap > 1000000 )) && bad="$bad swcap+${dcap}us"; (( lo >= 3 )) && bad="$bad sm<2000x${lo}"
  [[ -n $bad ]] && { log "CLOCKGUARD FAIL in cell $name:$bad"; echo "$name:$bad" >> "$OUT/guard-fail-$WN.txt"; exit 4; }
  return 0
}
log "window $WN extra='$EXTRA' env='$ENVS' cells: $CELLS; fleetq job $FLEETQ_JOB"
rsh "mkdir -p $W && rm -f $W/mem-$WN.txt $W/clk-$WN.tsv"
scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/clk.sh" "$HOST:intarsiaml-samphost/out/" || { log "script copy failed"; exit 1; }
rsh "ss -ltn | grep -q ':$PORT '" && { log "port $PORT busy, abort"; exit 1; }
log "free: $(rsh 'free -g | sed -n 2p'); earlyoom $(rsh 'systemctl is-active earlyoom'); binary: $(rsh "$BIN --version | head -1")"
CPID=$(rsh "nohup bash $W/clk.sh $W/clk-$WN.tsv > /dev/null 2>&1 < /dev/null & echo \$!")
log "serve: $ENVS $BIN serve $SERVE_ARGS"
SPID=$(rsh "nohup env $ENVS $BIN serve $SERVE_ARGS > $W/serve-$WN.log 2>&1 < /dev/null & echo \$!")
WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$WN.txt > /dev/null 2>&1 < /dev/null & echo \$!")
up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
[[ $up == 1 ]] || { log "serve did not come up"; rsh "tail -20 $W/serve-$WN.log" | tee -a "$WLOG"; exit 1; }
log "serve up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$WN.metrics.before" 2>/dev/null
for c in $CELLS; do
  if [[ $c == seeded || $c == seeded:* ]]; then   # byte-for-byte reply comparison between windows (scripts/samphost/seeded.py); seeded:<sizes> picks the prompt sizes
    guard_start "$WN-seeded"
    ( cd /opt/code/deepeval && uv run python "$SC/seeded.py" --url "$UB" --model $MODEL --arm "$WN" --out "$OUT" $( [[ $c == seeded:* ]] && echo "--sizes ${c#seeded:}" ) ) > "$OUT/$WN-seeded.out" 2>&1
    tail -8 "$OUT/$WN-seeded.out" | tee -a "$WLOG"; guard_end "$WN-seeded"; continue
  fi
  IFS=: read -r T SZ RP <<< "$c"; RP=${RP:-3}; name="$WN-T$T"
  guard_start "$name"
  ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "$UB" --model $MODEL --arm "$name" --out "$OUT" --metrics "http://$HN:$PORT/metrics" \
      --sizes "$SZ" --reps "$RP" --temperature "$T" ) > "$OUT/$name-sampled.out" 2>&1
  tail -8 "$OUT/$name-sampled.out" | tee -a "$WLOG"; guard_end "$name"
done
sleep 3; curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$WN.metrics" 2>/dev/null
rsh "cat $W/serve-$WN.log" > "$OUT/serve-$WN.log"; rsh "cat $W/clk-$WN.tsv" > "$OUT/clk-$WN.tsv"; rsh "cat $W/mem-$WN.txt" > "$OUT/mem-$WN.txt" 2>/dev/null
log "min MemAvailable MB: $(sort -k2 -n "$OUT/mem-$WN.txt" | head -1); KILLED lines: $(grep -c KILLED "$OUT/mem-$WN.txt"); warn/error lines in serve log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$WN.log")"
