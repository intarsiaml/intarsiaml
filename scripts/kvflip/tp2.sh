#!/usr/bin/env bash
# kvflip tp2 gate: the 35B NVFP4 at tp = 2 ACROSS the two Sparks (rank 0 = spark1 `serve --tp-remote`, rank 1 = spark2 `tp-worker`, NCCL over CX-7),
# `--spec mtp --spec-k 2`, vLLM up on both and untouched. INTARSIA_CUDA_ATTN_KV_ONCE goes in the environment of BOTH ranks. One fresh server per window.
#   FLEETQ_OWNER=claude-kvflip fleetq run --host spark1,spark2 --quiet gpu,cpu --mem spark1=45,spark2=45 --expect 120 --label "kvflip tp2" -- bash scripts/kvflip/tp2.sh a1 - ids cells ...
# usage: tp2.sh <wname> "<knob env assignment or ->" <cell>...   cell = ids | <temperature>:<sizes>[:<reps>]
# Clock guard: samples on both boxes every 2 s; SM < 2300 MHz at util > 50 % for 3+ samples (tp2 GPUs idle half of each step, so > 90 would never fire) or growth of SW power capping > 1 s fails the window.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; ENVS=$2; shift 2; CELLS="$*"; [[ $ENVS == "-" ]] && ENVS=""
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-kvflip/raw; mkdir -p "$OUT"
S=$REPO/scripts/ab35b; SC=$REPO/scripts/kvflip
BIN="${KVFLIP_BIN:-\$HOME/intarsiaml-kvflip/bin/intarsia}"; W='~/intarsiaml-kvflip/out'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555; PORT=8193; MODEL=guidance; UB=http://spark-4e77:$PORT/v1
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN"
FL="-m $M35 --backend cuda --max-seq 70000 --mem-budget 32 --profile-every 0 --tp 2 --spec mtp --spec-k 2"
T0=$(date +%s); WLOG=$OUT/window-$WN.log; MARKS=$OUT/marks-$WN.tsv; : > "$MARKS"
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
capus() { r spark1 "nvidia-smi -q -d PERFORMANCE | awk '/SW Power Capping/ {print \$(NF-1); exit}'"; }
mark() { printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$(capus)" "$(r spark2 "nvidia-smi -q -d PERFORMANCE | awk '/SW Power Capping/ {print \$(NF-1); exit}'")" >> "$MARKS"; }
GPIDS=""
kill_all() { for h in spark1 spark2; do r $h "touch ~/intarsiaml-kvflip/stop-worker; pkill -TERM -f '[i]ntarsiaml-kvflip/bin/'; for i in \$(seq 1 25); do pgrep -f '[i]ntarsiaml-kvflip/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsiaml-kvflip/bin/'; pkill -f '[k]vflip/out/(clk|memwatch)' ; true"; done; }
cleanup() { trap - EXIT INT TERM; kill_all; for h in spark1 spark2; do r $h "cat $W/clk-$WN.tsv" > "$OUT/clk-$WN-$h.tsv" 2>/dev/null; r $h "cat $W/mem-$WN.txt" > "$OUT/mem-$WN-$h.txt" 2>/dev/null; done
  r spark1 "cat $W/serve-$WN.log" > "$OUT/serve-$WN.log" 2>/dev/null; r spark2 "cat $W/worker-$WN.log" > "$OUT/worker-$WN.log" 2>/dev/null; log "window end"; }
trap cleanup EXIT; trap 'exit 130' INT TERM
guard_start() { CELLNAME=$1; mark "start:$1"; log "cell $1 start; $(r spark1 'nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader') | $(r spark2 'nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader')"; }
guard_end() {
  mark "end:$1"; local name=$CELLNAME bad="" h
  local s0 e1; s0=$(awk -F'\t' -v c="start:$1" '$2==c {print $1}' "$MARKS" | tail -1); e1=$(awk -F'\t' -v c="end:$1" '$2==c {print $1}' "$MARKS" | tail -1)
  local c0s c1s c0r c1r; c0s=$(awk -F'\t' -v c="start:$1" '$2==c {print $3}' "$MARKS" | tail -1); c1s=$(awk -F'\t' -v c="end:$1" '$2==c {print $3}' "$MARKS" | tail -1)
  c0r=$(awk -F'\t' -v c="start:$1" '$2==c {print $4}' "$MARKS" | tail -1); c1r=$(awk -F'\t' -v c="end:$1" '$2==c {print $4}' "$MARKS" | tail -1)
  for h in spark1 spark2; do
    local g; g=$(r $h "awk -v s=$s0 -v e=$e1 '\$1>=s && \$1<=e {n++; if (\$4>50) {u++; if (\$2<2300) lo++; if (min==\"\"||\$2<min) min=\$2; sum+=\$2; if (\$3>pm) pm=\$3}} END {printf \"%d %d %d %d %d %d\", n, u, lo, min, (u?sum/u:0), pm}' $W/clk-$WN.tsv")
    set -- $g; log "cell $name end $h: ${1} samples, ${2} at util>50, ${3} of those SM<2300 MHz, min SM ${4}, mean SM ${5}, max power ${6} W"; (( $3 >= 3 )) && bad="$bad $h:sm<2300x$3"
  done
  local d1=$((c1s-c0s)) d2=$((c1r-c0r)); log "cell $name SW power cap delta spark1 ${d1} us spark2 ${d2} us"; (( d1 > 1000000 || d2 > 1000000 )) && bad="$bad swcap"
  [[ -n $bad ]] && { log "CLOCKGUARD FAIL in cell $name:$bad"; echo "$name:$bad" >> "$OUT/guard-fail-$WN.txt"; exit 4; }
  return 0
}
log "window $WN env='$ENVS' cells: $CELLS; fleetq job $FLEETQ_JOB"
for h in spark1 spark2; do r $h "mkdir -p $W && rm -f $W/mem-$WN.txt $W/clk-$WN.tsv $W/serve-$WN.log $W/worker-$WN.log ~/intarsiaml-kvflip/stop-worker"; scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/clk.sh" "$h:intarsiaml-kvflip/out/"; done
r spark1 "ss -ltn | grep -q ':$PORT '" && { log "port busy, abort"; exit 1; }
log "free: spark1 $(r spark1 'free -g | sed -n 2p') | spark2 $(r spark2 'free -g | sed -n 2p'); earlyoom $(r spark1 'systemctl is-active earlyoom') $(r spark2 'systemctl is-active earlyoom'); vLLM $(curl -s -o /dev/null -w '%{http_code}' http://100.120.88.104:8000/v1/models) $(curl -s -o /dev/null -w '%{http_code}' http://100.96.95.89:8000/v1/models); binary: $(r spark1 "$BIN --version | head -1")"
for h in spark1 spark2; do r $h "nohup bash $W/clk.sh $W/clk-$WN.tsv > /dev/null 2>&1 < /dev/null &"; done
# rank 1 first (it listens), then rank 0; both carry the knob
r spark2 "nohup env $NCCLENV $ENVS $BIN tp-worker --listen $WIP:$WPORT --serving $FL > $W/worker-$WN.log 2>&1 < /dev/null &"
SPID=$(r spark1 "nohup env $NCCLENV $ENVS $BIN serve $FL --tp-remote $WIP:$WPORT --port $PORT --host 0.0.0.0 --served-model-name $MODEL > $W/serve-$WN.log 2>&1 < /dev/null & echo \$!")
r spark1 "nohup bash $W/memwatch.sh $SPID $W/mem-$WN.txt > /dev/null 2>&1 < /dev/null &"
WPID=$(r spark2 "pgrep -f '[i]ntarsiaml-kvflip/bin/intarsia tp-worker' | head -1"); [[ -n $WPID ]] && r spark2 "nohup bash $W/memwatch.sh $WPID $W/mem-$WN.txt > /dev/null 2>&1 < /dev/null &"
up=0; for i in $(seq 1 100); do curl -sf "http://spark-4e77:$PORT/health" >/dev/null && { up=1; break; }; r spark1 "kill -0 $SPID" 2>/dev/null || break; sleep 3; done
[[ $up == 1 ]] || { log "serve did not come up"; r spark1 "tail -20 $W/serve-$WN.log" | tee -a "$WLOG"; r spark2 "tail -20 $W/worker-$WN.log" | tee -a "$WLOG"; exit 1; }
log "serve up; MemAvailable MB spark1 $(r spark1 "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo") spark2 $(r spark2 "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(r spark1 "grep -aEi 'spec|mtp|tp-remote|kv.once' $W/serve-$WN.log | head -6 | cut -c1-200" | tr '\n' ';')"
curl -sf "http://spark-4e77:$PORT/metrics" > "$OUT/serve-$WN.metrics.before" 2>/dev/null
for c in $CELLS; do
  if [[ $c == ids ]]; then
    guard_start "$WN-ids"
    ( cd /opt/code/deepeval && uv run python "$SC/ids.py" --url "$UB" --model $MODEL --arm "$WN" --out "$OUT" ) > "$OUT/$WN-ids.out" 2>&1
    tail -8 "$OUT/$WN-ids.out" | tee -a "$WLOG"; guard_end "$WN-ids"; continue
  fi
  IFS=: read -r T SZ RP <<< "$c"; RP=${RP:-3}; name="$WN-T$T"
  guard_start "$name"
  ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "$UB" --model $MODEL --arm "$name" --out "$OUT" --metrics "http://spark-4e77:$PORT/metrics" --sizes "$SZ" --reps "$RP" --temperature "$T" ) > "$OUT/$name-sampled.out" 2>&1
  tail -8 "$OUT/$name-sampled.out" | tee -a "$WLOG"; guard_end "$name"
done
sleep 3; curl -sf "http://spark-4e77:$PORT/metrics" > "$OUT/serve-$WN.metrics" 2>/dev/null
log "min MemAvailable MB: spark1 $(r spark1 "sort -k2 -n $W/mem-$WN.txt | head -1") spark2 $(r spark2 "sort -k2 -n $W/mem-$WN.txt | head -1"); KILLED lines: $(r spark1 "grep -c KILLED $W/mem-$WN.txt") $(r spark2 "grep -c KILLED $W/mem-$WN.txt")"
