#!/usr/bin/env bash
# storeprefault window (copied from scripts/storettft/window3.sh, arms pf / pf4k / onpf / pfdbg / smdbg added): one or more VISITS to spark2 (vLLM `retro` stays up and untouched). A visit = start one intarsia server in an arm's
# configuration, run the cold-prompt TTFT cell (localeval speed, same flags as ab35d: --sizes 4096,16384,32768 --gen 256 --reps 3,
# thinking off, nonce first in every prompt, one request at a time), stop the server. Visits run in the order given (ABBA at the call site).
#   FLEETQ_OWNER=claude-overnight-storeprefault fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect N --label "storeprefault wX ..." -- \
#     bash scripts/storeprefault/window.sh <wname> <visit>...      visit = <arm> (e.g. off on) ; a visit may repeat an arm
# arms (name: extra `serve` args | extra env):
#   off      --max-batch 2                                   store off, recipe macro prefill on, packed on (the ab35d control N)
#   offm0    --max-batch 2          | INTARSIA_PREFILL_MACRO=0   store off, macro prefill off (packed is moot for a single request)
#   on       --max-batch 2 --prefix-store on                 the shipping config (ab35d B)
#   nocap    --max-batch 2 --prefix-store min=100000000      store lookup and settle machinery, no capture at all (min_tokens above any prompt)
#   gridoff  --max-batch 2 --prefix-store grid=100000000     captures only at message-header cuts, not every 2048 tokens
#   dbg      like on, INTARSIA_PREFIX_STORE_DEBUG=1 (per-insert and per-settle timing lines; diagnostic, not for ratios)
#   dbgsync  like dbg with INTARSIA_PREFIX_STORE_DEBUG=sync (drain / sync of each capture; serialises the engine, never for ratios)
#   smacro   on + INTARSIA_PREFIX_STORE_MACRO=1 (opt-in, this lane: macro prefill with the store)
#   smacrodefer (= smacro plus the first build's temporary deferral-cap knob, set by hand through ENVS; the knob was removed again, it had no effect), slice (on + INTARSIA_SETTLE_SLICE_MB=64)
#   fix      --max-batch 2 --prefix-store on | FIXENV         the opt-in fix under test (FIXENV="INTARSIA_X=1 ...")
# Env: STOREOPT (the --prefix-store value of the store arms, default on; e.g. on,host_mb=4096 reaches the full-host-tier steady state sooner), FILL (32k requests before
# the measured run), MODE=speed|gap|replay, GAP (s), SEEDS (replay), REPS (default 3), SIZES (default 4096,16384,32768), TAG (raw-file label suffix).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; shift; VISITS="$*"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-storeprefault/raw; mkdir -p "$OUT/recipes"
SC=$REPO/scripts/storeprefault
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; MODEL=guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-storeprefault/bin/intarsia'; W='~/intarsiaml-storeprefault/out'; MS=131072; MB=40
UB=http://$HN:$PORT/v1
REPS=${REPS:-3}; SIZES=${SIZES:-4096,16384,32768}
SPID=""; WPID=""; CPID=""; T0=$(date +%s); WLOG=$OUT/window-$WN.log; MARKS=$OUT/marks-$WN.tsv
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "%d %d", s, g}'; }
capus() { rsh "nvidia-smi -q -d PERFORMANCE | awk '/SW Power Capping/ {print \$(NF-1); exit}'"; }
mark() { printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$(capus)" "$(vcount)" >> "$MARKS"; }   # epoch label sw_cap_us vllm_success vllm_gen_tokens
stop_server() {
  [[ -n "$CPID" ]] && { rsh "kill $CPID 2>/dev/null"; CPID=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "kill -TERM $SPID 2>/dev/null; for i in \$(seq 1 25); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3"; SPID=""; }
}
trap 'stop_server; log "window end"' EXIT; trap 'exit 130' INT TERM
arm_conf() { # sets ARGS and ENVS for an arm name
  ENVS=""; local base="--max-batch 2"
  case $1 in
    off) ARGS="$base" ;;
    offm0) ARGS="$base"; ENVS="INTARSIA_PREFILL_MACRO=0" ;;
    on) ARGS="$base --prefix-store ${STOREOPT:-on}" ;;
    nocap) ARGS="$base --prefix-store min=100000000" ;;
    gridoff) ARGS="$base --prefix-store grid=100000000" ;;
    dbg) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_DEBUG=1" ;;
    dbgsync) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_DEBUG=sync" ;;
    smacro) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1" ;;
    smacrodefer) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1 ${DEFERENV:-}" ;;  # DEFERENV: the temporary deferral-cap knob of the first build (removed: no effect, w7)
    slice) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_SETTLE_SLICE_MB=64" ;;
    pf) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1 INTARSIA_PREFIX_STORE_PREFAULT=1" ;;   # this lane: smacro + aligned huge-page host buffers
    pf4k) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1 INTARSIA_PREFIX_STORE_PREFAULT=4k" ;; # smacro + page-aligned 4 KiB-page host buffers
    onpf) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_PREFAULT=1" ;;   # the shipping store + prefault, no macro
    pfdbg) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1 INTARSIA_PREFIX_STORE_PREFAULT=1 INTARSIA_PREFIX_STORE_DEBUG=1" ;;
    smdbg) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="INTARSIA_PREFIX_STORE_MACRO=1 INTARSIA_PREFIX_STORE_DEBUG=1" ;;
    fix) ARGS="$base --prefix-store ${STOREOPT:-on}"; ENVS="${FIXENV:?FIXENV}" ;;
    fixoff) ARGS="$base"; ENVS="${FIXENV:?FIXENV}" ;;
    *) echo "unknown arm $1"; exit 2 ;;
  esac
}
guard_end() { # clock guard (lane rules 2026-11-19): abort the window on SM < 2000 MHz at >90% util (3+ samples) or SW power cap growth > 1 s; vLLM traffic is flagged
  local cell=$1 a b g
  mark "end:$cell"
  a=$(awk -F'\t' -v c="start:$cell" '$2==c {print $1, $3, $4, $5}' "$MARKS" | tail -1); b=$(awk -F'\t' -v c="end:$cell" '$2==c {print $1, $3, $4, $5}' "$MARKS" | tail -1)
  set -- $a $b; local s0=$1 cap0=$2 vs0=$3 vg0=$4 e1=$5 cap1=$6 vs1=$7 vg1=$8
  g=$(rsh "awk -v s=$s0 -v e=$e1 '\$1>=s && \$1<=e {n++; if (\$4>90) {u++; if (\$2<2000) lo++; if (min==\"\"||\$2<min) min=\$2; sum+=\$2; if (\$3>pm) pm=\$3}} END {printf \"%d %d %d %d %d %d\", n, u, lo, min, (u?sum/u:0), pm}' $W/clk-$WN.tsv")
  set -- $g; local n=$1 u=$2 lo=$3 min=$4 mean=$5 pmax=$6 dcap=$((cap1-cap0)) dvs=$((vs1-vs0)) dvg=$((vg1-vg0))
  log "cell $cell end: clock guard: ${n} samples, ${u} at util>90, ${lo} of those SM<2000 MHz, min SM ${min}, mean SM ${mean}, max power ${pmax} W; SW power cap delta ${dcap} us; vLLM delta success ${dvs} gen_tokens ${dvg}"
  local bad=""
  (( dcap > 1000000 )) && bad="$bad swcap+${dcap}us"
  (( lo >= 3 )) && bad="$bad sm<2000x${lo}"
  (( dvs > 0 )) && echo "$cell: vllm-traffic-during-run success+${dvs}" >> "$OUT/contamination-$WN.txt"
  [[ -n $bad ]] && { log "CLOCKGUARD FAIL in cell $cell:$bad"; echo "$cell:$bad" >> "$OUT/guard-fail-$WN.txt"; exit 4; }
  return 0
}
vi=0
for arm in $VISITS; do
  vi=$((vi+1)); V="$WN-v$vi-$arm"; arm_conf "$arm"
  SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 $ARGS"
  printf '%s\n' "intarsia serve $SERVE_ARGS   (env: ${ENVS:-none})" > "$OUT/recipes/$V.txt"
  rsh "mkdir -p $W && rm -f $W/mem-$V.txt $W/clk-$WN.tsv"
  scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/clk.sh" "$HOST:intarsiaml-storeprefault/out/" || { log "script copy failed"; exit 1; }
  if rsh "ss -ltn | grep -q ':$PORT '"; then log "port $PORT busy on $HOST, aborting"; exit 1; fi
  log "visit $V: vLLM /v1/models $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$HN:$VPORT/v1/models); counters $(vcount); earlyoom $(rsh 'systemctl is-active earlyoom'); free: $(rsh 'free -g | sed -n 2p')"
  log "binary: $(rsh "$BIN --version | head -1"); serve $SERVE_ARGS; env ${ENVS:-none}"
  CPID=$(rsh "nohup bash $W/clk.sh $W/clk-$WN.tsv > /dev/null 2>&1 < /dev/null & echo \$!")
  SPID=$(rsh "env $ENVS nohup $BIN serve $SERVE_ARGS > $W/serve-$V.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$V.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve did not come up"; rsh "tail -20 $W/serve-$V.log" | tee -a "$WLOG"; exit 1; }
  log "serve up; MemAvailable MB $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |macro|packed|prefix store: ' $W/serve-$V.log | cut -c1-110 | tr '\n' '|'")"
  mark "start:$V"; curl -s --max-time 8 "http://$HN:$PORT/metrics" | gzip > "$OUT/snap-$V.before.gz"
  log "thp before: $(rsh "grep -E '^(thp_fault_alloc|thp_fault_fallback|compact_stall) ' /proc/vmstat | tr '\n' ' '")"
  if [[ -n ${FILL:-} ]]; then
    # steady-state store: FILL requests of 32k tokens (several GB of captures each) before the measured run, so the 8 GiB host tier is full
    # and every settle evicts and recycles buffers, as in production. The same fill runs for every arm (store off ones too).
    log "fill: $FILL x 32768-token requests (gen 16)"
    ( cd /opt/code/deepeval && uv run localeval speed --url "$UB" --model $MODEL --thinking off --structured prompt --wait-idle 600 --recipe "$OUT/recipes/$V.txt" \
        --sizes 32768 --gen 16 --reps "$FILL" --label "storeprefault-$V-fill" ) > "$OUT/$V-fill.out" 2>&1
    log "fill done: $(grep -ac 'tok rep' "$OUT/$V-fill.out") reps; store host bytes $(curl -s --max-time 8 http://$HN:$PORT/metrics | awk '/^intarsia_prefix_store_tier_bytes\{tier="host"\}/ {printf "%.2f GiB", $2/1073741824}')"
    mark "start:$V"; curl -s --max-time 8 "http://$HN:$PORT/metrics" | gzip > "$OUT/snap-$V.before.gz"
  fi
  if [[ ${MODE:-speed} == replay ]]; then
    # 2-stream interleaved multi-turn replay (ab35d's cell, same plan and traces; vLLM's recorded answers as history): TTFT sum, decode, reply texts
    SI=$REPO/scripts/interleave
    for sd in ${SEEDS:-0 1}; do
      mark "start:$V-s$sd"
      ( cd /opt/code/deepeval && uv run python "$SI/replay2.py" --url "$UB" --model $MODEL --arm "$V" --order interleaved --seed "$sd" --streams 2 \
          --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$sd.json" --out "$OUT" ) > "$OUT/$V-replay-seed$sd.out" 2>&1
      tail -2 "$OUT/$V-replay-seed$sd.out" | cut -c1-250
      CELLNAME_SAVE=$V; guard_end "$V-s$sd"
    done
  elif [[ ${MODE:-speed} == ids ]]; then
    # gate (b), cold prompts: greedy replies for fixed prompts, each sent cold then again (restore), compared between arms with compare_ids.py
    ( cd /opt/code/deepeval && uv run python "$SC/ids_check.py" --url "$UB" --model $MODEL --arm "$V" --out "$OUT" ) > "$OUT/$V-ids.out" 2>&1
    tail -3 "$OUT/$V-ids.out" | cut -c1-200
  elif [[ ${MODE:-speed} == gap ]]; then
    # idle-gap variant: same prompts and TTFT definition, GAP seconds of idle after every request (the store's settle then runs in idle time)
    ( cd /opt/code/deepeval && uv run python "$SC/ttft_gap.py" --url "$UB" --model $MODEL --arm "$V" --gap "${GAP:-5}" --sizes "$SIZES" --reps "$REPS" --gen 256 --out "$OUT" ) > "$OUT/$V-gap${GAP:-5}.out" 2>&1
    tail -3 "$OUT/$V-gap${GAP:-5}.out" | cut -c1-200
  else
  ( cd /opt/code/deepeval && uv run localeval speed --url "$UB" --model $MODEL --thinking off --structured prompt --wait-idle 600 --recipe "$OUT/recipes/$V.txt" \
      --sizes "$SIZES" --gen 256 --reps "$REPS" --label "storeprefault-$V${TAG:-}" ) > "$OUT/$V-speed.out" 2>&1
  tail -3 "$OUT/$V-speed.out" | cut -c1-200
  fi
  if [[ ${MODE:-speed} == speed ]]; then
    rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$OUT/$V-speed.out" | head -1 | sed 's#results/##; s#/.*##'); echo "$V run_id=$rid" | tee -a "$OUT/index.txt"
    [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }
  fi
  curl -s --max-time 8 "http://$HN:$PORT/metrics" | gzip > "$OUT/snap-$V.after.gz"
  log "thp after: $(rsh "grep -E '^(thp_fault_alloc|thp_fault_fallback|compact_stall) ' /proc/vmstat | tr '\n' ' '"); server $(rsh "grep -E '^(Rss|AnonHugePages):' /proc/$SPID/smaps_rollup | tr -s ' ' | tr '\n' ' '")"
  [[ ${MODE:-speed} == replay ]] || guard_end "$V"
  rsh "cat $W/serve-$V.log" > "$OUT/serve-$V.log"; rsh "cat $W/mem-$V.txt" > "$OUT/mem-$V.txt" 2>/dev/null
  rsh "cat $W/clk-$WN.tsv" > "$OUT/clk-$V.tsv"
  log "visit $V done; min MemAvailable MB (time MB) $(sort -k2 -n "$OUT/mem-$V.txt" | head -1); KILLED lines $(grep -c KILLED "$OUT/mem-$V.txt"); warn/error lines in serve log $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$V.log")"
  stop_server
done
log "window $WN done: visits $VISITS; fleetq job $FLEETQ_JOB"
