#!/usr/bin/env bash
# ab35d CONTROL window (store off): same as window.sh, but arm N = `serve --max-batch 2` WITHOUT --prefix-store (ab35 B-default shape), to attribute the prefill gap of window w2.
# ab35d window: intarsiaml (arm B, shipping config) beside the production vLLM (arm A, never touched) on spark2.
#   FLEETQ_OWNER=claude-overnight-ab35d fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect N --label "ab35d wX ..." -- bash scripts/ab35d/window.sh <wname> <cell>...
# cells: tools replay speed sampled gsm8k. One B server for the whole window, vLLM stays up; arms alternate A B B A inside each cell,
# one request stream at a time, so nothing of ours generates on both engines at once. Clock guard marks go to $OUT/marks-<wname>.tsv.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
WN=$1; shift; CELLS="$*"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-19-ab35d-cutover-scoreboard/raw; mkdir -p "$OUT/recipes"
S=$REPO/scripts/ab35b; SC=$REPO/scripts/ab35d; SI=$REPO/scripts/interleave
HOST=spark2; HN=spark-fa00; PORT=8191; VPORT=8000; MODEL=guidance
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-ab35d/bin/intarsia'; W='~/intarsiaml-ab35d/out'; MS=131072; MB=40
UA=http://$HN:$VPORT/v1; UB=http://$HN:$PORT/v1
SERVE_ARGS="--backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name $MODEL --max-seq $MS --mem-budget $MB --profile-every 0 ${CTL_SERVE:---max-batch 2}"
SPID=""; WPID=""; CPID=""; T0=$(date +%s); WLOG=$OUT/window-$WN.log; MARKS=$OUT/marks-$WN.tsv
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$WLOG"; }
vcount() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:request_success_total/ {s+=$2} /^vllm:generation_tokens_total/ {g=$2} END {printf "%d %d", s, g}'; }
capus() { rsh "nvidia-smi -q -d PERFORMANCE | awk '/SW Power Capping/ {print \$(NF-1); exit}'"; }
mark() { printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$1" "$(capus)" "$(vcount)" >> "$MARKS"; }   # epoch label sw_cap_us vllm_success vllm_gen_tokens
stop_all() {
  [[ -n "$CPID" ]] && { rsh "kill $CPID 2>/dev/null"; CPID=""; }
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 25); do pgrep -f 'intarsiaml-ab35d/bin/intarsia serve' >/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-ab35d/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_all; log "window end"' EXIT; trap 'exit 130' INT TERM
recipe() { printf '%s\n' "$2" > "$OUT/recipes/$1.txt"; echo "$OUT/recipes/$1.txt"; }
RA=$(recipe A "production vLLM (missioncntrl-guidance, image missioncntrl/vllm-openai:65b7662d): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.32 --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 (env VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel; vllm 0.26.1rc1.dev602; no speculative config; spark2 retro role)")
RB=$(recipe "N-$WN" "intarsia serve $SERVE_ARGS")
keep() { local rid; rid=$(grep -aoE 'results/[0-9]{8}-[^ ]*' "$1" | head -1 | sed 's#results/##; s#/.*##'); echo "$2 run_id=$rid" | tee -a "$OUT/index.txt"
  [[ -n $rid && -d /opt/code/deepeval/results/$rid ]] && { mkdir -p "$OUT/$rid"; cp /opt/code/deepeval/results/$rid/summary.md /opt/code/deepeval/results/$rid/result.json "$OUT/$rid/" 2>/dev/null; }; }
le() { # le <speed|quality> <label> <url> <recipe> <out> [args]
  local kind=$1 label=$2 url=$3 rec=$4 out=$5; shift 5
  ( cd /opt/code/deepeval && uv run localeval "$kind" --url "$url" --model $MODEL --thinking off --structured prompt --wait-idle 600 --recipe "$rec" "$@" --label "ab35d-$label" ) > "$out" 2>&1
  tail -4 "$out"; keep "$out" "$label"; }
snap() { # snap <cell> <before|after>: gzip both engines' /metrics next to the marks (diffed by clockguard.py / tables)
  mkdir -p "$OUT/snap"; curl -s --max-time 8 "http://$HN:$VPORT/metrics" | gzip > "$OUT/snap/$1.vllm.$2.gz"; curl -s --max-time 8 "http://$HN:$PORT/metrics" | gzip > "$OUT/snap/$1.b.$2.gz"; }
guard_start() { CELLNAME=$1; mark "start:$1"; snap "$1" before; log "cell $1 start; vLLM counters (success gen_tokens): $(vcount); $(rsh 'nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader')"; }
guard_end() { # enforcing clock guard + shared-box contamination check (lane rules 2026-11-19); a failure aborts the window (trap stops B)
  mark "end:$1"; snap "$1" after
  local a b; a=$(awk -F'\t' -v c="start:$1" '$2==c {print $1, $3, $4, $5}' "$MARKS" | tail -1); b=$(awk -F'\t' -v c="end:$1" '$2==c {print $1, $3, $4, $5}' "$MARKS" | tail -1)
  set -- $a $b; local s0=$1 cap0=$2 vs0=$3 vg0=$4 e1=$5 cap1=$6 vs1=$7 vg1=$8 name=$CELLNAME
  local g; g=$(rsh "awk -v s=$s0 -v e=$e1 '\$1>=s && \$1<=e {n++; if (\$4>90) {u++; if (\$2<2000) lo++; if (min==\"\"||\$2<min) min=\$2; sum+=\$2; if (\$3>pm) pm=\$3}} END {printf \"%d %d %d %d %d %d\", n, u, lo, min, (u?sum/u:0), pm}' $W/clk-$WN.tsv")
  set -- $g; local n=$1 u=$2 lo=$3 min=$4 mean=$5 pmax=$6 dcap=$((cap1-cap0)) dvs=$((vs1-vs0)) dvg=$((vg1-vg0))
  log "cell $name end: clock guard: ${n} samples, ${u} at util>90, ${lo} of those SM<2000 MHz, min SM ${min}, mean SM ${mean}, max power ${pmax} W; SW power cap delta ${dcap} us; vLLM delta success ${dvs} gen_tokens ${dvg}"
  local bad=""
  (( dcap > 1000000 )) && bad="$bad swcap+${dcap}us"
  (( lo >= 3 )) && bad="$bad sm<2000x${lo}"
  [[ $name == *-N* ]] && (( dvs > 0 )) && bad="$bad vllm-traffic-during-B(success+${dvs})"
  [[ -n $bad ]] && { log "CLOCKGUARD/CONTAMINATION FAIL in cell $name:$bad"; echo "$name:$bad" >> "$OUT/guard-fail-$WN.txt"; exit 4; }
  return 0
}
vllm_idle() { curl -s --max-time 5 "http://$HN:$VPORT/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/ {n+=$2} END {exit (n>0)}'; }
start_b() {
  rsh "mkdir -p $W && rm -f $W/mem-$WN.txt $W/clk-$WN.tsv"
  scp -q -o BatchMode=yes "$SC/memwatch.sh" "$SC/clk.sh" "$HOST:intarsiaml-ab35d/out/" || { log "script copy failed"; exit 1; }
  if rsh "ss -ltn | grep -q ':$PORT '"; then log "port $PORT busy on $HOST, aborting"; exit 1; fi
  log "vLLM /v1/models: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 $UA/models); counters $(vcount); earlyoom $(rsh 'systemctl is-active earlyoom'); free: $(rsh 'free -g | sed -n 2p')"
  log "binary: $(rsh "$BIN --version | head -1")"
  CPID=$(rsh "nohup bash $W/clk.sh $W/clk-$WN.tsv > /dev/null 2>&1 < /dev/null & echo \$!")
  log "serve B: $BIN serve $SERVE_ARGS"
  SPID=$(rsh "nohup $BIN serve $SERVE_ARGS > $W/serve-$WN.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $W/memwatch.sh $SPID $W/mem-$WN.txt > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 200); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve B did not come up"; rsh "tail -20 $W/serve-$WN.log" | tee -a "$WLOG"; exit 1; }
  log "serve B up; MemAvailable MB: $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo"); $(rsh "grep -E '^loaded |spec mtp|batched MTP|prefix store: |MTP|context' $W/serve-$WN.log | tr '\n' '|'")"
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$WN.metrics.before" 2>/dev/null
  # alias check (production vLLM serves the names guidance AND capcom on one port; B has one --served-model-name): which names does B answer?
  log "B /v1/models: $(curl -s --max-time 5 http://$HN:$PORT/v1/models | head -c 300)"
  for nm in guidance capcom no-such-model; do
    local body="{\"model\":\"$nm\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
    local resp; resp=$(curl -s --max-time 60 -w ' HTTP%{http_code}' http://$HN:$PORT/v1/chat/completions -H 'Content-Type: application/json' --data-raw "$body" | tr '\n' ' ' | head -c 300)
    log "B chat with model=$nm -> $resp"
  done
  log "A /v1/models: $(curl -s --max-time 5 $UA/models | head -c 400)"
}
finish_b() {
  sleep 3; curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$WN.metrics" 2>/dev/null
  rsh "cat $W/serve-$WN.log" > "$OUT/serve-$WN.log"; rsh "cat $W/clk-$WN.tsv" > "$OUT/clk-$WN.tsv"; rsh "cat $W/mem-$WN.txt" > "$OUT/mem-$WN.txt" 2>/dev/null
  log "min MemAvailable MB (time MB): $(sort -k2 -n "$OUT/mem-$WN.txt" | head -1); KILLED lines: $(grep -c KILLED "$OUT/mem-$WN.txt"); warn/error lines in serve log: $(grep -ciE 'warn|error|panic|fail' "$OUT/serve-$WN.log")"
}

cell_tools() {
  guard_start tools
  for i in $(seq 1 60); do vllm_idle && break; sleep 10; done; vllm_idle || { log "vLLM busy for 10 min, skipping tools"; return; }
  for arm in A B; do u=$UA; [[ $arm == B ]] && u=$UB
    python3 "$SC/toolsmoke.py" --url "$u" --model $MODEL --arm "$arm" --thinking off --reps 5 --stream-reps 2 --out "$OUT" > "$OUT/$arm-tools.out" 2>&1; tail -4 "$OUT/$arm-tools.out"
    python3 "$SC/toolsmoke.py" --url "$u" --model $MODEL --arm "$arm-think" --thinking default --reps 3 --stream-reps 2 --out "$OUT" > "$OUT/$arm-tools-think.out" 2>&1; tail -4 "$OUT/$arm-tools-think.out"
  done; guard_end tools
}
cell_replay() { # ABBA, 2 streams, interleaved; seed 0 and seed 1 (different prompts) so no arm replays a prompt the other arm's cache or store has seen of its own
  for p in "A 0" "B 0" "B 1" "A 1"; do set -- $p; arm=$1; seed=$2; u=$UA; [[ $arm == B ]] && u=$UB
    guard_start "replay-$arm-s$seed"; [[ $arm == B ]] && curl -sf "http://$HN:$PORT/metrics" > "$OUT/B-replay-s$seed.metrics.before"
    ( cd /opt/code/deepeval && uv run python "$SI/replay2.py" --url "$u" --model $MODEL --arm "$arm" --order interleaved --seed "$seed" --streams 2 \
        --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$seed.json" --out "$OUT" ) > "$OUT/$arm-replay2-interleaved-seed$seed.out" 2>&1
    tail -12 "$OUT/$arm-replay2-interleaved-seed$seed.out"; guard_end "replay-$arm-s$seed"
    [[ $arm == B ]] && { sleep 3; curl -sf "http://$HN:$PORT/metrics" > "$OUT/B-replay-s$seed.metrics"; }
  done
}
cell_speed() {
  for p in A1 N1 N2 A2; do arm=${p:0:1}; u=$UA; rec=$RA; [[ $arm == N ]] && { u=$UB; rec=$RB; }
    guard_start "ctlspeed-$p"; le speed "ctl-$p-speed" "$u" "$rec" "$OUT/ctl-$p-speed.out" --sizes 4096,16384,32768 --gen 256 --reps 3; guard_end "ctlspeed-$p"; done
}
cell_sampled() { # control arm N (CTL_SERVE) vs A, ABBA, sampled T=0.7 at SIZES (default 32768,65536)
  for p in A1 N1 N2 A2; do arm=${p:0:1}; u=$UA; [[ $arm == N ]] && u=$UB
    guard_start "ctlsampled-$p"
    ( cd /opt/code/deepeval && uv run python "$S/sampled.py" --url "$u" --model $MODEL --arm "ctl-$p" --out "$OUT" $( [[ $arm == N ]] && echo "--metrics http://$HN:$PORT/metrics" ) --sizes "${SIZES:-32768,65536}" ) > "$OUT/ctl-$p-sampled.out" 2>&1
    tail -6 "$OUT/ctl-$p-sampled.out"; guard_end "ctlsampled-$p"; done
}
cell_replay1() { # K=1 interleaved replay (blocker 3's original shape): A N N A on seeds 0 1 1 0 (use CTL_SERVE with the store on)
  for p in "A 0" "N 0" "N 1" "A 1"; do set -- $p; arm=$1; seed=$2; u=$UA; [[ $arm == N ]] && u=$UB
    guard_start "ctlreplay1-$arm-s$seed"
    ( cd /opt/code/deepeval && uv run python "$SI/replay2.py" --url "$u" --model $MODEL --arm "ctl1-$arm" --order interleaved --seed "$seed" --streams 1 \
        --plan "$OUT/replay-plan.json" --trace "$OUT/replay-trace-seed$seed.json" --out "$OUT" ) > "$OUT/ctl1-$arm-replay-seed$seed.out" 2>&1
    tail -3 "$OUT/ctl1-$arm-replay-seed$seed.out" | cut -c1-250; guard_end "ctlreplay1-$arm-s$seed"
  done
}
cell_gsm8k() {
  for p in A1 B1 A2; do arm=${p:0:1}; u=$UA; rec=$RA; [[ $arm == B ]] && { u=$UB; rec=$RB; }
    guard_start "gsm8k-$p"; le quality "$p-gsm8k" "$u" "$rec" "$OUT/$p-gsm8k.out" --bench gsm8k --limit 200 --max-tokens 1024 --pin-gsm8k; guard_end "gsm8k-$p"; done
}

cell_soak() { # soak of the shipping config at current main: mixed greedy / T=1.0 replays (scripts/soak2/replay2.py), K=2, fresh seeds, SOAKMIN minutes (default 25). Stability, not timing: clock guard logged, never aborts.
  local SOAKMIN=${SOAKMIN:-25} seed=${SOAKSEED:-81} fails=0 st end
  scp -q -o BatchMode=yes "$REPO/scripts/soak2/sampler.sh" "$HOST:intarsiaml-ab35d/out/" || { log "sampler copy failed"; return; }
  local SAMP; SAMP=$(rsh "nohup bash $W/sampler.sh $SPID $PORT $W/sample-$WN.tsv 60 > /dev/null 2>&1 < /dev/null & echo \$!")
  mark "start:soak"; snap soak before; st=$(date +%s); end=$((st + SOAKMIN*60)); log "soak start, no new replay after $(date -r $end +%T); vLLM counters $(vcount)"
  while (( $(date +%s) < end )); do
    ( cd /opt/code/deepeval && perl -e 'alarm 1500; exec @ARGV' uv run python "$REPO/scripts/soak2/replay2.py" --url "$UB" --model $MODEL --arm "S$seed" --order interleaved --seed "$seed" --streams 2 \
        --plan "$OUT/replay-plan.json" --trace "$OUT/soak-trace-seed$seed.json" --out "$OUT" --temp-mode mixed ) > "$OUT/S$seed-replay2.out" 2>&1
    tail -3 "$OUT/S$seed-replay2.out" | cut -c1-200
    if [[ ! -s $OUT/S$seed-interleaved-seed$seed-k2.summary.json ]]; then fails=$((fails+1)); log "replay S$seed produced no summary (consecutive failures $fails)"; (( fails >= 2 )) && { log "ABORT soak: two failed replays"; break; }; else fails=0; fi
    curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$WN.metrics.after-seed$seed"
    seed=$((seed+1)); rsh "kill -0 $SPID" 2>/dev/null || { log "SERVER GONE"; break; }
  done
  mark "end:soak"; snap soak after; rsh "kill $SAMP 2>/dev/null"; rsh "cat $W/sample-$WN.tsv" > "$OUT/sample-$WN.tsv"
  log "soak done: $((seed-${SOAKSEED:-81})) replays in $(( ($(date +%s)-st)/60 )) min; vLLM counters $(vcount)"
}

log "window $WN cells: $CELLS; fleetq job $FLEETQ_JOB"
start_b
for c in $CELLS; do "cell_$c"; done
finish_b
