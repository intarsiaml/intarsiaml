#!/usr/bin/env bash
# Lane G1b (docs/design/35b-tp2.md): the 35B at tp=2 ACROSS the two Sparks. rank 0 = spark1 (`--tp-remote`), rank 1 =
# spark2 (`intarsia tp-worker`), NCCL 2.29.7 over ConnectX-7 port f1 (both PCIe halves). Runs HERE (the Mac) as the one
# command of ONE fleetq job that holds both Sparks:
#   smoke (vLLM up):    MODE=smoke  fleetq run --host spark1,spark2 --mem spark1=20,spark2=20 --label "..." -- scripts/35b-tp2/xhost.sh
#   speed (vLLM down):  MODE=window fleetq run --host spark1,spark2 --exclusive --takes-down flight,retro \
#                         --mem spark1=100,spark2=95 --label "..." -- scripts/35b-tp2/xhost.sh
# The window stops both vLLM containers and ALWAYS restarts them (trap on EXIT: docker start, wait for /v1/models),
# kills every process it started, and appends one line per role to raw/downtime.log. STEPS picks steps (default all).
# Binary: ~/intarsia-tp2/bin/$TREE-intarsia on both boxes (TREE default xdev). Output: raw/$OUT here.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=${MODE:?smoke or window}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-21-35b-tp2-cuda/raw
OUT=${OUT:-x-$MODE-$(date +%H%M)}
L=$RAW/$OUT; mkdir -p "$L"
TREE=${TREE:-xdev}
MAX_MIN=${MAX_MIN:-24}   # window work budget (vLLM restart comes on top; the cap is 30 min down)
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
P='$HOME/intarsia-tp2'; BIN="$P/bin/$TREE-intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
VLLM_C=missioncntrl-guidance
vllm_url() { [[ $1 == spark1 ]] && echo http://100.120.88.104:8000/v1/models || echo http://100.96.95.89:8000/v1/models; } # bash 3.2 (macOS): no -A
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }

cleanup_procs() {
  for h in spark1 spark2; do
    r $h "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[i]ntarsia-tp2/bin/'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-tp2/bin/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-tp2/bin/'; true"
  done
}
STOPPED=0; T_STOP=0
restore() {
  local rc=$?
  log "restore (exit $rc)"
  cleanup_procs
  if [[ $STOPPED == 1 ]]; then
    for h in spark1 spark2; do r $h "docker start $VLLM_C >/dev/null" || log "WARNING: docker start on $h failed"; done
    for h in spark1 spark2; do
      local ok=0
      for i in $(seq 1 240); do curl -sf -m 5 "$(vllm_url $h)" >/dev/null && { ok=1; break; }; sleep 5; done
      local t=$(date +%s)
      [[ $ok == 1 ]] && log "vLLM on $h serving again" || log "WARNING: vLLM on $h did not answer /v1/models within 20 min"
      local role=$([[ $h == spark1 ]] && echo spark1-vllm || echo spark2-vllm)
      printf '%s job=%s role=%s stop=%s up=%s down=%d s label="G1b 35B tp2 cross-host speed" serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" "$role" \
        "$(date -r $T_STOP +%H:%M:%S)" "$(date -r $t +%H:%M:%S)" $(( t - T_STOP )) "$([[ $ok == 1 ]] && echo yes || echo NO)" >> "$RAW/downtime.log"
    done
  fi
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-tp2/xout/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM
  exit $rc
}
trap restore EXIT INT TERM

for h in spark1 spark2; do r $h "mkdir -p ~/intarsia-tp2/xout/$OUT; rm -f ~/intarsia-tp2/stop-worker; echo $h: \$($BIN --version | tr '\n' ' ')" | tee -a "$L/steps.log"; done
if [[ $MODE == window ]]; then
  T_STOP=$(date +%s); STOPPED=1
  for h in spark1 spark2; do r $h "docker stop -t 30 $VLLM_C >/dev/null" && log "vLLM stopped on $h"; done
  for h in spark1 spark2; do
    for i in $(seq 1 30); do gb=$(r $h "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"); (( gb >= 95 )) && break; sleep 2; done
    log "$h MemAvailable $gb GB"
  done
fi
r spark1 "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader" | tee -a "$L/steps.log"

# rank 1 on spark2: a loop of tp-workers with the given model flags (one per rank-0 connection) until stop-worker
worker_start() { # <tag> <flags...>
  local tag=$1; shift
  r spark2 "rm -f ~/intarsia-tp2/stop-worker; nohup bash -c 'i=0; while [ ! -f ~/intarsia-tp2/stop-worker ]; do i=\$((i+1)); env $NCCLENV $BIN tp-worker --listen $WIP:$WPORT $* >> ~/intarsia-tp2/xout/$OUT/worker-$tag.log 2>&1; echo \"worker exit \$? (\$i)\" >> ~/intarsia-tp2/xout/$OUT/worker-$tag.log; sleep 1; done' > /dev/null 2>&1 < /dev/null &"
}
worker_stop() {
  r spark2 "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[t]p-worker --listen'; for i in \$(seq 1 20); do pgrep -f '[t]p-worker --listen' >/dev/null || break; sleep 1; done; pkill -KILL -f '[t]p-worker --listen'; true"
}
XO='~/intarsia-tp2/xout/'$OUT

# ---- smoke: the CLI paths with a remote rank, and cross-host = one-box bits ----
if [[ $MODE == smoke ]]; then
  GEN_FLAGS="-m $M35 --backend cuda --tp 2 --max-seq 20480 --mem-budget 16 --profile-every 0"
  if want gen; then
    worker_start gen $GEN_FLAGS
    r spark1 "env $NCCLENV timeout 900 $BIN generate $GEN_FLAGS --tp-remote $WIP:$WPORT --chat --no-think --ignore-eos --show-ids \
      -p \"\$(head -c 5000 ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 256 > $XO/gen-x-5k.out 2> $XO/gen-x-5k.err; echo rc=\$?; grep -a finish= $XO/gen-x-5k.err | head -1; grep -a '^ids:' $XO/gen-x-5k.err > $XO/gen-x-5k.ids" | tee -a "$L/steps.log"
    worker_stop
    rsync -a spark2:intarsia-tp2/out/gates2/gen-tp2-5k.ids "$L/onebox-gen-tp2-5k.ids"
    rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/"
    if cmp -s <(cut -d: -f2- "$L/gen-x-5k.ids") <(cut -d: -f2- "$L/onebox-gen-tp2-5k.ids"); then log "IDS cross-host vs one-box tp2 (5k, 256): IDENTICAL"; else log "IDS cross-host vs one-box tp2 (5k, 256): DIFFER"; fi
  fi
  if want acc; then
    r spark2 "scp -q ~/intarsia-tp2/out/gates2/S1-short.safetensors spark1:intarsia-tp2/golden/S1-short.safetensors"
    AF="-m $M35 --backend cuda --tp 2 --max-seq 16384 --prefill-chunk 2048 --mem-budget 18"
    worker_start acc $AF
    r spark1 "env $NCCLENV timeout 1200 $BIN accuracy $AF --tp-remote $WIP:$WPORT --self-golden ~/intarsia-tp2/golden/S1-short.safetensors --mode both --allow-config-mismatch \
      --self-max-kl 1.2e-3 --self-max-pos-kl 9.8e-2 --self-max-flip-pct 0.84 --self-max-tail 4.2e-4 --self-prompt-factor 1.92 --pre-change ${PRE_CHANGE:?} > $XO/self-x-short.log 2>&1; echo rc=\$?; grep -aE 'SELF-STATS|self-golden gate|^Error' $XO/self-x-short.log | cut -c1-260" | tee -a "$L/steps.log"
    worker_stop
  fi
fi

# ---- serve + bench speed (smoke: one small size; window: tp=1 graphs, tp=1 eager, tp=2 cross-host + localeval) ----
SFLAGS="-m $M35 --backend cuda --max-seq ${SMAXSEQ:-20480} --mem-budget ${MEMB:-45}"
bench_arm() { # <arm> <bench extra> <serve extra...>
  local arm=$1 bx=$2; shift 2
  want arm-$arm || return 0
  (( $(left) > 150 )) || { log "SKIP arm $arm: $(left) s left"; return 0; }
  local o=$XO/$arm
  r spark1 "mkdir -p $o; nohup env $NCCLENV $BIN serve $SFLAGS --port 8090 --host 0.0.0.0 --served-model-name intarsia-35b $* > $o/serve.log 2>&1 < /dev/null & echo \$! > $o/serve.pid"
  local up=0; for i in $(seq 1 150); do curl -sf -m 3 http://spark-4e77:8090/health >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { log "arm $arm: server did not come up"; r spark1 "tail -5 $o/serve.log"; r spark1 "kill \$(cat $o/serve.pid)"; return 1; }
  log "arm $arm: serving"
  r spark1 "$BIN bench speed --url http://127.0.0.1:8090 --model intarsia-35b --sizes ${SIZES:-1024,4096,16384} --gen ${GEN:-128} --reps ${REPS:-2} --warmup 1 --thinking off --label g1b-$arm --out-dir $o $bx > $o/bench.log 2>&1; echo rc=\$?; tail -n 6 $o/bench.log | cut -c1-170" | tee -a "$L/steps.log"
  if [[ ${LOCALEVAL:-} == "$arm" ]] && (( $(left) > 240 )); then
    log "localeval arm $arm"
    (cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url http://spark-4e77:8090/v1 --model intarsia-35b \
      --thinking off --sizes ${SIZES:-1024,4096,16384} --gen 128 --reps 3 --label intarsia-35b-tp2-xhost) > "$L/localeval-$arm.log" 2>&1
    log "localeval arm $arm exit $?: $(tail -n 12 "$L/localeval-$arm.log" | tr '\n' '|' | cut -c1-600)"
  fi
  r spark1 "kill \$(cat $o/serve.pid); for i in \$(seq 1 30); do kill -0 \$(cat $o/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 \$(cat $o/serve.pid) 2>/dev/null; true"
}
if [[ $MODE == smoke ]]; then
  if want serve; then
    worker_start serve --serving $SFLAGS --tp 2
    SIZES=1024 GEN=32 REPS=1 bench_arm tp2x-smoke "" --tp 2 --tp-remote $WIP:$WPORT
    worker_stop
  fi
else
  bench_arm tp1-graphs "--require-graphs"
  bench_arm tp1-eager "" --no-graphs
  worker_start serve --serving $SFLAGS --tp 2
  bench_arm tp2x "" --tp 2 --tp-remote $WIP:$WPORT
  worker_stop
fi
log "== $MODE done, $(left) s of the work budget left"
