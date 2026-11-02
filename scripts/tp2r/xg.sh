#!/usr/bin/env bash
# Lane TP2R: re-measure CUDA tensor parallelism, 35B across the two Sparks (rank 0 = spark1 `--tp-remote`, rank 1 = spark2
# `intarsia tp-worker`, NCCL over ConnectX-7 port f1). Derived from scripts/tp2-graphs/xg.sh (window mode only: the gates,
# mutants, nsys and soak steps are dropped). ONE fleetq job holds both Sparks:
#   MODE=window fleetq run --host spark1,spark2 --exclusive --takes-down flight,guidance,capcom,retro \
#     --mem spark1=100,spark2=95 --expect 80 --label "..." -- scripts/tp2r/xg.sh      (no --each: one driver)
# Steps (STEPS picks, default all):
#   ab       interleaved A/B x ROUNDS (default 3), order reversed in even rounds. Arms (all `--spec off`, as in 2026-10-21):
#              tp1g   tp=1, graphs         tp2g   tp=2, graphs, INTARSIA_TP_FUSION=0 on both ranks
#              tp2gf  tp=2, graphs, INTARSIA_TP_FUSION=1 on both ranks
#            odd rounds add EXTRA_ARMS (default tp1m = tp=1 with the shipped default spec, MTP on for 35B on CUDA)
#   streams  tp1s: tp=1 `--max-batch 16 --spec off`, bench --concurrency 1,4,16 at 1k; tp2s: the same flags under tp=2
#            (expected refusal: batched decode is hip-only under --tp; the exact error is captured, nothing is patched)
#   mtp      tp2mtp: tp=2 with `--spec mtp` on both ranks (expected refusal; the exact error is captured)
# The window stops both vLLM containers and ALWAYS restarts them (trap on EXIT: docker start, wait for /v1/models), kills
# every process it started, and appends one line per role to raw/downtime.log. DRY=1 prints every remote command.
# Binaries: ~/intarsia-tp2/bin/tp2r-$TREE-intarsia on both boxes (TREE default base; scripts/tp2r/build.sh).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=${MODE:?window}
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-29-tp2r-refresh/raw
OUT=${OUT:-$MODE-$(date +%H%M)}
L=$RAW/$OUT; mkdir -p "$L"
TREE=${TREE:-base}
MAX_MIN=${MAX_MIN:-55}   # window work budget (the vLLM restart comes on top)
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
BIN="\$HOME/intarsia-tp2/bin/tp2r-$TREE-intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
VLLM_C=missioncntrl-guidance
vllm_url() { [[ $1 == spark1 ]] && echo http://100.120.88.104:8000/v1/models || echo http://100.96.95.89:8000/v1/models; } # bash 3.2: no -A
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
r() { local h=$1; shift; if [[ -n ${DRY:-} ]]; then echo "DRY[$h] $*"; return 0; fi; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# DRY=1 (with any FLEETQ_JOB): print the remote commands, run nothing remote, no waits
if [[ -n ${DRY:-} ]]; then sleep() { :; }; rsync() { :; }; scp() { echo "DRY[scp] $*"; }; curl() { return 0; }; fi
# the unfused baseline: fusion is on by default for CUDA at tp=2, so every step meant as "unfused" sets this on both ranks
UNF=INTARSIA_TP_FUSION=0

cleanup_procs() {
  for h in spark1 spark2; do
    r $h "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[i]ntarsia-tp2/bin/tp2r-'; pkill -TERM -f '[n]sys profile'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-tp2/bin/tp2r-' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-tp2/bin/tp2r-'; true"
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
      printf '%s job=%s role=%s stop=%s up=%s down=%d s label="TP2R tp1 vs tp2 %s" serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" "$role" \
        "$(date -r $T_STOP +%H:%M:%S)" "$(date -r $t +%H:%M:%S)" $(( t - T_STOP )) "$OUT" "$([[ $ok == 1 ]] && echo yes || echo NO)" >> "$RAW/downtime.log"
    done
  fi
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-tp2/xout/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM
  exit $rc
}
trap restore EXIT INT TERM

for h in spark1 spark2; do r $h "mkdir -p ~/intarsia-tp2/xout/$OUT; rm -f ~/intarsia-tp2/stop-worker; echo $h: \$($BIN --version | tr '\n' ' ')" | tee -a "$L/steps.log"; done
# pre-flight: both boxes must run the binary built from EXPECT_HEAD, else exit before anything is taken down
if [[ -z ${DRY:-} ]]; then
  for h in spark1 spark2; do
    v=$(r $h "strings $BIN | grep -m1 -o 'head=[0-9a-f]* code-tree=[0-9a-f]* dirty=[a-z]*'")  # --version prints no stamp
    [[ $v == *"head=${EXPECT_HEAD:-694b519}"* ]] || { log "PRE-FLIGHT FAIL: $h binary is not head=${EXPECT_HEAD:-694b519}: $v"; exit 3; }
  done
fi
if [[ $MODE == window ]]; then
  T_STOP=$(date +%s); [[ -n ${DRY:-} ]] || STOPPED=1  # DRY: no restore, no downtime.log line
  for h in spark1 spark2; do r $h "docker stop -t 30 $VLLM_C >/dev/null" && log "vLLM stopped on $h"; done
  for h in spark1 spark2; do
    for i in $(seq 1 30); do gb=$(r $h "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"); (( gb >= 95 )) && break; sleep 2; done
    log "$h MemAvailable $gb GB"
  done
fi
r spark1 "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader" | tee -a "$L/steps.log"

XO='~/intarsia-tp2/xout/'$OUT
GD='$HOME/intarsia-tp2/golden'; VG='$HOME/intarsiaml-golden'; DD='$HOME/intarsia-tp2/dumps'  # expanded on the Sparks
# rank 1 on spark2: a loop of tp-workers (one per rank-0 connection) until stop-worker. WENV: worker env; WWRAP: a
# command prefix (nsys); WBIN: another binary.
worker_start() { # <tag> <flags...>
  local tag=$1; shift
  r spark2 "rm -f ~/intarsia-tp2/stop-worker; nohup bash -c 'i=0; while [ ! -f ~/intarsia-tp2/stop-worker ]; do i=\$((i+1)); env $NCCLENV ${XENV:-} ${WENV:-} ${WWRAP:-} ${WBIN:-$BIN} tp-worker --listen $WIP:$WPORT $* >> $XO/worker-$tag.log 2>&1; echo \"worker exit \$? (\$i)\" >> $XO/worker-$tag.log; sleep 1; done' > /dev/null 2>&1 < /dev/null &"
  sleep 1
}
worker_stop() {
  r spark2 "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[t]p-worker --listen'; for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen' >/dev/null || break; sleep 1; done; pkill -KILL -f '[t]p-worker --listen'; true"
}
# greedy ids: <tag> <chars> <binary> <extra flags>; ids to $XO/gen-<tag>.ids, worker's graph line logged
GEN_FLAGS="-m $M35 --backend cuda --tp 2 --max-seq 20480 --mem-budget 16 --profile-every 0"
xgen() {
  local tag=$1 chars=$2 bin=$3; shift 3
  WBIN=$bin worker_start $tag $GEN_FLAGS "$@"
  r spark1 "env $NCCLENV ${XENV:-} timeout ${GEN_TIMEOUT:-600} $bin generate $GEN_FLAGS $* --tp-remote $WIP:$WPORT --chat --no-think --ignore-eos --show-ids \
    -p \"\$(head -c $chars ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 256 > $XO/gen-$tag.out 2> $XO/gen-$tag.err; echo $tag rc=\$?; grep -aE 'finish=|graph' $XO/gen-$tag.err | head -3; grep -a '^ids:' $XO/gen-$tag.err > $XO/gen-$tag.ids" | tee -a "$L/steps.log"
  worker_stop
  rank1_check $tag "$@"
}
# rank 1's graph counters for run <tag>; a graphs run (no --no-graphs in <flags>) must replay: else REPLAY-CHECK FAIL
rank1_check() { # <tag> <flags...>
  local tag=$1; shift
  local line; line=$(r spark2 "grep -a 'decode graph' $XO/worker-$tag.log | tail -1")
  echo "  rank1 $tag: $line" | tee -a "$L/steps.log"
  if [[ " $* " != *" --no-graphs "* ]]; then
    [[ $line =~ replays=([0-9]+) ]] && (( ${BASH_REMATCH[1]} > 0 )) && log "  REPLAY-CHECK ok $tag" || log "  REPLAY-CHECK FAIL $tag: rank 1 did not replay"
  fi
}

# ---- window (vLLM down): interleaved A/B, then the streams and MTP steps ----
sflags() { echo "-m $M35 --backend cuda --max-seq ${SMAXSEQ:-20480} --mem-budget ${MEMB:-45}"; }
# One serving arm. TRY=1: expect the server to refuse to start (--no-supervisor so the process exits; no bench; the error
# lines of both ranks go to steps.log). CONC/CONC_SIZE/SIZES/GEN/REPS shape the bench.
bench_arm() { # <arm> <round> <rank-0 env> <worker flags or "-" for tp=1> <serve extra...>
  local arm=$1 rd=$2 senv=$3 wflags=$4; shift 4
  (( $(left) > 120 )) || { log "SKIP arm $arm round $rd: $(left) s left"; return 0; }
  local o=$XO/$arm-r$rd SFLAGS; SFLAGS=$(sflags)
  local ns=""; [[ -n ${TRY:-} ]] && ns="--no-supervisor"
  log "arm $arm r$rd: env[$senv] worker[$wflags] serve[$SFLAGS $* $ns]"
  if [[ $wflags != - ]]; then WENV="$senv" worker_start $arm-r$rd --serving $SFLAGS --tp 2 $wflags; fi
  r spark1 "mkdir -p $o; nohup env $NCCLENV $senv $BIN serve $SFLAGS --port 8090 --host 0.0.0.0 --served-model-name intarsia-35b $ns $* > $o/serve.log 2>&1 < /dev/null & echo \$! > $o/serve.pid"
  local up=0 dead=0
  for i in $(seq 1 ${UPWAIT:-150}); do
    curl -sf -m 3 http://spark-4e77:8090/health >/dev/null && { up=1; break; }
    sleep 2
    [[ -n ${TRY:-} ]] && { r spark1 "kill -0 \$(cat $o/serve.pid) 2>/dev/null" || { dead=1; break; }; }
  done
  if [[ $up == 1 ]]; then
    r spark1 "grep -aiE 'spec|mtp|fusion|graph' $o/serve.log | head -4 | cut -c1-200" | tee -a "$L/steps.log"
    local gx=""; [[ $arm == *g* ]] && gx="--require-graphs"
    r spark1 "$BIN bench speed --url http://127.0.0.1:8090 --model intarsia-35b --sizes ${SIZES:-1024,4096,16384} --gen ${GEN:-128} --reps ${REPS:-2} --warmup 1 --thinking off --at-most $gx --concurrency ${CONC:-1} --conc-size ${CONC_SIZE:-1024} --label tp2r-$arm-r$rd --out-dir $o > $o/bench.log 2>&1; echo $arm r$rd rc=\$?; tail -n ${TAIL:-5} $o/bench.log | cut -c1-170" | tee -a "$L/steps.log"
  else
    log "arm $arm r$rd: server did not come up (process exited: $dead)"
    r spark1 "echo rank0 serve.log; grep -aE 'rror|nsupported|refus|panic' $o/serve.log | head -6 | cut -c1-400; echo ---; tail -n 6 $o/serve.log | cut -c1-400" | tee -a "$L/steps.log"
    [[ $wflags != - ]] && r spark2 "echo rank1 worker log; grep -aE 'rror|nsupported|refus|panic' $XO/worker-$arm-r$rd.log | head -6 | cut -c1-400; echo ---; tail -n 6 $XO/worker-$arm-r$rd.log | cut -c1-400" | tee -a "$L/steps.log"
  fi
  r spark1 "kill \$(cat $o/serve.pid) 2>/dev/null; for i in \$(seq 1 30); do kill -0 \$(cat $o/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 \$(cat $o/serve.pid) 2>/dev/null; true"
  if [[ $wflags != - ]]; then worker_stop; [[ -z ${TRY:-} ]] && rank1_check $arm-r$rd $wflags; fi
}
if [[ $MODE == window ]] && want ab; then
  # tp1g = tp=1 graphs --spec off, tp2g = tp=2 graphs unfused, tp2gf = tp=2 graphs row-local fusion; order reversed in even rounds
  ARMS_FWD=${ARMS:-tp1g tp2g tp2gf}
  for rd in $(seq 1 ${ROUNDS:-3}); do
    order=$ARMS_FWD; (( rd % 2 == 0 )) && order=$(echo $ARMS_FWD | tr ' ' '\n' | tail -r | tr '\n' ' ')
    (( rd % 2 == 1 )) && order="$order ${EXTRA_ARMS-tp1m}"
    for a in $order; do
      case $a in
        tp1g) bench_arm tp1g $rd "" - --spec off ;;
        tp1m) bench_arm tp1m $rd "" - ;;   # the shipped default spec: MTP on (35B, CUDA, tp=1)
        tp2g) bench_arm tp2g $rd "$UNF" "" --tp 2 --tp-remote $WIP:$WPORT --spec off ;;
        tp2gf) bench_arm tp2gf $rd "INTARSIA_TP_FUSION=1" "" --tp 2 --tp-remote $WIP:$WPORT --spec off ;;
        *) log "unknown arm $a" ;;
      esac
    done
  done
fi
if [[ $MODE == window ]] && want streams; then
  # one round: aggregate decode of 1/4/16 simultaneous 1k-prompt streams
  SMAXSEQ=8192 MEMB=40 SIZES=1024 CONC=1,4,16 TAIL=14 bench_arm tp1s 1 "" - --max-batch 16 --spec off
  SMAXSEQ=8192 MEMB=40 SIZES=1024 CONC=1,4,16 TAIL=14 TRY=1 UPWAIT=60 bench_arm tp2s 1 "$UNF" "" --tp 2 --tp-remote $WIP:$WPORT --max-batch 16 --spec off
fi
if [[ $MODE == window ]] && want mtp; then
  # MTP under tp=2: expected to be refused on load (the 35B MTP head is MoE, its experts are not sharded)
  TRY=1 UPWAIT=60 bench_arm tp2mtp 1 "" "--spec mtp" --tp 2 --tp-remote $WIP:$WPORT --spec mtp
fi
log "== $MODE done, $(left) s of the work budget left"
