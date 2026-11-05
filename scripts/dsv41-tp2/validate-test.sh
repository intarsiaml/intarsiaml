#!/bin/bash
# Lane dsv41/validate: hardware validation of the DeepSeek-V4.1 TP=2 paths that became the default this week and never ran
# on hardware: bounded-replay prefill (#303) and encoder-only dead chunks (#305, INTARSIA_DSV41_REPLAY_SKIP). Measurement and test
# only. One fleetq job over both Sparks, vLLM taken down and ALWAYS restored (EXIT trap), memory watcher at 5 GiB.
# Pass rules are stated in docs/results/2026-11-05-dsv41-validate.md before the run. Sessions (fresh ranks each):
#   S1 default  : c1 prefix hits, speed A (default), c2 question-first 32K, c3 thinking, c4 sampled, catspeed
#   S2 off      : c1, c2, c3, c4                                           (INTARSIA_DSV41_REPLAY=0)
#   S3 skip0    : c1 (the cancel case), speed B (REPLAY_SKIP=0)             (INTARSIA_DSV41_REPLAY_SKIP=0)
#   S4 default  : speed A again;  S5 skip0: speed B again                  (ABAB)
# Usage: scripts/dsv41-tp2/validate-test.sh [--dry-run]     (queues itself as one fleetq job; inner mode needs FLEETQ_JOB)
# Knobs: SPARK_DIR (intarsia-dsv41-validate), CAP (66000), KILL_GIB (5), EXPECT_MIN (140), RAW, ONLY ("S1 S2 ..."),
#   (ONLY is comma separated)  C3_TOKENS (1536), SP_SIZES (1024,65536), SP_REPS (2), EARLYOOM_OK ("3,2 2,1"), HOT_MB, TP_SPIN_MS, XENV.
set -u -o pipefail
S="$(cd "$(dirname "$0")" && pwd)"; SELF="$S/validate-test.sh"; REPO="$(cd "$S/../.." && pwd)"

# ---- the vLLM restore, the way run-full.sh does it: `docker start` the guidance container on both Sparks, wait for a
# "Application startup complete" line newer than StartedAt, then `fleetq mark-up`. Container name: one place.
VLLM_C=missioncntrl-guidance
VLLM_ROLES=spark1-vllm,spark2-vllm

# ---- configuration
SPARK_DIR_IN=${SPARK_DIR:-intarsia-dsv41-validate}; SPARK_DIR=$SPARK_DIR_IN
source "$S/spark-dir.sh"
[[ $SPARK_REL == i* ]] || { echo "SPARK_DIR '$SPARK_DIR_IN' must start with the letter i (pkill patterns use [i]...)"; exit 2; }
BIN_BR="$SPARK_DIR/bin/intarsia"; LOGD="$SPARK_DIR/logs"
C0='$HOME/intarsia-dsv41-tp2/cache/r0of2'; C1='$HOME/intarsia-dsv41-tp2/cache/r1of2'
WIP=10.100.152.3; WPORT=29701; SPORT=8899; SURL=http://100.120.88.104:$SPORT
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}"
CAP=${CAP:-66000}                      # 32K prompts at step 3 and 64K items at step 5 (chunk32k-2 used 66000)
TOK=${TOKENIZER:-/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json}
RESDIR=${RESDIR:-$REPO/docs/results/2026-11-05-dsv41-validate}; RAW=${RAW:-$RESDIR/raw}
ITEMS=${ITEMS:-$REPO/.scratch}      # c2-qfirst12.jsonl c3-last8.jsonl recent/recent.4k.jsonl (regenerate: see the results doc)
C3_TOKENS=${C3_TOKENS:-1536}; SP_SIZES=${SP_SIZES:-1024,65536}; SP_REPS=${SP_REPS:-2}
ONLY=${ONLY:-S1,S2,S3,S4,S5}
EXPECT=${EXPECT_MIN:-140}
PLAN_MIN=$EXPECT
VC=$S/validate-client.py
FQ=(fleetq run --host spark1,spark2 --exclusive --takes-down "$VLLM_ROLES" --mem spark1=116,spark2=116 --expect "$EXPECT"
    --label "dsv41/validate: replay + dead-chunk hardware checks (vLLM restored after)" -- env REPLAY_TEST_INNER=1 SPARK_DIR="$SPARK_DIR_IN" CAP="$CAP" ONLY="$ONLY" EXPECT_MIN="$EXPECT" C3_TOKENS="$C3_TOKENS" SP_SIZES="$SP_SIZES" SP_REPS="$SP_REPS" ${KILL_GIB:+KILL_GIB=$KILL_GIB} ${XENV:+XENV="$XENV"} "$SELF")
if [[ ${1:-} == --dry-run ]]; then echo "DRY RUN: nothing is run."; printf '%q ' "${FQ[@]}"; echo; echo "sessions: $ONLY; items in $ITEMS:"; ls -la "$ITEMS"/c2-qfirst12.jsonl "$ITEMS"/c3-last8.jsonl "$ITEMS"/recent/recent.4k.jsonl; exit 0; fi
if [[ ${REPLAY_TEST_INNER:-0} != 1 ]]; then echo "queueing as one fleetq job over spark1,spark2 (--expect $EXPECT)"; exec "${FQ[@]}"; fi
: "${FLEETQ_JOB:?inner mode runs inside a fleetq job}"

# ============================================================ inner part
mkdir -p "$RAW"
mkdir -p "$RAW"
LOG=$RAW/steps.log
log() { echo "$(date +%T) $*" | tee -a "$LOG"; }
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
lib() { uv run --no-project --with numpy --with safetensors python "$S/replay-test-lib.py" "$@"; }
FAIL_WHY=""; STEP_NOTE=""
fail() { FAIL_WHY="$*"; return 1; }
RUN0=$(date +%s)
log "validate-test start: sessions $ONLY, planned ${PLAN_MIN} min, fleetq job $FLEETQ_JOB, out $RAW"

# ---- memory watchers and the restore (as in run-full.sh)
STOPPED=0; WATCH=""; BGPID=""
RES_N=(); RES_ST=(); RES_EL=(); RES_PL=()
summary() {
  local i
  log "================ SUMMARY (total elapsed $(( ($(date +%s) - RUN0) / 60 )) min, planned ${PLAN_MIN} min) ================"
  for ((i = 0; i < ${#RES_N[@]}; i++)); do
    log "STEP ${RES_N[$i]}: ${RES_ST[$i]}  elapsed $(( RES_EL[i] / 60 ))m$(( RES_EL[i] % 60 ))s (planned ${RES_PL[$i]} min)  $(echo "${RES_N[$i]}")"
  done
}
kill_ours() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]${SPARK_REL:1}/bin/intarsia'; for i in \$(seq 1 20); do pgrep -f '[i]${SPARK_REL:1}/bin/intarsia' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]${SPARK_REL:1}/bin/intarsia'; true"; done; }
restore() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [[ -z $BGPID ]] || killtree "$BGPID"
  summary
  log "restore (exit $rc)"
  [[ -z $WATCH ]] || kill "$WATCH" 2>/dev/null
  if [[ $STOPPED == 1 ]]; then
    kill_ours
    for h in spark1 spark2; do r $h "pkill -f '[b]in/memfine.sh'; true"; done
    for h in spark1 spark2; do scp -q "$h:$SPARK_REL/logs/$h-${MF:-none}" "$RAW/" 2>/dev/null; done
    # a `docker logs --since` prints nothing on these hosts; wait for a "startup complete" line stamped (UTC) after StartedAt
    for h in spark1 spark2; do r $h "docker start $VLLM_C >/dev/null" || log "WARNING: docker start on $h failed"; done
    for h in spark1 spark2; do
      local ok=0 i
      for i in $(seq 1 120); do
        r $h "s=\$(docker inspect -f '{{.State.StartedAt}}' $VLLM_C); t=\$(docker logs -t --tail 2000 $VLLM_C 2>&1 | grep 'Application startup complete' | tail -1 | cut -d' ' -f1); [[ -n \$t && \$t > \$s ]]" && { ok=1; break; }
        sleep 10
      done
      [[ $ok == 1 ]] && log "vLLM on $h up ($(r $h "docker ps --filter name=$VLLM_C --format {{.Status}}"))" || log "WARNING: vLLM on $h not up after 20 min"
    done
    fleetq mark-up spark1-vllm; fleetq mark-up spark2-vllm
  fi
  log "restored"
  exit $rc
}
trap restore EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

# kill a process and its children (macOS has no setsid)
killtree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; return 0; }
# alive HOST PATTERN: 0 = process there, 1 = the host answered and there is none (a dead rank), 2 = the host could not be asked
# (ssh exit 255 on every one of ALIVE_TRIES tries, 5 s apart: step 5 polls for hours, one network blip must not kill an arm)
ALIVE_TRIES=${ALIVE_TRIES:-4}
alive() {
  local rc i
  for ((i = 1; i <= ALIVE_TRIES; i++)); do
    r "$1" "pgrep -f '$2' >/dev/null"; rc=$?
    [[ $rc != 255 ]] && return $((rc != 0))
    log "ssh to $1 failed (try $i of $ALIVE_TRIES)"; sleep 5
  done
  return 2
}
RANK0_PAT="[i]${SPARK_REL:1}/bin/intarsia[^ ]* serve"; RANK1_PAT="[i]${SPARK_REL:1}/bin/intarsia[^ ]* tp-worker"
# ranks_alive: both ranks run (0), a host answered that its rank is gone (1), or a host could not be asked and no rank is known gone (2).
# 2 says nothing about the ranks: an ssh drop from this Mac once stopped two healthy ranks (#8476, 2026-11-05)
ranks_alive() {
  local a=0 b=0
  alive spark1 "$RANK0_PAT" || a=$?
  alive spark2 "$RANK1_PAT" || b=$?
  (( a == 1 || b == 1 )) && return 1
  (( a == 2 || b == 2 )) && return 2
  return 0
}
# bgrun MAX_SECONDS LOGFILE cmd...: run in the background and poll every 15 s (no single wait is long): stops at MAX_SECONDS, at the
# step deadline, or (WATCH_RANKS=1) when a host says its rank has died; an unreachable host only logs. Output goes to LOGFILE; its tail to the steps log. Returns the command's status.
bgrun() {
  local max=$1 lf=$2 t0 rc ra; shift 2; t0=$(date +%s)
  ( "$@" ) > "$lf" 2>&1 < /dev/null &
  BGPID=$!
  while kill -0 "$BGPID" 2>/dev/null; do
    sleep 15
    kill -0 "$BGPID" 2>/dev/null || break
    if (( $(date +%s) - t0 > max )); then killtree "$BGPID"; log "timeout after ${max}s: $*"; tail -n 15 "$lf" | tee -a "$LOG"; BGPID=""; return 124; fi
    if (( $(date +%s) > STEP_DEADLINE )); then killtree "$BGPID"; log "step deadline passed: $*"; BGPID=""; return 124; fi
    if [[ ${WATCH_RANKS:-0} == 1 ]]; then
      ra=0; ranks_alive || ra=$?
      if (( ra == 1 )); then killtree "$BGPID"; log "a rank died during: $*"; tail -n 15 "$lf" | tee -a "$LOG"; BGPID=""; return 125; fi
      (( ra == 2 )) && log "a host is unreachable, ranks left running, still watching: $*"
    fi
  done
  wait "$BGPID"; rc=$?; BGPID=""
  tail -n "${BG_TAIL:-40}" "$lf" | tee -a "$LOG"
  return $rc
}
STEP_DEADLINE=$(( $(date +%s) + 86400 ))

preflight() {
  local h eo
  for h in spark1 spark2; do
    eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
    [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is at -m '$eo', not one of '${EARLYOOM_OK:-3,2 2,1}'; not starting"; return 1; }
    log "earlyoom on $h: -m $eo"
    r $h "test -x $BIN_BR && test -f ~/intarsia-nccl/libnccl.so.2" || { log "binary $BIN_BR or the NCCL library missing on $h"; return 1; }
    r $h "grep -qa INTARSIA_DSV41_REPLAY_SKIP $BIN_BR" || { log "$BIN_BR on $h lacks INTARSIA_DSV41_REPLAY_SKIP: built from a tree without the skip"; return 1; }
  done
  r spark1 "test -f $C0/manifest.json" && r spark2 "test -f $C1/manifest.json" || { log "rank cache missing"; return 1; }
  [[ "$(r spark1 "cat $SPARK_DIR/bin/STAMP")" == "$(r spark2 "cat $SPARK_DIR/bin/STAMP")" ]] || { log "binary stamps differ"; return 1; }
  STAMP=$(r spark1 "cat $SPARK_DIR/bin/STAMP"); log "binary $STAMP; out $RAW"
  [[ $STAMP == 1faac7bf5\ dirty=no* ]] || { log "binary is not the clean 1faac7bf5 build: $STAMP"; return 1; }
  local f; for f in c2-qfirst12.jsonl c3-last8.jsonl recent/recent.4k.jsonl; do [[ -s $ITEMS/$f ]] || { log "item file $ITEMS/$f missing"; return 1; }; done
  LE_RECIPE="intarsia $STAMP scripts/dsv41-tp2/validate-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap $CAP, chunk 2048, engram hot ${HOT_MB:-1024} MiB"
}

vllm_down() {
  for h in spark1 spark2; do r $h "docker stop -t 60 $VLLM_C >/dev/null"; STOPPED=1; done
  log "vLLM stopped on both"
  # our own watcher: kill our ranks on a box under KILL_GIB (default 5) GiB available; memfine.sh logs every second on each box
  ( while true; do
      line="$(date +%T)"
      for h in spark1 spark2; do
        a=$(ssh -n -o ConnectTimeout=5 $h "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA)
        line="$line $h=$a"
        if [[ "$a" != NA ]] && awk "BEGIN{exit !($a < ${KILL_GIB:-5})}"; then ssh -n $h "pkill -KILL -f '[i]${SPARK_REL:1}/bin/intarsia'"; line="$line KILLED-$h"; fi
      done
      echo "$line"; sleep 5
    done ) > "$RAW/mem.log" 2>&1 & WATCH=$!; disown "$WATCH"
  MF=memfine-$(date +%H%M%S).log
  for h in spark1 spark2; do
    scp -q "$S/memfine.sh" "$h:$SPARK_REL/bin/memfine.sh"
    r $h "mkdir -p $LOGD; nohup bash $SPARK_DIR/bin/memfine.sh ${KILL_GIB:-5} '[i]${SPARK_REL:1}/bin/intarsia' > $LOGD/$h-$MF 2>&1 < /dev/null &"
  done
}

# ---- engine: start_ranks BIN CHUNK MODE (off|on|main|default). The SAME env string goes to both ranks; INTARSIA_DSV41_REPLAY is
# removed first (env -u) so an inherited value never leaks, then set to 1 for on, 0 for off and main, and left unset for default.
start_ranks() {
  local bin=$1 chunk=$2 mode=$3 renv t0 up=0 ra
  renv="INTARSIA_TP_SPIN_MS=${TP_SPIN_MS:-100} INTARSIA_DSV41_CAP=$CAP INTARSIA_DSV41_CHUNK=$chunk INTARSIA_DSV41_ENGRAM=\$HOME/dsv41-engram INTARSIA_DSV41_ENGRAM_HOT_MB=${HOT_MB:-1024} RUST_BACKTRACE=1 ${XENV:-} ${SRV_ENV:-}"
  case $mode in off) renv="$renv INTARSIA_DSV41_REPLAY=0" ;; skip0) renv="$renv INTARSIA_DSV41_REPLAY_SKIP=0" ;; default) ;; *) echo "bad mode $mode"; return 1 ;; esac
  TAG=$(date +%H%M%S)
  r spark2 "mkdir -p $LOGD; nohup env -u INTARSIA_DSV41_REPLAY -u INTARSIA_DSV41_REPLAY_SKIP $NCCLENV $renv $bin tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/worker-$TAG.log 2>&1 < /dev/null &"
  sleep 2
  r spark1 "mkdir -p $LOGD; nohup env -u INTARSIA_DSV41_REPLAY -u INTARSIA_DSV41_REPLAY_SKIP $NCCLENV $renv $bin serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/serve-$TAG.log 2>&1 < /dev/null &"
  log "ranks started (tag $TAG, $(basename "$bin"), chunk $chunk, replay $mode)"
  t0=$(date +%s)
  while (( $(date +%s) - t0 < 900 )); do
    curl -sf -m 5 "$SURL/health" >/dev/null && { up=1; break; }
    ra=0; alive spark1 "$RANK0_PAT" || ra=$?
    (( ra == 1 )) && { log "rank 0 exited"; break; }
    ra=0; alive spark2 "$RANK1_PAT" || ra=$?
    (( ra == 1 )) && { log "rank 1 exited"; break; }
    sleep 5
  done
  log "serve up=$up after $(( $(date +%s) - t0 )) s"
  r spark1 "grep -aEi 'dsv41|error|panick|replay' $LOGD/serve-$TAG.log | tail -25" | tee -a "$LOG"
  r spark2 "grep -aEi 'dsv41|error|panick|replay' $LOGD/worker-$TAG.log | tail -15" | tee -a "$LOG"
  [[ $up == 1 ]] || { fail "ranks did not come up (see $LOG)"; return 1; }
}
# serve_cfg NAME BIN CHUNK MODE cmd...: fresh ranks, run cmd, keep the logs, stop the ranks. Returns cmd's status.
serve_cfg() {
  local name=$1 bin=$2 chunk=$3 mode=$4 rc; shift 4
  kill_ours
  start_ranks "$bin" "$chunk" "$mode" || { scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$RAW/serve-$name.log" 2>/dev/null; kill_ours; return 1; }
  WATCH_RANKS=1 "$@"; rc=$?
  scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$RAW/serve-$name.log" 2>/dev/null; scp -q "spark2:$SPARK_REL/logs/worker-$TAG.log" "$RAW/worker-$name.log" 2>/dev/null
  kill_ours
  return $rc
}

# ---- clients
# gold OUT NOTE promptargs...: greedy token ids (tools/dsv41/golden_tf.py tokens), temperature 0, ignore_eos
gold() {
  local out=$1 note=$2; shift 2
  bgrun 1200 "$out.log" bash -c 'cd "$1" && shift && exec uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens "$@"' _ "$REPO" \
    --base-url "$SURL/v1" --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$out" --note "$note $STAMP cap $CAP" "$@"
}
# le ...: localeval inside our slot (FLEETQ_JOB is set, so it does not queue again)
le() { ( cd /opt/code/deepeval && uv run localeval "$@" --url "$SURL/v1" --model dsv41-intarsia --thinking off --recipe "$LE_RECIPE" ); }
result_of() { grep -ao '/[^ ]*/results/[^ ]*result.json' "$1" | tail -1; }

# ============================================================ tasks and sessions
TASK_N=(); TASK_RC=()
# task NAME MAXSEC cmd...: one client or localeval command, in the background with the rank watch; a failure is recorded and the
# window goes on (the summary and the exit status carry it)
task() {
  local name=$1 max=$2 t0 rc; shift 2; t0=$(date +%s)
  log "--- task $name (max ${max}s)"
  BG_TAIL=${BG_TAIL:-25} bgrun "$max" "$RAW/$name.log" "$@"; rc=$?
  TASK_N+=("$name"); TASK_RC+=("$rc"); log "task $name rc=$rc in $(( $(date +%s) - t0 )) s"
  return 0
}
keep_result() { # NAME: the localeval result.json named in $RAW/NAME.log, copied to $RAW/NAME.result.json
  local res; res=$(result_of "$RAW/$1.log"); [[ -f ${res:-/nonexistent} ]] && cp "$res" "$RAW/$1.result.json" && cp -r "$(dirname "$res")" "$RAW/$1.dir" 2>/dev/null; return 0
}
t_c1() { task "c1-$1" 900 uv run --no-project --with tokenizers python "$VC" c1 "$SURL" "$RAW/c1-$1.json" "$1" "$TOK"; }
t_speed() { task "speed-$1" 1800 le speed --sizes "$SP_SIZES" --gen 8 --reps "$SP_REPS" --label "dsv41validate-speed-$1"; keep_result "speed-$1"; }
t_c2() { task "c2-$1" 2400 le quality --dataset "$ITEMS/c2-qfirst12.jsonl" --judge none --max-tokens 400 --label "dsv41validate-c2-$1"; keep_result "c2-$1"; }
t_c3() { task "c3-$1" 3000 uv run --no-project python "$VC" c3 "$SURL" "$RAW/c3-$1.json" "$1" "$C3_TOKENS" "4k=$ITEMS/recent/recent.4k.jsonl:8" "32k=$ITEMS/c3-last8.jsonl:8"; }
t_c4() { task "c4-$1" 900 uv run --no-project python "$VC" c4 "$SURL" "$RAW/c4-$1.json" "$1" "$ITEMS/recent/recent.4k.jsonl"; }
t_cat() { task "catspeed-$1" 1500 le quality --dataset /opt/code/deepeval/datasets/categories-rcp.jsonl --dataset "$S/chat-rcp.jsonl" --judge none --max-tokens 256 --label "dsv41validate-cat-$1"; keep_result "catspeed-$1"; }
S1() { t_c1 default; t_speed def1; t_c2 default; t_c3 default; t_c4 default; t_cat default; }
S2() { t_c1 off; t_c2 off; t_c3 off; t_c4 off; }
S3() { t_c1 skip0; t_speed skip0_1; }
S4() { t_speed def2; }
S5() { t_speed skip0_2; }
sess_min() { case $1 in S1) echo 50 ;; S2) echo 55 ;; S3) echo 20 ;; S4) echo 12 ;; S5) echo 14 ;; esac; }
sess_mode() { case $1 in S2) echo off ;; S3|S5) echo skip0 ;; *) echo default ;; esac; }

prep_ok=1
preflight || { log "PREP FAILED: preflight"; exit 1; }
vllm_down
RC=0
for ss in ${ONLY//,/ }; do
  t0=$(date +%s); FAIL_WHY=""; STEP_DEADLINE=$(( t0 + 60 * $(sess_min "$ss") ))
  log "=== SESSION $ss: mode $(sess_mode "$ss"), chunk 2048, CAP $CAP (planned $(sess_min "$ss") min) ==="
  if serve_cfg "$ss-$(sess_mode "$ss")" "$BIN_BR" 2048 "$(sess_mode "$ss")" "$ss"; then st=OK; else st=FAIL; RC=1; log "SESSION $ss FAILED: ${FAIL_WHY:-see $LOG}"; fi
  RES_N+=("$ss"); RES_ST+=("$st"); RES_EL+=($(( $(date +%s) - t0 ))); RES_PL+=("$(sess_min "$ss")")
done
for i in "${!TASK_N[@]}"; do log "TASK ${TASK_N[$i]} rc=${TASK_RC[$i]}"; [[ ${TASK_RC[$i]} == 0 ]] || RC=1; done
exit $RC
