#!/bin/bash
# Lane dsv41/rdma: the Qwen3.6-35B-A3B NVFP4 at TP=2 over both Sparks (`tp-worker --serving` on spark2, `serve --tp-remote` on
# spark1, the model's own recipe, NCCL), for two binaries: the same greedy ids and the same failure behaviour?
#   SPARK_DIR=~/intarsia-dsv41-rdma BIN_MAIN=~/intarsia-dsv41-rdma-main/bin/intarsia BIN_HEAD=~/intarsia-dsv41-rdma/bin/intarsia \
#     fleetq run --host spark1,spark2 --load gpu --mem 30 --expect 26 --label "35B tp2 compare" -- scripts/dsv41-tp2/tp35-compare.sh
# Per binary (ARMS, default "main head"):
#   ids    256 greedy ids (/v1/completions, temperature 0, ignore_eos, return_token_ids) at a 128-token and a 4k-token prompt
#   f1     SIGKILL the worker (rank 1) 4 s into an 800-token reply: HTTP result of the request, /health every second, both exit codes
#   again  restart both ranks: the short prompt must give the same ids as before the fault
#   f2     SIGKILL the primary (rank 0) 4 s into an 800-token reply: the worker's fate and exit code
#   after  restart both ranks after f2: both prompts must give the same ids as before the fault
#   f3/f4  SIGTERM / SIGKILL the primary with no request running: the worker's fate and exit code
# vLLM stays up (the 35B is about 11 GB a rank). Output in $OUT (default .scratch/tp35c-HHMM). Processes are started under a bash
# wrapper that writes the exit code of the rank to a file; they are killed by PID only (the rank is the wrapper's child).
# EXTRA_ENV: more environment for both ranks. WORKER_GDB=1: the worker runs under gdb (ptrace_scope 1 allows a debugger only on its
# own child) and, 10 s after a fault, gets a SIGINT that makes gdb write every thread's stack to the worker log. SERVE_GDB=1: the
# same for the primary; gdb then kills it, so a request still open at 10 s ends with curl rc 52 and an answered one with its HTTP code.
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
set -u
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$REPO/.scratch/tp35c-$(date +%H%M)}; mkdir -p "$OUT"
ARMS=${ARMS:-"main head"}
BIN_MAIN=${BIN_MAIN:?path of the baseline binary on the Sparks}; BIN_HEAD=${BIN_HEAD:-$SPARK_DIR/bin/intarsia}
LOGD="$SPARK_DIR/logs"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=${WPORT:-29712}; SPORT=${SPORT:-8908}; SURL=http://100.120.88.104:$SPORT
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN ${EXTRA_ENV:-}"
FL="-m $M35 --backend cuda --tp 2 --max-seq 8192 --mem-budget 20"
P=${PROMPTS:-/opt/code/deepeval/tests/data/prompts/natural-v1}
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

# start_ranks <tag> <binary>: wrappers $LOGD/<tag>-{worker,serve}.{pid,exit}; the ranks' own PIDs are the wrappers' children
start_ranks() {
  TAG=$1; BIN=$2
  local GDB=""; [[ ${WORKER_GDB:-0} == 1 ]] && GDB="gdb -q -batch -ex \"handle all nostop noprint pass\" -ex run -ex \"thread apply all bt 30\" -ex kill --args"
  r spark2 "mkdir -p $LOGD; rm -f $LOGD/$TAG-worker.exit; nohup bash -c 'env $NCCLENV $GDB $BIN tp-worker --serving --listen $WIP:$WPORT $FL; echo \$? > $LOGD/$TAG-worker.exit' > $LOGD/$TAG-worker.log 2>&1 < /dev/null & echo \$! > $LOGD/$TAG-worker.pid"
  sleep 2
  local SGDB=""; [[ ${SERVE_GDB:-0} == 1 ]] && SGDB="gdb -q -batch -ex \"handle all nostop noprint pass\" -ex run -ex \"thread apply all bt 30\" -ex kill --args"
  r spark1 "mkdir -p $LOGD; rm -f $LOGD/$TAG-serve.exit; nohup bash -c 'env $NCCLENV $SGDB $BIN serve $FL --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name q35 --no-supervisor --no-ui; echo \$? > $LOGD/$TAG-serve.exit' > $LOGD/$TAG-serve.log 2>&1 < /dev/null & echo \$! > $LOGD/$TAG-serve.pid"
  local T0 up=0
  T0=$(date +%s)
  while (( $(date +%s) - T0 < 420 )); do
    curl -sf -m 5 "$SURL/health" > /dev/null && { up=1; break; }
    r spark1 "kill -0 \$(cat $LOGD/$TAG-serve.pid)" 2> /dev/null || { log "rank 0 exited while loading"; break; }
    r spark2 "kill -0 \$(cat $LOGD/$TAG-worker.pid)" 2> /dev/null || { log "rank 1 exited while loading"; break; }
    sleep 5
  done
  log "[$TAG] serve up=$up after $(( $(date +%s) - T0 )) s"
  [[ $up == 1 ]]
}
rank_pid() { # <host> <wrapper pid file> -> the rank's own PID (the wrapper's child), or nothing
  r "$1" "w=\$(cat $2 2>/dev/null); [ -n \"\$w\" ] && pgrep -P \$w | head -1"
}
# kill_all: leftovers by PID (rank first, then its wrapper)
kill_all() {
  local h f p w
  for hf in "spark1 $TAG-serve" "spark2 $TAG-worker"; do
    h=${hf% *}; f=${hf#* }
    p=$(rank_pid "$h" "$LOGD/$f.pid"); w=$(r "$h" "cat $LOGD/$f.pid 2>/dev/null")
    [[ -n $p ]] && r "$h" "kill -KILL $p 2>/dev/null; true"
    [[ -n $w ]] && r "$h" "kill -KILL $w 2>/dev/null; true"
  done
  sleep 3
}
exitcode() { r "$1" "cat $LOGD/$2.exit 2>/dev/null || echo none"; }
completion() { # <name> <prompt file> -> ids in $OUT/<name>.ids
  python3 -c '
import json, sys
print(json.dumps({"model": "q35", "prompt": open(sys.argv[1]).read(), "temperature": 0, "max_tokens": 256, "ignore_eos": True, "return_token_ids": True}))' "$2" > "$OUT/$1.req"
  curl -s -m 300 "$SURL/v1/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2> "$OUT/$1.curlerr"
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1])); c = d["choices"][0]; print(" ".join(str(i) for i in c["token_ids"]))
except Exception as e:
    print("<no ids: %s>" % type(e).__name__)' "$OUT/$1.resp" > "$OUT/$1.ids"
  log "[$TAG] $1: $(wc -w < "$OUT/$1.ids") ids, first '$(cut -d' ' -f1-6 "$OUT/$1.ids")'"
}
long_request() { # <name> -> background; result in $OUT/<name>.resp, times in $OUT/<name>.times
  echo '{"model":"q35","messages":[{"role":"user","content":"Write a very long and detailed story about a lighthouse keeper, a storm, a message in a bottle and a ship that never arrived. Use many paragraphs."}],"temperature":0,"max_tokens":800}' > "$OUT/$1.req"
  ( s=$(now_ms); curl -s -m 120 -w '\nHTTP %{http_code}\n' "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2> "$OUT/$1.curlerr"; rc=$?; echo "$s $(now_ms) $rc" > "$OUT/$1.times" ) &
  REQ_PID=$!
}
observe() { # <name> <seconds>: ms since the fault, /health code, both PIDs
  local end=$(( $(date +%s) + $2 )) h a0 a1 snapped=
  : > "$OUT/$1.timeline"
  while (( $(date +%s) < end )); do
    h=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health" 2> /dev/null); [[ -n $h ]] || h=000
    a0=$(r spark1 "kill -0 $P0 2>/dev/null && echo up || echo gone")
    a1=$(r spark2 "kill -0 $P1 2>/dev/null && echo up || echo gone")
    echo "$(( $(now_ms) - TF )) ms health=$h rank0=$a0 rank1=$a1" >> "$OUT/$1.timeline"
    # once, 10 s after the fault: what every thread of a surviving rank is doing (name, state, kernel wait channel) and the control socket
    if [[ -z $snapped ]] && (( $(now_ms) - TF > 10000 )); then
      snapped=1
      [[ $a0 == up ]] && r spark1 "for t in /proc/$P0/task/*; do echo \"\$(cat \$t/comm) \$(awk '{print \$3}' \$t/stat) \$(cat \$t/wchan 2>/dev/null)\"; done; ss -tn | grep $WPORT" > "$OUT/$1.threads0" 2>&1
      [[ $a1 == up ]] && r spark2 "for t in /proc/$P1/task/*; do echo \"\$(cat \$t/comm) \$(awk '{print \$3}' \$t/stat) \$(cat \$t/wchan 2>/dev/null)\"; done; ss -tn | grep $WPORT" > "$OUT/$1.threads1" 2>&1
      [[ $a1 == up && ${WORKER_GDB:-0} == 1 ]] && r spark2 "kill -INT $P1"
      [[ $a0 == up && ${SERVE_GDB:-0} == 1 ]] && r spark1 "kill -INT $P0"
    fi
    [[ $a0 == gone && $a1 == gone ]] && break
    sleep 0.5
  done
}
first() { awk -v f="$2" '$0 ~ f {print $1; exit}' "$1"; }
fault() { # <case> <victim: worker|serve>
  local c=$1 v=$2
  long_request "$c"
  sleep 4
  P0=$(rank_pid spark1 "$LOGD/$TAG-serve.pid"); P1=$(rank_pid spark2 "$LOGD/$TAG-worker.pid")
  [[ ${WORKER_GDB:-0} == 1 ]] && P1=$(r spark2 "pgrep -P $P1 | head -1")
  [[ ${SERVE_GDB:-0} == 1 ]] && P0=$(r spark1 "pgrep -P $P0 | head -1")
  log "[$TAG] $c: ranks' PIDs rank 0 $P0 (spark1), rank 1 $P1 (spark2)"
  TF=$(now_ms)
  if [[ $v == worker ]]; then r spark2 "kill -KILL $P1"; else r spark1 "kill -KILL $P0"; fi
  observe "$c" 60
  wait $REQ_PID
  local times st en rc
  times=$(cat "$OUT/$c.times" 2> /dev/null); st=${times%% *}; en=${times#* }; rc=${en#* }; en=${en%% *}
  log "[$TAG] $c RESULT: killed $v; request ended $(( en - st - 4000 )) ms after the kill (curl rc $rc, $(grep -a '^HTTP' "$OUT/$c.resp" | tail -1)); /health first 503 at $(first "$OUT/$c.timeline" 'health=503' || true) ms (never = empty); /health first 000 at $(first "$OUT/$c.timeline" 'health=000') ms; rank 0 gone at $(first "$OUT/$c.timeline" 'rank0=gone') ms; rank 1 gone at $(first "$OUT/$c.timeline" 'rank1=gone') ms"
  log "[$TAG] $c response head: $(head -c 240 "$OUT/$c.resp" | tr '\n' ' ')"
  sleep 3
  log "[$TAG] $c exit codes: rank 0 (serve) $(exitcode spark1 "$TAG-serve"), rank 1 (worker) $(exitcode spark2 "$TAG-worker")"
  r spark1 "grep -aiE 'tensor parallel|tp-worker|watchdog|failed|abort|exiting|panick' $LOGD/$TAG-serve.log | tail -8 | cut -c1-240" | tee -a "$OUT/steps.log"
  r spark2 "grep -aiE 'tensor parallel|tp-worker|watchdog|failed|abort|exiting|panick' $LOGD/$TAG-worker.log | tail -8 | cut -c1-240" | tee -a "$OUT/steps.log"
  cp_logs "$c"
  kill_all
}
idle_fault() { # <case> <signal>: the primary is signalled with no request running
  local c=$1
  P0=$(rank_pid spark1 "$LOGD/$TAG-serve.pid"); P1=$(rank_pid spark2 "$LOGD/$TAG-worker.pid")
  TF=$(now_ms)
  r spark1 "kill -$2 $P0"
  observe "$c" 30
  log "[$TAG] $c RESULT: SIG$2 to the idle primary; rank 0 gone at $(first "$OUT/$c.timeline" 'rank0=gone') ms; rank 1 gone at $(first "$OUT/$c.timeline" 'rank1=gone') ms"
  sleep 3
  log "[$TAG] $c exit codes: rank 0 (serve) $(exitcode spark1 "$TAG-serve"), rank 1 (worker) $(exitcode spark2 "$TAG-worker")"
  r spark2 "grep -aiE 'tensor parallel|tp-worker' $LOGD/$TAG-worker.log | tail -4 | cut -c1-240" | tee -a "$OUT/steps.log"
  cp_logs "$c"
  kill_all
}
cp_logs() { scp -q "spark1:$SPARK_REL/logs/$TAG-serve.log" "$OUT/$TAG-$1.serve.log" 2> /dev/null; scp -q "spark2:$SPARK_REL/logs/$TAG-worker.log" "$OUT/$TAG-$1.worker.log" 2> /dev/null; }
trap 'TAG=${TAG:-x}; kill_all' EXIT

STAGES=${STAGES:-"ids f1 again f2 after f3 f4"}   # a subset to repeat one fault: STAGES=f2
has() { [[ " $STAGES " == *" $1 "* ]]; }
for arm in $ARMS; do
  [[ $arm == main ]] && B=$BIN_MAIN || B=$BIN_HEAD
  log "=== arm $arm: binary $B md5 $(r spark1 "md5sum $B | cut -c1-12") stamp '$(r spark1 "cat $(dirname $B)/STAMP 2>/dev/null")'"
  if has ids || has f1; then
    start_ranks "$arm" "$B" || { log "[$arm] did not come up"; cp_logs start; kill_all; continue; }
    log "[$arm] startup lines: $(r spark1 "grep -aiE 'joining NCCL|RDMA|rdma|speculative|spec ' $LOGD/$arm-serve.log | head -4 | cut -c1-200 | tr '\n' '|'")"
    log "[$arm] worker lines: $(r spark2 "grep -aiE 'joined NCCL|RDMA|rdma' $LOGD/$arm-worker.log | head -3 | cut -c1-200 | tr '\n' '|'")"
    if has ids; then
      completion "$arm-n128" "$P/n128.txt"
      completion "$arm-n4096" "$P/n4096.txt"
      cp_logs ids
    fi
    if has f1; then fault "$arm-f1" worker; else kill_all; fi
  fi
  if has again; then
    if start_ranks "$arm" "$B"; then
      completion "$arm-n128-again" "$P/n128.txt"
      cmp -s "$OUT/$arm-n128.ids" "$OUT/$arm-n128-again.ids" && log "[$arm] AFTER RESTART: n128 ids identical to the run before the fault" || log "[$arm] AFTER RESTART: n128 ids DIFFER"
      cp_logs again
      kill_all
    else log "[$arm] restart after f1 did not come up"; cp_logs restart; kill_all; fi
  fi
  if has f2; then
    if start_ranks "$arm" "$B"; then fault "$arm-f2" serve; else log "[$arm] start before f2 did not come up"; kill_all; fi
  fi
  if has after; then
    if start_ranks "$arm" "$B"; then
      for k in n128 n4096; do
        completion "$arm-$k-after" "$P/$k.txt"
        cmp -s "$OUT/$arm-$k.ids" "$OUT/$arm-$k-after.ids" && log "[$arm] AFTER PRIMARY KILL: $k ids identical to the run before the fault" || log "[$arm] AFTER PRIMARY KILL: $k ids DIFFER"
      done
      cp_logs after
      kill_all
    else log "[$arm] restart after f2 did not come up"; cp_logs restart; kill_all; fi
  fi
  for c in "f3 TERM" "f4 KILL"; do
    if has "${c% *}"; then
      if start_ranks "$arm" "$B"; then
        completion "$arm-${c% *}-warm" "$P/n128.txt"
        idle_fault "$arm-${c% *}" "${c#* }"
      else log "[$arm] start before ${c% *} did not come up"; kill_all; fi
    fi
  done
done
# head vs main
for k in n128 n4096; do
  if [[ -s $OUT/main-$k.ids && -s $OUT/head-$k.ids ]]; then
    if cmp -s "$OUT/main-$k.ids" "$OUT/head-$k.ids"; then log "COMPARE $k: $(wc -w < "$OUT/head-$k.ids") ids IDENTICAL head vs main"
    else log "COMPARE $k: DIFFER, first differing position: $(python3 -c '
import sys
a=open(sys.argv[1]).read().split(); b=open(sys.argv[2]).read().split()
i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
print(i, "main", a[i:i+1], "head", b[i:i+1], "lengths", len(a), len(b))' "$OUT/main-$k.ids" "$OUT/head-$k.ids")"; fi
  fi
done
log "tp35-compare done"
