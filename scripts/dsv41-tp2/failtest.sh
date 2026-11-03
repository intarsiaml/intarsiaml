#!/bin/bash
# Lane dsv41/rdma: failure tests of the TP=2 pair with the RDMA ring, run inside a run-full.sh window (vLLM taken down and
# restored by that script):
#   HOOK=scripts/dsv41-tp2/failtest.sh CASES="c0 a1 a2" OUT=... SPARK_DIR=... scripts/dsv41-tp2/run-full.sh      (as one fleetq job)
# Every case: fresh start of both ranks (so it also checks that a restarted pair serves) -> golden requests compared with the
# clean baseline (case c0) -> a request in flight -> the fault -> a timeline of /health and of both PIDs every ~0.4 s -> the
# ranks' logs -> leftovers killed by PID -> wait for the memory to come back and a GPU check -> next case.
# Cases:
#   c0  clean run, no fault: the baseline texts (a long decode, three short prompts)
#   a1  SIGKILL rank 1 (spark2 tp-worker) mid-decode          a2  SIGKILL rank 0 (spark1 serve) mid-decode
#   a3  SIGKILL rank 0 during a long prefill (NCCL leg)       a4  SIGKILL rank 1 during a long prefill
#   b1  SIGSTOP rank 1 for 10 s (< the deadline), SIGCONT: the request must finish with identical text
#   b2  SIGSTOP rank 1 for 45 s (> the deadline of 30 s), SIGCONT
#   c1  fault hook on rank 1: queue pair of rail 0 to ERR after its 5000th send     c2  same on rank 0     c3  rail 1 of rank 1
#   d1  one rail named (NCCL_IB_HCA=rocep1s0f1) on both      d2  a bad HCA name on both     d3  INTARSIA_DSV41_RDMA=0 on rank 1 only
#   d4  INTARSIA_DSV41_RDMA=1 on rank 0, 0 on rank 1 (an error on both)      d5  three HCA names      d6  memlock 4 MiB (ulimit -l)
#   d7  INTARSIA_DSV41_RDMA=1 with one rail named (required: one rail is allowed)    d8  fault hook "nolib" on both (no libibverbs)
#   d9  fault hook "ping" on rank 1 (host ping fails after connect)
# FT_DEADLINE (default 20 for a, c, d) and FT_DEADLINE_B (30 for b) -> INTARSIA_DSV41_RDMA_DEADLINE_S.
# No system, network or driver setting is touched: faults are signals to our own PIDs and the test hook of the ring.

PRE_MEM=$(r spark1 "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo")
PRE_MEM2=$(r spark2 "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo")
log "failtest: cases '${CASES:-}'; MemAvailable before the first load: spark1 $PRE_MEM GiB, spark2 $PRE_MEM2 GiB"
now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }
P_LONG="Write a very long and detailed story about a lighthouse keeper, a storm, a message in a bottle and a ship that never arrived. Use many paragraphs."
P_CAP="What is the capital of France? Answer in one sentence."
P_CNT="Count from 1 to 20, separated by commas."
P_CODE="Write a Python function that returns the n-th Fibonacci number iteratively."
# a prompt of about 12000 tokens (pseudo-random words, fixed seed), then a short instruction
python3 - "$OUT/prefill-prompt.txt" <<'PY'
import random, sys
random.seed(7)
w = "time person year way day thing man world life hand part child eye woman place work week case point government company number group problem fact river stone engine garden letter window forest bridge market castle harbor lantern valley meadow thunder copper velvet marble".split()
open(sys.argv[1], "w").write(" ".join(random.choice(w) for _ in range(9000)) + "\n\nSummarize the text above in one short sentence.")
PY

req_json() { # <max_tokens> <prompt or @file> -> json on stdout
  python3 -c '
import json, sys
p = sys.argv[2]
if p.startswith("@"): p = open(p[1:]).read()
print(json.dumps({"model": "dsv41-intarsia", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": int(sys.argv[1])}))' "$1" "$2"
}
text_of() { python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1])); print((d["choices"][0]["message"]["content"]))
except Exception as e:
    print("<no text: %s>" % type(e).__name__)' "$1"; }
# synchronous request -> $OUT/$1.resp, text in $OUT/$1.txt
chat1() { # <name> <max_tokens> <prompt or @file>
  req_json "$2" "$3" > "$OUT/$1.req"
  curl -s -m 300 "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2>"$OUT/$1.curlerr"
  text_of "$OUT/$1.resp" > "$OUT/$1.txt"
}
# background request: times (start end curl-exit) in $OUT/$1.times
req_bg() { # <name> <max_tokens> <prompt or @file>
  req_json "$2" "$3" > "$OUT/$1.req"
  ( s=$(now_ms); curl -s -m 300 -w '\nHTTP %{http_code}\n' "$SURL/v1/chat/completions" -H 'content-type: application/json' -d @"$OUT/$1.req" > "$OUT/$1.resp" 2>"$OUT/$1.curlerr"; rc=$?; echo "$s $(now_ms) $rc" > "$OUT/$1.times" ) &
  REQ_PID=$!
}
same_as_base() { # <name> <base name>
  if cmp -s "$OUT0/$2.txt" "$OUT/$1.txt"; then echo identical; else echo "DIFFERENT ($(head -c 80 "$OUT/$1.txt" | tr '\n' ' '))"; fi
}
pids() {
  P0=$(r spark1 "pgrep -f '[i]${SPARK_REL:1}/bin/intarsia[^ ]* serve' | head -1")
  P1=$(r spark2 "pgrep -f '[i]${SPARK_REL:1}/bin/intarsia[^ ]* tp-worker' | head -1")
  log "pids: rank 0 (spark1) ${P0:-none}, rank 1 (spark2) ${P1:-none}"
}
# timeline: elapsed ms since $TF, /health code, both PIDs; ends when both are gone or after $1 s
observe() { # <name> <seconds>
  local end=$(( $(date +%s) + $2 )) h a0 a1
  : > "$OUT/$1.timeline"
  while (( $(date +%s) < end )); do
    h=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$SURL/health" 2>/dev/null); [[ -n $h ]] || h=000
    a0=$(r spark1 "kill -0 ${P0:-0} 2>/dev/null && echo up || echo gone")
    a1=$(r spark2 "kill -0 ${P1:-0} 2>/dev/null && echo up || echo gone")
    echo "$(( $(now_ms) - TF )) ms health=$h rank0=$a0 rank1=$a1" >> "$OUT/$1.timeline"
    [[ $a0 == gone && $a1 == gone ]] && break
    sleep 0.3
  done
}
firstgone() { # <timeline> <field> -> ms of the first line where the field says gone
  awk -v f="$2" '$0 ~ f"=gone" {print $1; exit}' "$1"
}
logs() { # <name>: the tails of both ranks' logs
  r spark1 "tail -n 80 $LOGD/serve-$TAG.log" > "$OUT/$1.serve.log" 2>&1
  r spark2 "tail -n 80 $LOGD/worker-$TAG.log" > "$OUT/$1.worker.log" 2>&1
}
cleanup() { # kill leftovers by PID, wait for the memory, GPU check
  local left=""
  [[ -n ${P0:-} ]] && r spark1 "kill -0 $P0 2>/dev/null" && { left="$left rank0($P0)"; r spark1 "kill -KILL $P0"; }
  [[ -n ${P1:-} ]] && r spark2 "kill -0 $P1 2>/dev/null" && { left="$left rank1($P1)"; r spark2 "kill -KILL $P1"; }
  log "leftover processes killed by PID after the case:${left:- none}"
  local i m1 m2
  for i in $(seq 1 40); do
    m1=$(r spark1 "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo"); m2=$(r spark2 "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo")
    (( m1 >= PRE_MEM - 6 && m2 >= PRE_MEM2 - 6 )) && break
    sleep 3
  done
  log "memory back after $(( i * 3 )) s: spark1 $m1 GiB (was $PRE_MEM), spark2 $m2 GiB (was $PRE_MEM2)"
  for h in spark1 spark2; do
    log "gpu check $h: $(r $h "timeout 25 nvidia-smi --query-gpu=name,utilization.gpu --format=csv,noheader 2>&1 | head -2 | tr '\n' ' '; echo \"rc=\${PIPESTATUS[0]}\"; timeout 25 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | head -3 | tr '\n' ' '")"
  done
}
# fresh start with the case's environment, then the golden requests against the baseline
begin() { # <name> <env both ranks>   (R0ENV/R1ENV/R0PRE/R1PRE set by the caller)
  CASE=$1; OUT=$OUT0/$1; mkdir -p "$OUT"
  log "=== case $CASE: env '$2' rank0 '${R0ENV:-}' rank1 '${R1ENV:-}' pre0 '${R0PRE:-}' pre1 '${R1PRE:-}'"
  start_ranks "$2"
  pids
  if [[ $CASE != c0 ]]; then
    chat1 g-cap 48 "$P_CAP"; chat1 g-cnt 64 "$P_CNT"
    log "after a (re)start: capital $(same_as_base g-cap g-cap), count $(same_as_base g-cnt g-cnt)"
  fi
}
finish() { # <name>
  logs "$1"
  log "case $1 ranks' log lines (rank 0):"; grep -aiE "rdma|failed|abort|watchdog|trap|error|disconnect|exiting|group" "$OUT/$1.serve.log" | tail -12 | cut -c1-260 | tee -a "$OUT0/steps.log"
  log "case $1 ranks' log lines (rank 1):"; grep -aiE "rdma|failed|abort|trap|error|disconnect|exiting|group" "$OUT/$1.worker.log" | tail -12 | cut -c1-260 | tee -a "$OUT0/steps.log"
  cleanup
  R0ENV=""; R1ENV=""; R0PRE=""; R1PRE=""
}
verdict() { # <name>: one summary line from the timeline and the request
  local t0 t1 h503 times st en rc
  t0=$(firstgone "$OUT/$1.timeline" rank0); t1=$(firstgone "$OUT/$1.timeline" rank1)
  h503=$(awk '/health=503/ {print $1; exit}' "$OUT/$1.timeline")
  times=$(cat "$OUT/$1.times" 2>/dev/null); st=${times%% *}; en=${times#* }; rc=${en#* }; en=${en%% *}
  log "RESULT $1: fault at $TF-relative 0 ms; request ended $(( ${en:-0} - ${st:-0} - ${REQ_FAULT_AFTER_MS:-0} )) ms after the fault (curl rc ${rc:-?}, $(grep -a HTTP "$OUT/$1.resp" | tail -1)); /health first 503 at ${h503:-never} ms; rank 0 gone at ${t0:-never} ms; rank 1 gone at ${t1:-never} ms"
  log "RESULT $1 response head: $(head -c 300 "$OUT/$1.resp" | tr '\n' ' ')"
}

OUT0=$OUT
DL=${FT_DEADLINE:-20}
run_case() {
  case $1 in
  c0)
    begin c0 "INTARSIA_DSV41_RDMA_DEADLINE_S=$DL"
    chat1 g-cap 48 "$P_CAP"; chat1 g-cnt 64 "$P_CNT"; chat1 g-code 200 "$P_CODE"; chat1 g-long 400 "$P_LONG"
    cp "$OUT"/g-*.txt "$OUT0/" 2>/dev/null
    log "baseline: capital '$(head -c 100 "$OUT/g-cap.txt" | tr '\n' ' ')' long ($(wc -c < "$OUT/g-long.txt") bytes)"
    grep -a "over RDMA\|RDMA exchange not used" "$OUT/../c0/"*.log 2>/dev/null | head -2
    r spark1 "grep -a 'RDMA' $LOGD/serve-$TAG.log | head -3" | tee -a "$OUT0/steps.log"
    finish c0 ;;
  a1|a2|a3|a4)
    begin $1 "INTARSIA_DSV41_RDMA_DEADLINE_S=$DL"
    if [[ $1 == a3 || $1 == a4 ]]; then req_bg $1 16 "@$OUT0/prefill-prompt.txt"; REQ_FAULT_AFTER_MS=2500; sleep 2.5
    else req_bg $1 800 "$P_LONG"; REQ_FAULT_AFTER_MS=5000; sleep 5; fi
    TF=$(now_ms)
    case $1 in
      a1|a4) log "SIGKILL rank 1 (pid $P1)"; r spark2 "kill -KILL $P1" ;;
      a2|a3) log "SIGKILL rank 0 (pid $P0)"; r spark1 "kill -KILL $P0" ;;
    esac
    observe $1 75
    wait $REQ_PID; verdict $1
    finish $1 ;;
  b1|b2)
    DLB=${FT_DEADLINE_B:-30}
    begin $1 "INTARSIA_DSV41_RDMA_DEADLINE_S=$DLB"
    req_bg $1 400 "$P_LONG"; REQ_FAULT_AFTER_MS=4000; sleep 4
    TF=$(now_ms)
    secs=10; [[ $1 == b2 ]] && secs=45
    log "SIGSTOP rank 1 (pid $P1) for $secs s"
    ( r spark2 "kill -STOP $P1"; sleep $secs; r spark2 "kill -CONT $P1 2>/dev/null; true" ) &
    observe $1 $(( secs + 60 ))
    wait $REQ_PID; verdict $1
    [[ $1 == b1 ]] && log "RESULT b1 text vs the clean run: $(same_as_base $1 g-long)"
    finish $1 ;;
  c1|c2|c3)
    case $1 in
      c1) R1ENV="INTARSIA_DSV41_RDMA_FAULT=qp-err:5000" ;;
      c2) R0ENV="INTARSIA_DSV41_RDMA_FAULT=qp-err:5000" ;;
      c3) R1ENV="INTARSIA_DSV41_RDMA_FAULT=qp-err:5000:1" ;;
    esac
    begin $1 "INTARSIA_DSV41_RDMA_DEADLINE_S=$DL"
    req_bg $1 800 "$P_LONG"; REQ_FAULT_AFTER_MS=0; TF=$(now_ms)
    observe $1 75
    wait $REQ_PID; verdict $1
    finish $1 ;;
  d1|d2|d3|d4|d5|d6|d7|d8|d9)
    local denv="INTARSIA_DSV41_RDMA_DEADLINE_S=$DL"
    case $1 in
      d1) denv="$denv NCCL_IB_HCA=rocep1s0f1" ;;
      d2) denv="$denv NCCL_IB_HCA=rocepNOSUCH0,roceNOSUCH1" ;;
      d3) R1ENV="INTARSIA_DSV41_RDMA=0" ;;
      d4) R0ENV="INTARSIA_DSV41_RDMA=1"; R1ENV="INTARSIA_DSV41_RDMA=0" ;;
      d5) denv="$denv NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1,rocep1s0f0" ;;
      d6) R0PRE="ulimit -l 4096; "; R1PRE="ulimit -l 4096; " ;;
      d7) denv="$denv NCCL_IB_HCA=rocep1s0f1 INTARSIA_DSV41_RDMA=1" ;;
      d8) denv="$denv INTARSIA_DSV41_RDMA_FAULT=nolib" ;;
      d9) R1ENV="INTARSIA_DSV41_RDMA_FAULT=ping" ;;
    esac
    CASE=$1; OUT=$OUT0/$1; mkdir -p "$OUT"
    log "=== case $1: env '$denv' rank0 '${R0ENV:-}' rank1 '${R1ENV:-}' pre '${R0PRE:-}'"
    if [[ $1 == d4 ]]; then
      # the ranks refuse to pair: both processes must report it and exit, nothing hangs
      TAG=$(date +%H%M%S)
      r spark2 "mkdir -p $LOGD; ${R1PRE:-}nohup env $NCCLENV $DENV $denv ${R1ENV:-} $BIN tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/worker-$TAG.log 2>&1 < /dev/null &"
      sleep 2
      r spark1 "mkdir -p $LOGD; ${R0PRE:-}nohup env $NCCLENV $DENV $denv ${R0ENV:-} $BIN serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/serve-$TAG.log 2>&1 < /dev/null &"
      T0=$(date +%s)
      for i in $(seq 1 60); do
        a=$(r spark1 "pgrep -f '[i]${SPARK_REL:1}/bin/intarsia[^ ]* serve' >/dev/null && echo up || echo gone"); b=$(r spark2 "pgrep -f '[i]${SPARK_REL:1}/bin/intarsia[^ ]* tp-worker' >/dev/null && echo up || echo gone")
        [[ $a == gone && $b == gone ]] && break; sleep 2
      done
      log "RESULT d4: rank 0 $a, rank 1 $b after $(( $(date +%s) - T0 )) s"
      pids; logs d4
      grep -aiE "RDMA|set it the same" "$OUT/d4.serve.log" | head -3 | cut -c1-250 | tee -a "$OUT0/steps.log"
      grep -aiE "RDMA|set it the same" "$OUT/d4.worker.log" | head -3 | cut -c1-250 | tee -a "$OUT0/steps.log"
      cleanup; R0ENV=""; R1ENV=""; return
    fi
    start_ranks "$denv"; pids
    chat1 g-cap 48 "$P_CAP"; chat1 g-cnt 64 "$P_CNT"; chat1 g-long 400 "$P_LONG"
    log "RESULT $1: capital $(same_as_base g-cap g-cap), count $(same_as_base g-cnt g-cnt), long $(same_as_base g-long g-long)"
    r spark1 "grep -aiE 'RDMA' $LOGD/serve-$TAG.log | head -4" | cut -c1-300 | tee -a "$OUT0/steps.log"
    r spark2 "grep -aiE 'RDMA' $LOGD/worker-$TAG.log | head -4" | cut -c1-300 | tee -a "$OUT0/steps.log"
    # which path ran: decode speed of the long request (RDMA is about 6 % faster)
    finish $1 ;;
  *) log "unknown case $1" ;;
  esac
}
for c in ${CASES:-c0}; do
  [[ $(left) -lt 240 ]] && { log "out of time before case $c"; break; }
  run_case $c
done
OUT=$OUT0
