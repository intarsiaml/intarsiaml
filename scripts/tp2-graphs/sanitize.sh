#!/usr/bin/env bash
# Lane G1c follow-up: compute-sanitizer (memcheck) on BOTH ranks of the 35B at tp=2 across the Sparks, with the decode
# graph on (CUDA API errors not reported: NCCL init probes kernel images and hits error 209 by design), through a request sequence that would replay a graph into a moved buffer: a short request (prefill of a few
# rows, then the decode graph is captured), a long one (prefill of ~700 rows: grows any buffer sized on demand), and a
# short one again (replays the graph captured in the first request). With the NCCL f32 landing buffer reserved at load
# (`Comm::reserve`) nothing moves; the `no-reserve` mutant (scripts/tp2-graphs/mutate.py) frees and reallocates it.
# Runs HERE (the Mac) as ONE fleetq job over both Sparks, vLLM up:
#   fleetq run --host spark1,spark2 --mem spark1=30,spark2=20 --label "..." -- scripts/tp2-graphs/sanitize.sh [tree...]
# Trees: binaries ~/intarsia-tp2/bin/g1c-<tree>-intarsia on both boxes (scripts/tp2-graphs/build.sh); default
# "mut-no-reserve dev". Per tree: the memcheck logs of both ranks and the three responses, and one line
#   SANITIZE <tree>: rank0 <n> errors, rank1 <n> errors   (memcheck's ERROR SUMMARY; a clean tree has 0 on both)
# Result 2026-10-21 (raw/sanitize): main 1ab83b69 0 / 0; no-reserve 512 / 512, all "Invalid __global__ write" by
# ncclDevKernel_AllReduce_Sum_f32 just past a freed 64 KB allocation (the replayed graph writes the moved buffer).
# Output: docs/results/2026-10-21-tp2-cuda-graphs/raw/$OUT.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-sanitize-$(date +%H%M)}
L=$REPO/docs/results/2026-10-21-tp2-cuda-graphs/raw/$OUT; mkdir -p "$L"
TREES=${*:-mut-no-reserve dev}
RUN_SEC=${RUN_SEC:-360}   # per tree: load + three requests under memcheck
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29556; PORT=8093
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN INTARSIA_TP_REMOTE_TIMEOUT_S=900"
FLAGS="-m $M35 --backend cuda --tp 2 --max-seq 4096 --mem-budget 16"
SAN="/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check no --report-api-errors no --print-limit 30 --error-exitcode 0"
XO='$HOME/intarsia-tp2/xout/'$OUT
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
# stop the target (not the sanitizer) first, so memcheck sees it exit and writes its ERROR SUMMARY
kill_all() {
  for h in spark1 spark2; do
    r $h "pkill -TERM -f '^/home/nvidia/intarsia-tp2/bin/g1c-'; for i in \$(seq 1 20); do pgrep -f '[c]ompute-sanitizer.*g1c-' >/dev/null || break; sleep 1; done; pkill -KILL -f '[c]ompute-sanitizer.*g1c-'; pkill -KILL -f '[i]ntarsia-tp2/bin/g1c-'; true"
  done
}
finish() {
  local rc=$?
  kill_all
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-tp2/xout/$OUT/" "$L/spark2/" 2>/dev/null
  for t in $TREES; do
    n0=$(grep -ohE 'ERROR SUMMARY: [0-9]+ errors$' "$L/san-r0-$t.log" 2>/dev/null | grep -oE '[0-9]+')
    n1=$(grep -ohE 'ERROR SUMMARY: [0-9]+ errors$' "$L/spark2/san-r1-$t.log" 2>/dev/null | grep -oE '[0-9]+')
    k=$(grep -hE '^========= Invalid' "$L/san-r0-$t.log" "$L/spark2/san-r1-$t.log" 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    log "SANITIZE $t: rank0 ${n0:-?} errors, rank1 ${n1:-?} errors${k:+ (printed: $k)}"
  done
  trap - EXIT INT TERM; exit $rc
}
trap finish EXIT INT TERM
for h in spark1 spark2; do r $h "mkdir -p $XO"; done
r spark1 "python3 - > $XO/req-long.json <<'PY'
import json, os
t = open(os.path.expanduser('~/intarsia-tp2/prompts/corpus_docs.txt')).read()[:3000]
print(json.dumps({'model': 'intarsia-35b', 'messages': [{'role': 'user', 'content': t + '\nSummarise in one line.'}], 'max_tokens': 6, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}))
PY
echo '{\"model\":\"intarsia-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Name a colour.\"}],\"max_tokens\":6,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}' > $XO/req-short.json"
for t in $TREES; do
  B="\$HOME/intarsia-tp2/bin/g1c-$t-intarsia"
  log "== $t: $(r spark1 "$B --version | tr '\n' ' '")"
  r spark2 "nohup env $NCCLENV timeout $RUN_SEC $SAN --log-file $XO/san-r1-$t.log $B tp-worker --listen $WIP:$WPORT --serving $FLAGS > $XO/worker-$t.log 2>&1 < /dev/null &"
  sleep 2
  r spark1 "nohup env $NCCLENV timeout $RUN_SEC $SAN --log-file $XO/san-r0-$t.log $B serve $FLAGS --tp-remote $WIP:$WPORT --port $PORT --host 127.0.0.1 --served-model-name intarsia-35b > $XO/serve-$t.log 2>&1 < /dev/null &"
  t0=$(date +%s); up=0
  while (( $(date +%s) - t0 < RUN_SEC - 60 )); do r spark1 "curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null" && { up=1; break; }; sleep 5; done
  log "$t server up=$up after $(( $(date +%s) - t0 )) s"
  if [[ $up == 1 ]]; then
    for q in short long short; do
      r spark1 "curl -s -m $(( RUN_SEC / 3 )) http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' -d @$XO/req-$q.json >> $XO/resp-$t.jsonl; echo >> $XO/resp-$t.jsonl"
      log "$t request $q done ($(( $(date +%s) - t0 )) s)"
    done
  fi
  kill_all; sleep 3
done
log "== sanitize done"
