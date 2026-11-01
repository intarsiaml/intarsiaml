#!/bin/bash
# Lane dsv41/tp2: one full-model TP=2 window (rank 0 = spark1 `serve --tp-remote`, rank 1 = spark2 `tp-worker`, NCCL over
# ConnectX-7), vLLM down for the window and ALWAYS restored. Run as ONE exclusive job over both Sparks:
#   fleetq run --host spark1,spark2 --exclusive --takes-down spark1-vllm,spark2-vllm --mem spark1=116,spark2=116 --expect 90 \
#     --label "dsv41 intarsia TP2 full load; vLLM restored after" -- scripts/dsv41-tp2/run-full.sh
# STEPS (default "greedy"): greedy = 3 chat prompts at temperature 0; speed = decode/prefill timing; hold = keep the server up
# for HOLD_MIN minutes (localeval from outside, in this job). MAX_MIN bounds the work (the vLLM restart comes on top).
set -u
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$REPO/.scratch/full-$(date +%m%d-%H%M)}; mkdir -p "$OUT"
STEPS=${STEPS:-greedy}; MAX_MIN=${MAX_MIN:-60}; DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
BIN='$HOME/intarsia-dsv41-tp2/bin/intarsia'
C0='$HOME/intarsia-dsv41-tp2/cache/r0of2'; C1='$HOME/intarsia-dsv41-tp2/cache/r1of2'
WIP=10.100.152.3; WPORT=29701; SPORT=8899; SURL=http://100.120.88.104:$SPORT
LOGD='$HOME/intarsia-dsv41-tp2/logs'
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}"
DENV="INTARSIA_DSV41_CAP=${CAP:-32768} INTARSIA_DSV41_CHUNK=${CHUNK:-512} INTARSIA_DSV41_ENGRAM=\$HOME/dsv41-engram INTARSIA_DSV41_ENGRAM_HOT_MB=${HOT_MB:-1024} RUST_BACKTRACE=1 ${XENV:-}"
VLLM_C=missioncntrl-guidance
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
left() { echo $(( DEADLINE - $(date +%s) )); }

for h in spark1 spark2; do
  # EARLYOOM_OK: the earlyoom -m settings the user has set on purpose (default 3,2; the user lowered it to 2,1 for the
  # BLACKFOLD runs of 2026-11-01). Never changed here; our own watcher stops us at KILL_GIB, above either kill line.
  eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
  [[ " ${EARLYOOM_OK:-3,2} " == *" $eo "* ]] || { log "earlyoom on $h is at -m $eo, not one of '${EARLYOOM_OK:-3,2}'; not starting"; exit 1; }
  log "earlyoom on $h: -m $eo"
  r $h "test -x ~/intarsia-dsv41-tp2/bin/intarsia && test -f ~/intarsia-nccl/libnccl.so.2" || { log "binary or NCCL library missing on $h"; exit 1; }
done
r spark1 "test -f $C0/manifest.json" && r spark2 "test -f $C1/manifest.json" || { log "rank cache missing"; exit 1; }
[[ "$(r spark1 'cat ~/intarsia-dsv41-tp2/bin/STAMP')" == "$(r spark2 'cat ~/intarsia-dsv41-tp2/bin/STAMP')" ]] || { log "binary stamps differ"; exit 1; }
log "binary $(r spark1 'cat ~/intarsia-dsv41-tp2/bin/STAMP'); out $OUT"

kill_ours() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]ntarsia-dsv41-tp2/bin/intarsia'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-dsv41-tp2/bin/intarsia' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-dsv41-tp2/bin/intarsia'; true"; done; }
STOPPED=0
restore() {
  local rc=$?
  log "restore (exit $rc)"
  kill "${WATCH:-}" 2>/dev/null
  kill_ours
  for h in spark1 spark2; do r $h "mkdir -p $LOGD; cp $LOGD/*.log /dev/null 2>/dev/null; true"; done
  scp -q "spark1:intarsia-dsv41-tp2/logs/serve-*.log" "$OUT/" 2>/dev/null; scp -q "spark2:intarsia-dsv41-tp2/logs/worker-*.log" "$OUT/" 2>/dev/null
  if [[ $STOPPED == 1 ]]; then
    # the container keeps its old log, whose "startup complete" line matched at once (window 2); `docker logs --since`
    # prints nothing on these hosts and log rotation drops old lines, so a count does not work either. Wait for a
    # "startup complete" line stamped (docker logs -t, UTC) after the container's StartedAt.
    for h in spark1 spark2; do r $h "docker start $VLLM_C >/dev/null" || log "WARNING: docker start on $h failed"; done
    for h in spark1 spark2; do
      local ok=0
      for i in $(seq 1 120); do
        r $h "s=\$(docker inspect -f '{{.State.StartedAt}}' $VLLM_C); t=\$(docker logs -t --tail 2000 $VLLM_C 2>&1 | grep 'Application startup complete' | tail -1 | cut -d' ' -f1); [[ -n \$t && \$t > \$s ]]" && { ok=1; break; }
        sleep 10
      done
      [[ $ok == 1 ]] && log "vLLM on $h up ($(r $h "docker ps --filter name=$VLLM_C --format {{.Status}}"))" || log "WARNING: vLLM on $h not up after 20 min"
    done
  fi
  fleetq mark-up spark1-vllm; fleetq mark-up spark2-vllm
  log "restored"
}
trap restore EXIT
for h in spark1 spark2; do r $h "docker stop -t 60 $VLLM_C >/dev/null"; done; STOPPED=1
log "vLLM stopped on both"
# watcher: kill our processes on a box that drops under KILL_GIB (default 5) GiB available (earlyoom -m 3,2 kills other
# users' processes at ~3.65 GiB; coordinator 2026-11-01: stop under 5 GiB)
( while true; do
    line="$(date +%T)"
    for h in spark1 spark2; do
      a=$(ssh -n -o ConnectTimeout=5 $h "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA)
      line="$line $h=$a"
      if [[ "$a" != NA ]] && awk "BEGIN{exit !($a < ${KILL_GIB:-5})}"; then ssh -n $h "pkill -KILL -f '[i]ntarsia-dsv41-tp2/bin/intarsia'"; line="$line KILLED-$h"; fi
    done
    echo "$line"; sleep 5
  done ) > "$OUT/mem.log" 2>&1 & WATCH=$!

TAG=$(date +%H%M)
r spark2 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $BIN tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/worker-$TAG.log 2>&1 < /dev/null &"
sleep 2
r spark1 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $BIN serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/serve-$TAG.log 2>&1 < /dev/null &"
log "ranks started (tag $TAG)"
T0=$(date +%s); up=0
while (( $(date +%s) - T0 < 900 )); do
  curl -sf -m 5 "$SURL/health" >/dev/null && { up=1; break; }
  r spark1 "pgrep -f '[i]ntarsia-dsv41-tp2/bin/intarsia serve' >/dev/null" || { log "rank 0 exited"; break; }
  r spark2 "pgrep -f '[i]ntarsia-dsv41-tp2/bin/intarsia tp-worker' >/dev/null" || { log "rank 1 exited"; break; }
  sleep 5
done
log "serve up=$up after $(( $(date +%s) - T0 )) s"
r spark1 "grep -aE 'dsv41|error|Error|panick' $LOGD/serve-$TAG.log | tail -25" | tee -a "$OUT/steps.log"
r spark2 "grep -aE 'dsv41|error|Error|panick' $LOGD/worker-$TAG.log | tail -15" | tee -a "$OUT/steps.log"
[[ $up == 1 ]] || exit 1

chat() { # <name> <max_tokens> <prompt>
  curl -s -m 600 "$SURL/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","messages":[{"role":"user","content":sys.argv[2]}],"temperature":0,"max_tokens":int(sys.argv[1])}))' "$2" "$3")" > "$OUT/chat-$1.json"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d.get("choices",[{}])[0]; print(sys.argv[2], repr((c.get("message") or {}).get("content","")[:600]), d.get("usage"))' "$OUT/chat-$1.json" "$1" | tee -a "$OUT/steps.log"
}
if [[ " $STEPS " == *" greedy "* ]]; then
  chat capital 48 "What is the capital of France? Answer in one sentence."
  chat count 64 "Count from 1 to 20, separated by commas."
  chat code 200 "Write a Python function that returns the n-th Fibonacci number iteratively."
  chat story 256 "Write a short paragraph about a lighthouse keeper who finds a message in a bottle."
fi
if [[ " $STEPS " == *" golden "* ]]; then
  # greedy token ids on the golden prompt set (tools/prompts/golden*.jsonl), same format as TensorFold's goldens
  # (tools/dsv41/golden_tf.py tokens); compare later with `golden_tf.py compare A B`
  TOK=${TOKENIZER:-$REPO/.scratch/cache/l3-r0of1/tokenizer.json}
  ( cd "$REPO" && perl -e 'alarm shift; exec @ARGV' $(( $(left) > 60 ? $(left) - 60 : 60 )) uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens \
      --base-url $SURL/v1 --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$OUT/tok-intarsia-tp2.st" \
      --note "intarsia $(r spark1 'cat ~/intarsia-dsv41-tp2/bin/STAMP') TP=2 cap ${CAP:-32768} chunk ${CHUNK:-512}" ) 2>&1 | tee -a "$OUT/steps.log"
fi
if [[ " $STEPS " == *" speed "* ]]; then
  for n in 1 2; do chat speed$n 512 "Explain how a transformer language model generates text, in detail."; done
  python3 - "$OUT" <<'PY' | tee -a "$OUT/steps.log"
import json,sys,glob
for f in sorted(glob.glob(sys.argv[1]+"/chat-speed*.json")):
    d=json.load(open(f)); print(f.split("/")[-1], d.get("usage"), d.get("timings") or d.get("intarsia"))
PY
  curl -s -m 5 "$SURL/metrics" | grep -E 'decode|prefill|ttft' | head -20 | tee -a "$OUT/steps.log"
fi
# localeval (the fleet's harness, /opt/code/deepeval) inside this slot (FLEETQ_JOB is set): speed sweep and MMLU
LE_RECIPE="intarsia $(r spark1 'cat ~/intarsia-dsv41-tp2/bin/STAMP') scripts/dsv41-tp2/run-full.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap ${CAP:-32768} chunk ${CHUNK:-512}, engram hot ${HOT_MB:-1024} MiB"
le() { ( cd /opt/code/deepeval && perl -e 'alarm shift; exec @ARGV' $(( $(left) > 60 ? $(left) - 60 : 60 )) uv run localeval "$@" --url $SURL/v1 --model dsv41-intarsia --thinking off --recipe "$LE_RECIPE" ) 2>&1 | tee "$OUT/le-$1-$(date +%H%M%S).log" | tail -40 | tee -a "$OUT/steps.log"; }
if [[ " $STEPS " == *" lspeed "* ]]; then
  le speed --sizes 128,1024,4096,16384 --gen 128 --reps 3 --label dsv41-intarsia-tp2-speed
fi
if [[ " $STEPS " == *" mmlu "* ]] && [[ " $STEPS " == *" greedy "* ]] && ! grep -qi paris "$OUT/chat-capital.json"; then
  log "chat check failed (no 'Paris' in the capital answer): skipping MMLU"; STEPS="${STEPS/mmlu/}"
fi
if [[ " $STEPS " == *" mmlu "* ]]; then
  le quality --bench mmlu --limit ${MMLU_LIMIT:-25} --label dsv41-intarsia-tp2-mmlu
fi
if [[ " $STEPS " == *" hold "* ]]; then
  log "holding the server for ${HOLD_MIN:-30} min (or until $OUT/stop exists)"
  for i in $(seq 1 $(( ${HOLD_MIN:-30} * 6 ))); do [[ -f $OUT/stop ]] && break; (( $(left) > 0 )) || break; sleep 10; done
fi
r spark1 "grep -aE 'dsv41' $LOGD/serve-$TAG.log | tail -12" | tee -a "$OUT/steps.log"
log "done"
