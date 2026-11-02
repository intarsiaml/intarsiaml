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
BIN='$HOME/intarsia-dsv41-graphs/bin/intarsia'
C0='$HOME/intarsia-dsv41-tp2/cache/r0of2'; C1='$HOME/intarsia-dsv41-tp2/cache/r1of2'
WIP=10.100.152.3; WPORT=29701; SPORT=8899; SURL=http://100.120.88.104:$SPORT
LOGD='$HOME/intarsia-dsv41-graphs/logs'
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
  r $h "test -x ~/intarsia-dsv41-graphs/bin/intarsia && test -f ~/intarsia-nccl/libnccl.so.2" || { log "binary or NCCL library missing on $h"; exit 1; }
done
r spark1 "test -f $C0/manifest.json" && r spark2 "test -f $C1/manifest.json" || { log "rank cache missing"; exit 1; }
[[ "$(r spark1 'cat ~/intarsia-dsv41-graphs/bin/STAMP')" == "$(r spark2 'cat ~/intarsia-dsv41-graphs/bin/STAMP')" ]] || { log "binary stamps differ"; exit 1; }
log "binary $(r spark1 'cat ~/intarsia-dsv41-graphs/bin/STAMP'); out $OUT"

kill_ours() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]ntarsia-dsv41-graphs/bin/intarsia'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-dsv41-graphs/bin/intarsia' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-dsv41-graphs/bin/intarsia'; true"; done; }
STOPPED=0
restore() {
  local rc=$?
  log "restore (exit $rc)"
  kill "${WATCH:-}" 2>/dev/null
  kill_ours
  for h in spark1 spark2; do r $h "mkdir -p $LOGD; cp $LOGD/*.log /dev/null 2>/dev/null; true"; done
  scp -q "spark1:intarsia-dsv41-graphs/logs/serve-*.log" "$OUT/" 2>/dev/null; scp -q "spark2:intarsia-dsv41-graphs/logs/worker-*.log" "$OUT/" 2>/dev/null
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
      if [[ "$a" != NA ]] && awk "BEGIN{exit !($a < ${KILL_GIB:-5})}"; then ssh -n $h "pkill -KILL -f '[i]ntarsia-dsv41-graphs/bin/intarsia'"; line="$line KILLED-$h"; fi
    done
    echo "$line"; sleep 5
  done ) > "$OUT/mem.log" 2>&1 & WATCH=$!

# start both ranks with extra env $1 (on top of DENV / XENV); exits the script (restore runs) if they do not come up
start_ranks() {
TAG=$(date +%H%M%S)
# NSYS=1: both ranks run under `nsys launch` (no collection until the nsys step starts a session)
NL0=""; NL1=""
# (with --cuda-graph-trace=node: replayed CUDA graphs show their kernels one by one)
if [[ ${NSYS:-0} == 1 ]]; then NL0="nsys launch --session-new=dsv41r0x$TAG --trace=cuda,nvtx --cuda-graph-trace=node"; NL1="${NL0/dsv41r0x/dsv41r1x}"; fi
r spark2 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $1 $NL1 $BIN tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/worker-$TAG.log 2>&1 < /dev/null &"
sleep 2
r spark1 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $1 $NL0 $BIN serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/serve-$TAG.log 2>&1 < /dev/null &"
log "ranks started (tag $TAG)"
T0=$(date +%s); up=0
while (( $(date +%s) - T0 < 900 )); do
  curl -sf -m 5 "$SURL/health" >/dev/null && { up=1; break; }
  r spark1 "pgrep -f '[i]ntarsia-dsv41-graphs/bin/intarsia[^ ]* serve' >/dev/null" || { log "rank 0 exited"; break; }
  r spark2 "pgrep -f '[i]ntarsia-dsv41-graphs/bin/intarsia[^ ]* tp-worker' >/dev/null" || { log "rank 1 exited"; break; }
  sleep 5
done
log "serve up=$up after $(( $(date +%s) - T0 )) s"
r spark1 "grep -aE 'dsv41|error|Error|panick' $LOGD/serve-$TAG.log | tail -25" | tee -a "$OUT/steps.log"
r spark2 "grep -aE 'dsv41|error|Error|panick' $LOGD/worker-$TAG.log | tail -15" | tee -a "$OUT/steps.log"
[[ $up == 1 ]] || exit 1
}

# the steps of $STEPS against the running ranks
run_steps() {
# speculation lines per step: specn = lines so far in this config's serve log; specstep N LABEL [names] sums the lines
# after N (per request and per category with names; scripts/dsv41-tp2/specsum.py)
specn() { local n; n=$(r spark1 "grep -ac 'dsv41: spec ' $LOGD/serve-$TAG.log; true" 2>/dev/null | tail -1); echo "${n:-0}"; }
specstep() { r spark1 "grep -a 'dsv41: spec ' $LOGD/serve-$TAG.log | tail -n +$(( $1 + 1 ))" > "$OUT/spec-$2.txt" 2>/dev/null; [[ -s "$OUT/spec-$2.txt" ]] && python3 "$S/specsum.py" "$OUT/spec-$2.txt" "$2" "${3:-}" | tee -a "$OUT/steps.log"; true; }
chat() { # <name> <max_tokens> <prompt>
  curl -s -m 600 "$SURL/v1/chat/completions" -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","messages":[{"role":"user","content":sys.argv[2]}],"temperature":0,"max_tokens":int(sys.argv[1])}))' "$2" "$3")" > "$OUT/chat-$1.json"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); c=d.get("choices",[{}])[0]; print(sys.argv[2], repr((c.get("message") or {}).get("content","")[:600]), d.get("usage"))' "$OUT/chat-$1.json" "$1" | tee -a "$OUT/steps.log"
}
if [[ " $STEPS " == *" greedy "* ]]; then
  sn=$(specn)
  chat capital 48 "What is the capital of France? Answer in one sentence."
  chat count 64 "Count from 1 to 20, separated by commas."
  chat code 200 "Write a Python function that returns the n-th Fibonacci number iteratively."
  chat story 256 "Write a short paragraph about a lighthouse keeper who finds a message in a bottle."
  specstep "$sn" greedy chat_capital,count,code_fib,story
fi
if [[ " $STEPS " == *" neartie "* ]]; then
  # intarsia's top-20 at each first divergence from TensorFold's greedy golden (coordinator's near-tie check)
  NT_TF=${NT_TF:-/opt/code/intarsiaml/.worktrees/dsv41-golden-tf/tok-tensorfold-mia.st}
  NT_IA=${NT_IA:-$REPO/docs/results/2026-11-01-dsv41-tp2/tok-intarsia-tp2-w12-specoff.st}
  log "near-tie probe: $NT_TF vs $NT_IA"
  python3 "$S/neartie.py" "$NT_TF" "$NT_IA" "$SURL/v1" dsv41-intarsia "$OUT/neartie.md" 2>&1 | tee -a "$OUT/steps.log"
fi
if [[ " $STEPS " == *" golden "* ]]; then
  sn=$(specn)
  # greedy token ids on the golden prompt set (tools/prompts/golden*.jsonl), same format as TensorFold's goldens
  # (tools/dsv41/golden_tf.py tokens); compare later with `golden_tf.py compare A B`
  TOK=${TOKENIZER:-/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json}
  ( cd "$REPO" && perl -e 'alarm shift; exec @ARGV' $(( $(left) > 60 ? $(left) - 60 : 60 )) uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens \
      --base-url $SURL/v1 --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$OUT/tok-intarsia-tp2.st" \
      --note "intarsia $(r spark1 'cat ~/intarsia-dsv41-graphs/bin/STAMP') TP=2 cap ${CAP:-32768} chunk ${CHUNK:-512}" ) 2>&1 | tee -a "$OUT/steps.log"
  specstep "$sn" golden count,story,code,science,chat_short,chat_system,chat_multiturn,chat_reasoning,long_prose_300,long_prose_700
fi
if [[ " $STEPS " == *" neartie_fresh "* ]]; then
  # the same probe against this run's own golden tokens ($OUT/tok-intarsia-tp2.st from the golden step)
  NT_TF=${NT_TF:-/opt/code/intarsiaml/.worktrees/dsv41-golden-tf/tok-tensorfold-mia.st}
  log "near-tie probe (fresh): $NT_TF vs $OUT/tok-intarsia-tp2.st"
  python3 "$S/neartie.py" "$NT_TF" "$OUT/tok-intarsia-tp2.st" "$SURL/v1" dsv41-intarsia "$OUT/neartie-fresh.md" 2>&1 | tee -a "$OUT/steps.log"
fi
# prefillprof (with XENV="INTARSIA_DSV41_OPPROF=1 INTARSIA_DSV41_PROFILE=32"): one ~4K-token prompt + 40 tokens, then the
# per-kernel device-time table of rank 0 (multi-row = the prefill chunks, one-row = decode; a sync after every step)
if [[ " $STEPS " == *" prefillprof "* ]]; then
  chat prefillprof 40 "Summarize the following notes in one sentence. $(python3 -c 'print(" ".join(f"Note {i}: the lighthouse lamp was checked and the log was signed." for i in range(300)))')"
  sleep 2; r spark1 "grep -aA26 'dsv41 opprof (multi-row' $LOGD/serve-$TAG.log | tail -27; grep -aA26 'dsv41 opprof (one-row' $LOGD/serve-$TAG.log | tail -27" | tee -a "$OUT/steps.log"
fi
# rowcost (with INTARSIA_DSV41_PROFILE set): the whole-pass ms of one pass of t rows (a t-token id prompt, max_tokens 1),
# t = 1..8, 12, 16, 4 reps each after a warm-up: what a verify pass of t rows would cost (prompt passes start at
# position 0, so attention reads fewer keys than a verify deep in a reply; the warm-up passes join the medians)
if [[ " $STEPS " == *" rowcost "* ]]; then
  rc() { curl -s -m 120 "$SURL/v1/completions" -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys; t=int(sys.argv[1]); print(json.dumps({"model":"dsv41-intarsia","prompt":[1000 + 37 * i for i in range(t)],"max_tokens":1,"temperature":0}))' "$1")" > /dev/null; }
  for t in 4 8; do rc $t; done
  for rep in 1 2 3 4; do for t in 2 3 4 5 6 7 8 12 16; do rc $t; done; done
  sleep 1
  r spark1 "grep -a 'profile pass rows=' $LOGD/serve-$TAG.log" > "$OUT/rowcost.txt"
  python3 - "$OUT/rowcost.txt" <<'PY' | tee -a "$OUT/steps.log"
import re, sys, statistics as st, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.search(r"rows=(\d+): build ([\d.]+) ms, enqueue ([\d.]+) ms, whole pass ([\d.]+) ms", l)
    if m: d[int(m[1])].append((float(m[3]), float(m[4])))
print("rowcost: rows, passes, median enqueue ms, median whole pass ms")
for t in sorted(d): print(f"rowcost {t:>3} {len(d[t]):>3} {st.median(x[0] for x in d[t]):8.2f} {st.median(x[1] for x in d[t]):8.2f}")
PY
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
if [[ " $STEPS " == *" nsys "* && ${NSYS:-0} == 1 ]]; then
  # one forced-length request under nsys on both ranks; per-step breakdown from each rank's GPU trace
  NT=${NSYS_TOKENS:-48}
  # NSYS_LONG=1: the measured request has a ~4K-token prompt (prefill under nsys); the warm-up stays short
  NP="The history of the printing press begins"
  req() { local p=$NP; [[ $1 != 16 && ${NSYS_LONG:-0} == 1 ]] && p="$(python3 -c 'print(" ".join(f"Note {i}: the lighthouse lamp was checked and the log was signed." for i in range(300)))') The history of the printing press begins"
    curl -s -m 300 "$SURL/v1/completions" -H 'content-type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","prompt":sys.argv[2],"max_tokens":int(sys.argv[1]),"temperature":0,"ignore_eos":True}))' "$1" "$p")" > "$OUT/nsys-req-$1.json"; }
  req 16
  r spark1 "nsys start --session=dsv41r0x$TAG -o $LOGD/nsys-r0-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  r spark2 "nsys start --session=dsv41r1x$TAG -o $LOGD/nsys-r1-$TAG --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  t=$(date +%s); req $NT; log "nsys request: $NT tokens in $(( $(date +%s) - t )) s: $(head -c 300 "$OUT/nsys-req-$NT.json")"
  r spark1 "nsys stop --session=dsv41r0x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  r spark2 "nsys stop --session=dsv41r1x$TAG" 2>&1 | tail -2 | tee -a "$OUT/steps.log"
  for rk in 0 1; do
    h=spark$((rk + 1))
    r $h "nsys stats --report cuda_gpu_trace --format csv --output $LOGD/nsys-r$rk-$TAG $LOGD/nsys-r$rk-$TAG.nsys-rep > /dev/null 2>&1; ls -la $LOGD/nsys-r$rk-$TAG*" | tee -a "$OUT/steps.log"
    scp -q "$h:intarsia-dsv41-graphs/logs/nsys-r$rk-$TAG*" "$OUT/" 2>/dev/null
    f=$(ls "$OUT"/nsys-r$rk-$TAG*gpu_trace*.csv 2>/dev/null | head -1)
    [[ -n $f ]] && { log "rank $rk breakdown:"; python3 "$S/nsys_breakdown.py" "$f" --tokens $((NT - 2)) | tee -a "$OUT/steps.log"; }
  done
fi
# localeval (the fleet's harness, /opt/code/deepeval) inside this slot (FLEETQ_JOB is set): speed sweep and MMLU
LE_RECIPE="intarsia $(r spark1 'cat ~/intarsia-dsv41-graphs/bin/STAMP') scripts/dsv41-tp2/run-full.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap ${CAP:-32768} chunk ${CHUNK:-512}, engram hot ${HOT_MB:-1024} MiB"
le() { ( cd /opt/code/deepeval && perl -e 'alarm shift; exec @ARGV' $(( $(left) > 60 ? $(left) - 60 : 60 )) uv run localeval "$@" --url $SURL/v1 --model dsv41-intarsia --thinking off --recipe "$LE_RECIPE" ) 2>&1 | tee "$OUT/le-$1-$(date +%H%M%S).log" | tail -40 | tee -a "$OUT/steps.log"; }
if [[ " $STEPS " == *" lspeed "* ]]; then
  sn=$(specn)
  le speed --sizes ${LSIZES:-128,1024,4096,16384} --gen 128 --reps ${LREPS:-3} --label dsv41-intarsia-tp2-speed${LSUF:-}
  specstep "$sn" lspeed
fi
if [[ " $STEPS " == *" mmlu "* ]] && [[ " $STEPS " == *" greedy "* ]] && ! grep -qi paris "$OUT/chat-capital.json"; then
  log "chat check failed (no 'Paris' in the capital answer): skipping MMLU"; STEPS="${STEPS/mmlu/}"
fi
if [[ " $STEPS " == *" mmlu "* ]]; then
  sn=$(specn)
  le quality --bench mmlu --limit ${MMLU_LIMIT:-25} --label dsv41-intarsia-tp2-mmlu${LSUF:-}
  specstep "$sn" mmlu
fi
if [[ " $STEPS " == *" hold "* ]]; then
  log "holding the server for ${HOLD_MIN:-30} min (or until $OUT/stop exists)"
  for i in $(seq 1 $(( ${HOLD_MIN:-30} * 6 ))); do [[ -f $OUT/stop ]] && break; (( $(left) > 0 )) || break; sleep 10; done
fi
r spark1 "grep -aE 'dsv41' $LOGD/serve-$TAG.log | tail -12" | tee -a "$OUT/steps.log"
# speculation: the per-request lines (tokens a round, drafts accepted) summed over this config
r spark1 "grep -a 'dsv41: spec ' $LOGD/serve-$TAG.log" > "$OUT/spec.txt" || true
if [[ -s "$OUT/spec.txt" ]]; then
  python3 - "$OUT/spec.txt" <<'PY' | tee -a "$OUT/steps.log"
import re, sys
t = r = a = d = 0
for l in open(sys.argv[1]):
    m = re.search(r"spec (\d+) tokens in (\d+) rounds .*?, (\d+)/(\d+) drafts accepted", l)
    if m:
        t += int(m[1]); r += int(m[2]); a += int(m[3]); d += int(m[4])
print(f"spec total: {t} tokens in {r} rounds = {t / max(r, 1):.3f} a round; drafts accepted {a}/{d} = {100 * a / max(d, 1):.1f} %")
PY
fi
}

# AB="ENV_A|ENV_B" (e.g. "INTARSIA_DSV41_EXL3_SPLITK=0|INTARSIA_DSV41_EXL3_SPLITK=48"): run the steps once per config, each with
# a fresh start of both ranks (same binary, ~25 s load), outputs in $OUT/ab0, $OUT/ab1 and localeval labels suffixed -ab0 / -ab1.
# STEPS_ABi / NSYS_ABi override STEPS / NSYS per config (e.g. MMLU only for the new one, nsys in a config of its own).
if [[ -n ${AB:-} ]]; then
  OUT0=$OUT; i=0
  IFS='|' read -ra CFGS <<< "$AB"
  for cfg in "${CFGS[@]}"; do
    OUT=$OUT0/ab$i; mkdir -p "$OUT"; LSUF=-ab$i
    v="STEPS_AB$i"; STEPS_SAVE=$STEPS; STEPS=${!v:-$STEPS}
    v="NSYS_AB$i"; NSYS_SAVE=${NSYS:-0}; NSYS=${!v:-$NSYS_SAVE}
    # BIN_ABi: another binary for this config (same path prefix, so kill_ours and the watcher still match it)
    v="BIN_AB$i"; BIN_SAVE=$BIN; BIN=${!v:-$BIN}
    log "config ab$i: $cfg; steps: $STEPS; nsys $NSYS; bin $BIN"
    start_ranks "$cfg"
    run_steps
    STEPS=$STEPS_SAVE; NSYS=$NSYS_SAVE; BIN=$BIN_SAVE
    kill_ours; log "config ab$i done"
    i=$((i + 1))
  done
  OUT=$OUT0
else
  start_ranks ""
  run_steps
fi
log "done"
