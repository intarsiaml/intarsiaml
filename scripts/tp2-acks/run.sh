#!/usr/bin/env bash
# Lane fx12: tp=2 across the two Sparks (rank 0 = spark1 `--tp-remote`, rank 1 = spark2 `intarsia tp-worker`, NCCL over
# ConnectX-7), the 35B, for the acknowledged Truncate / Snapshot protocol: dev (this change) against base (origin/main).
# ONE fleetq job holds both Sparks and takes vLLM down; only between 00:00 and 07:00 EDT (the script refuses outside it):
#   fleetq run --host spark1,spark2 --exclusive --takes-down flight,guidance,capcom,retro --mem spark1=100,spark2=95 \
#     --label "fx12 tp=2 ack gate" -- scripts/tp2-acks/run.sh
# Per arm (base, dev): 256 greedy ids on a short and a ~4k-token prompt (`generate --show-ids`), then a `serve` with the
# 100-request seeded soak (scripts/tp2-acks/soak100.py: prefix reuse, repeats, turns). Then base ids / soak answers are
# compared with dev's, and a base rank 0 is paired with a dev worker (must refuse, both sides naming the versions).
# The EXIT trap kills every process started and restarts vLLM on both boxes; downtime goes to raw/downtime.log.
# Binaries from scripts/tp2-acks/build.sh. Output: docs/results/2026-10-22-fx12-tp-remote-acks/raw/$OUT.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
RAW=$REPO/docs/results/2026-10-22-fx12-tp-remote-acks/raw
OUT=${OUT:-window-$(date +%H%M)}
L=$RAW/$OUT; mkdir -p "$L"
ARMS=${ARMS:-base dev}
MAX_MIN=${MAX_MIN:-60}
now=$(date +%s)
end7=$(date -j -f '%Y-%m-%d %H:%M:%S' "$(date +%F) 07:00:00" +%s 2>/dev/null || echo $(( now + 3600 )))
hour=$(date +%-H)
if (( hour >= 7 )); then echo "outside the 00:00-07:00 EDT window (hour $hour): not touching vLLM" | tee -a "$L/steps.log"; exit 3; fi
if (( end7 - now < 1500 )); then echo "less than 25 min to 07:00 EDT: not touching vLLM" | tee -a "$L/steps.log"; exit 3; fi
DEADLINE=$(( now + MAX_MIN * 60 )); (( end7 < DEADLINE )) && DEADLINE=$end7
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29555
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
VLLM_C=missioncntrl-guidance
vllm_url() { [[ $1 == spark1 ]] && echo http://100.120.88.104:8000/v1/models || echo http://100.96.95.89:8000/v1/models; }
log() { echo "$(date +%T) $*" | tee -a "$L/steps.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
r() { local h=$1; shift; ssh -n -o BatchMode=yes "$h" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
bin() { echo "\$HOME/intarsia-tp2/bin/fx12-$1-intarsia"; }
XO='~/intarsia-tp2/xout/'$OUT

cleanup_procs() {
  for h in spark1 spark2; do
    r $h "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[i]ntarsia-tp2/bin/fx12-'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-tp2/bin/fx12-' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-tp2/bin/fx12-'; true"
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
      printf '%s job=%s role=%s stop=%s up=%s down=%d s label="fx12 tp=2 ack gate %s" serving=%s\n' "$(date +%F)" "$FLEETQ_JOB" "$role" \
        "$(date -r $T_STOP +%H:%M:%S)" "$(date -r $t +%H:%M:%S)" $(( t - T_STOP )) "$OUT" "$([[ $ok == 1 ]] && echo yes || echo NO)" >> "$RAW/downtime.log"
    done
  fi
  rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null; rsync -a "spark2:intarsia-tp2/xout/$OUT/" "$L/spark2/" 2>/dev/null
  trap - EXIT INT TERM
  exit $rc
}
trap restore EXIT INT TERM

for h in spark1 spark2; do
  for a in $ARMS; do
    # refuse to take vLLM down for a binary that is not there
    r $h "mkdir -p ~/intarsia-tp2/xout/$OUT; rm -f ~/intarsia-tp2/stop-worker; echo $h $a: \$($(bin $a) --version 2>&1 | tr '\n' ' ')" | tee -a "$L/steps.log"
    r $h "$(bin $a) --version >/dev/null 2>&1" || { log "ABORT: $(bin $a) does not run on $h (vLLM untouched)"; exit 4; }
  done
done
scp -q "$S/soak100.py" spark1:intarsia-tp2/soak100.py
T_STOP=$(date +%s); STOPPED=1
for h in spark1 spark2; do r $h "docker stop -t 30 $VLLM_C >/dev/null" && log "vLLM stopped on $h"; done
for h in spark1 spark2; do
  gb=0
  for i in $(seq 1 30); do gb=$(r $h "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"); (( gb >= 95 )) && break; sleep 2; done
  log "$h MemAvailable $gb GB"
done

# rank 1 on spark2: a loop of tp-workers (one per rank-0 connection) until stop-worker. <tag> <binary> <flags...>
worker_start() {
  local tag=$1 b=$2; shift 2
  r spark2 "rm -f ~/intarsia-tp2/stop-worker; nohup bash -c 'i=0; while [ ! -f ~/intarsia-tp2/stop-worker ]; do i=\$((i+1)); env $NCCLENV $b tp-worker --listen $WIP:$WPORT $* >> $XO/worker-$tag.log 2>&1; echo \"worker exit \$? (\$i)\" >> $XO/worker-$tag.log; sleep 1; done' > /dev/null 2>&1 < /dev/null &"
  sleep 1
}
worker_stop() {
  r spark2 "touch ~/intarsia-tp2/stop-worker; pkill -TERM -f '[t]p-worker --listen'; for i in \$(seq 1 30); do pgrep -f '[t]p-worker --listen' >/dev/null || break; sleep 1; done; pkill -KILL -f '[t]p-worker --listen'; true"
}

GEN_FLAGS="-m $M35 --backend cuda --tp 2 --max-seq 20480 --mem-budget 16 --profile-every 0"
# 256 greedy ids: <arm> <tag> <chars>
xgen() {
  local arm=$1 tag=$2 chars=$3 b; b=$(bin $arm)
  (( $(left) > 240 )) || { log "SKIP gen $arm $tag: $(left) s left"; return 0; }
  worker_start $arm-$tag $b $GEN_FLAGS
  r spark1 "env $NCCLENV timeout 600 $b generate $GEN_FLAGS --tp-remote $WIP:$WPORT --chat --no-think --ignore-eos --show-ids \
    -p \"\$(head -c $chars ~/intarsia-tp2/prompts/corpus_docs.txt)\" --max-tokens 256 > $XO/gen-$arm-$tag.out 2> $XO/gen-$arm-$tag.err; echo gen $arm $tag rc=\$?; \
    grep -aE 'finish=|prompt' $XO/gen-$arm-$tag.err | head -3; grep -a '^ids:' $XO/gen-$arm-$tag.err > $XO/gen-$arm-$tag.ids" | tee -a "$L/steps.log"
  worker_stop
}

SFLAGS="-m $M35 --backend cuda --max-seq 20480 --mem-budget 45"
# serve + the seeded 100-request soak: <arm>
xsoak() {
  local arm=$1 b o up=0; b=$(bin $arm); o=$XO/soak-$arm
  (( $(left) > 420 )) || { log "SKIP soak $arm: $(left) s left"; return 0; }
  worker_start soak-$arm $b --serving $SFLAGS --tp 2
  r spark1 "mkdir -p $o; nohup env $NCCLENV $b serve $SFLAGS --port 8090 --host 127.0.0.1 --served-model-name intarsia-35b --tp 2 --tp-remote $WIP:$WPORT > $o/serve.log 2>&1 < /dev/null & echo \$! > $o/serve.pid"
  for i in $(seq 1 150); do r spark1 "curl -sf -m 3 http://127.0.0.1:8090/health >/dev/null" && { up=1; break; }; sleep 2; done
  if [[ $up == 1 ]]; then
    r spark1 "python3 ~/intarsia-tp2/soak100.py --url http://127.0.0.1:8090 --corpus ~/intarsia-tp2/prompts/corpus_docs.txt --out $XO/soak-$arm.jsonl > $o/soak.log 2>&1; echo soak $arm rc=\$?; tail -n 3 $o/soak.log | cut -c1-400; \
      curl -sf -m 5 http://127.0.0.1:8090/health >/dev/null && echo health-after ok || echo health-after FAIL; grep -ciE 'error|failed|panic' $o/serve.log" | tee -a "$L/steps.log"
  else
    log "soak $arm: server did not come up"; r spark1 "tail -5 $o/serve.log" | tee -a "$L/steps.log"
  fi
  r spark1 "kill \$(cat $o/serve.pid); for i in \$(seq 1 30); do kill -0 \$(cat $o/serve.pid) 2>/dev/null || break; sleep 1; done; kill -9 \$(cat $o/serve.pid) 2>/dev/null; true"
  worker_stop
}

# the required comparisons first (short ids, 4k ids), then the soak, then the cross-build pairing
for t in short:300 4k:15000; do
  for arm in $ARMS; do xgen $arm ${t%%:*} ${t#*:}; done
done
for arm in dev base; do [[ " $ARMS " == *" $arm "* ]] && xsoak $arm; done

# a base rank 0 against a dev worker: both must refuse to pair (each loads its shard first, then the handshake fails)
if [[ " $ARMS " == *" base "* && " $ARMS " == *" dev "* ]] && (( $(left) > 420 )); then
  worker_start xpair $(bin dev) $GEN_FLAGS
  r spark1 "env $NCCLENV timeout 400 $(bin base) generate $GEN_FLAGS --tp-remote $WIP:$WPORT --chat --no-think -p hi --max-tokens 4 > $XO/xpair.out 2> $XO/xpair.err; echo xpair rc=\$?; grep -a -m2 -iE 'protocol|started differently|handshake' $XO/xpair.err | cut -c1-400" | tee -a "$L/steps.log"
  sleep 2; r spark2 "grep -a -m2 -iE 'protocol|started differently|handshake' $XO/worker-xpair.log | cut -c1-400" | tee -a "$L/steps.log"
  worker_stop
fi

# compare base with dev
rsync -a "spark1:intarsia-tp2/xout/$OUT/" "$L/" 2>/dev/null
python3 - "$L" <<'PY' | tee -a "$L/steps.log"
import json, os, re, sys
d = sys.argv[1]
def ids(f):
    if not os.path.exists(f) or not open(f).read().strip(): return None
    return [int(v) for v in re.findall(r"-?\d+", open(f).read().split(":", 1)[1])]
for t in ("short", "4k"):
    x, y = ids(f"{d}/gen-base-{t}.ids"), ids(f"{d}/gen-dev-{t}.ids")
    if x is None or y is None:
        print(f"IDS {t}: missing (base {'ok' if x else 'none'}, dev {'ok' if y else 'none'})"); continue
    k = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print(f"IDS {t}: {'IDENTICAL' if x == y else 'DIFFER'} (first divergence {k}, {len(x)}/{len(y)} ids)")
def rows(f):
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else None
x, y = rows(f"{d}/soak-base.jsonl"), rows(f"{d}/soak-dev.jsonl")
if x is None or y is None:
    print("SOAK compare: missing")
else:
    same = [a["text"] == b["text"] for a, b in zip(x, y)]
    print(f"SOAK compare: {len(x)} base / {len(y)} dev answers, {sum(same)} identical, cached>0: base {sum(r['cached']>0 for r in x)} dev {sum(r['cached']>0 for r in y)}")
PY
log "done, $(left) s of budget left"
