#!/bin/bash
# Lane dsv41/decode2: does `serve --tp 2 --tp-remote` still work for the Qwen3.6-35B on a binary with INTARSIA_TP_SPIN_MS unset?
# One short greedy chat request over TP=2 (rank 0 spark1, rank 1 spark2), vLLM left up (35B NVFP4 is ~11 GB a rank):
#   SPARK_DIR=~/intarsia-mylane BIN_NAME=intarsia.main fleetq run --host spark1,spark2 --load gpu --mem 30 --expect 12 \
#     --label "35B tp2 serve check" -- scripts/dsv41-tp2/tp35-check.sh
# Pass = the answer names Paris, rank 1 still alive after the requests, no panic in either log, no spin line. Output in $OUT (default .scratch/tp35-HHMM).
source "$(dirname "${BASH_SOURCE[0]}")/spark-dir.sh"
set -u
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$S/../.." && pwd)"
OUT=${OUT:-$REPO/.scratch/tp35-$(date +%H%M)}; mkdir -p "$OUT"
BIN="$SPARK_DIR/bin/${BIN_NAME:-intarsia}"; LOGD="$SPARK_DIR/logs"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
WIP=10.100.152.3; WPORT=29702; SPORT=8898; SURL=http://100.120.88.104:$SPORT
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN"
FL="-m $M35 --backend cuda --tp 2 --max-seq 4096 --mem-budget 20 --spec off"
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
# kill by the PIDs the launches wrote (never by pattern: another lane's processes may run on these boxes)
stop() { for h in spark1 spark2; do r $h "p=\$(cat $LOGD/tp35-$h.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; for i in \$(seq 1 20); do kill -0 \$p 2>/dev/null || break; sleep 1; done; kill -KILL \$p 2>/dev/null; true"; done; }
trap stop EXIT
log "binary $BIN md5 $(r spark1 "md5sum $BIN | cut -c1-12") (STAMP file: $(r spark1 "cat $SPARK_DIR/bin/STAMP")), INTARSIA_TP_SPIN_MS in env: '${INTARSIA_TP_SPIN_MS:-unset}'"
r spark2 "mkdir -p $LOGD; nohup env $NCCLENV $BIN tp-worker --serving --listen $WIP:$WPORT $FL > $LOGD/tp35-worker.log 2>&1 < /dev/null & echo \$! > $LOGD/tp35-spark2.pid"
sleep 2
r spark1 "mkdir -p $LOGD; nohup env $NCCLENV $BIN serve $FL --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name q35 --no-supervisor --no-ui > $LOGD/tp35-serve.log 2>&1 < /dev/null & echo \$! > $LOGD/tp35-spark1.pid"
T0=$(date +%s); up=0
while (( $(date +%s) - T0 < 420 )); do
  curl -sf -m 5 "$SURL/health" >/dev/null && { up=1; break; }
  r spark1 "kill -0 \$(cat $LOGD/tp35-spark1.pid)" || { log "rank 0 exited"; break; }
  r spark2 "kill -0 \$(cat $LOGD/tp35-spark2.pid)" || { log "rank 1 exited"; break; }
  sleep 5
done
log "serve up=$up after $(( $(date +%s) - T0 )) s"
if [[ $up == 1 ]]; then
  for i in 1 2 3; do
    curl -s -m 120 "$SURL/v1/chat/completions" -H 'content-type: application/json' \
      -d '{"model":"q35","messages":[{"role":"user","content":"What is the capital of France? Answer in one sentence."}],"temperature":0,"max_tokens":400}' > "$OUT/chat$i.json"
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("answer", repr(d["choices"][0]["message"]["content"][:200]), d.get("usage"))' "$OUT/chat$i.json" | tee -a "$OUT/steps.log"
  done
fi
r spark1 "cp $LOGD/tp35-serve.log /dev/null; grep -aEi 'panick|error|spin|decode graph|tp:' $LOGD/tp35-serve.log | tail -15" | tee -a "$OUT/steps.log"
r spark2 "grep -aEi 'panick|error|spin|decode graph|tp:' $LOGD/tp35-worker.log | tail -15" | tee -a "$OUT/steps.log"
scp -q "spark1:$SPARK_REL/logs/tp35-serve.log" "$OUT/"; scp -q "spark2:$SPARK_REL/logs/tp35-worker.log" "$OUT/"
# the ranks ran with the spin switch unset: neither log has the line a set switch prints
spin=$(cat "$OUT"/tp35-*.log | grep -ac 'reader spins')
log "spin lines in both logs: $spin"
ok=1
[[ $up == 1 ]] && grep -qi paris "$OUT/chat1.json" || ok=0
grep -aqi 'panick' "$OUT"/tp35-*.log && ok=0
[[ $spin == 0 ]] || ok=0
# rank 1 must still be serving after the requests (it prints no graph counters while it runs)
r spark2 "kill -0 \$(cat $LOGD/tp35-spark2.pid)" || ok=0
if [[ $ok == 1 ]]; then log "TP35 RESULT: PASS"; else log "TP35 RESULT: FAIL"; exit 1; fi
