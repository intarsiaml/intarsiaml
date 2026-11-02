#!/bin/bash
# 3-layer TP=2 over NCCL (real graph path, Engram layer 1 in the graph) with env arms: text and whole-pass ms. Under fleetq.
set -u
: "${FLEETQ_JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; KD="$HERE/../tfknock"; OUT=${OUT:-$HERE/out-$(date +%H%M)}; mkdir -p "$OUT"
SD='$HOME/intarsia-dsv41-draftdev'; BIN="$SD/bin/intarsia"; LOGD="$SD/logs"
C0='$HOME/intarsia-dsv41-tp2/cache/l3-r0of2'; C1='$HOME/intarsia-dsv41-tp2/cache/l3-r1of2'
WIP=10.100.152.3; WPORT=29741; SPORT=8941; SURL=http://100.120.88.104:$SPORT
NCCLENV='INTARSIA_NCCL_LIB=$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=WARN NCCL_MAX_NCHANNELS=2 NCCL_BUFFSIZE=1048576'
DENV='INTARSIA_DSV41_CAP=8192 INTARSIA_DSV41_CHUNK=512 INTARSIA_DSV41_ENGRAM=$HOME/dsv41-engram INTARSIA_DSV41_ENGRAM_HOT_MB=1024 INTARSIA_DSV41_SPEC=1 INTARSIA_DSV41_PROFILE=1000000 RUST_BACKTRACE=1'
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
kill_ours() { for h in spark1 spark2; do r $h "pkill -KILL -f '[i]ntarsia-dsv41-draftdev/bin/intarsia' ; true"; done; sleep 2; }
trap 'kill "${W:-}" 2>/dev/null; kill_ours; log end' EXIT
( while true; do l="$(date +%T)"; for h in spark1 spark2; do a=$(ssh -n -o ConnectTimeout=5 $h "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA); l="$l $h=$a"; if [[ $a != NA ]] && awk "BEGIN{exit !($a < 5)}"; then ssh -n $h "pkill -KILL -f '[i]ntarsia-dsv41-draftdev/bin/intarsia'"; l="$l KILLED-$h"; fi; done; echo "$l"; sleep 5; done ) > "$OUT/mem.log" 2>&1 & W=$!
r spark1 "cat ~/intarsia-dsv41-draftdev/bin/STAMP" | tee -a "$OUT/steps.log"
# ARMS: name=ENV ENV;name=ENV ...   (ENV "-" = none)
IFS=';' read -ra AS <<< "${ARMS:?}"
for a in "${AS[@]}"; do
  n=${a%%=*}; e=${a#*=}; [[ $e == "-" ]] && e=""
  log "arm $n: $e"; TAG=$n-$(date +%H%M%S)
  r spark2 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $e $BIN tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/w3-$TAG.log 2>&1 < /dev/null &"
  sleep 2
  r spark1 "mkdir -p $LOGD; nohup env $NCCLENV $DENV $e $BIN serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/s3-$TAG.log 2>&1 < /dev/null &"
  up=0; for i in $(seq 1 60); do curl -sf -m 3 $SURL/health >/dev/null && { up=1; break; }; sleep 3; done
  log "up=$up"; [[ $up == 1 ]] || { r spark1 "tail -5 $LOGD/s3-$TAG.log"; kill_ours; continue; }
  : > "$OUT/$n.txt"
  for rep in 1 2; do for p in 0 1 2 3; do
    curl -s -m 58 $SURL/v1/completions -H 'content-type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","prompt":open(sys.argv[1]).read(),"max_tokens":96,"temperature":0}))' "$KD/prompt$p.txt")" | python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin); t=d["choices"][0]["text"]; print(hashlib.sha1(t.encode()).hexdigest()[:12], len(t))' >> "$OUT/$n.txt" || { log "request failed/hung in arm $n"; break 2; }
  done; done
  # the same with the logprob of every token (exact float text): the logits, not only the argmax
  for p in 0 2; do
    curl -s -m 58 $SURL/v1/completions -H 'content-type: application/json' -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"dsv41-intarsia","prompt":open(sys.argv[1]).read(),"max_tokens":48,"temperature":0,"logprobs":3}))' "$KD/prompt$p.txt")" | python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin); l=d["choices"][0].get("logprobs"); print("lp", hashlib.sha1(json.dumps(l,sort_keys=True).encode()).hexdigest()[:12], len(json.dumps(l)))' >> "$OUT/$n.txt" || log "logprob request failed in arm $n"
  done
  sleep 1
  r spark1 "grep -a 'profile pass rows=\|dsv41: spec \|captured the\|panicked\|NCCL WARN' $LOGD/s3-$TAG.log" > "$OUT/$n.prof"
  r spark2 "grep -a 'panicked\|NCCL WARN' $LOGD/w3-$TAG.log" > "$OUT/$n.wprof"
  kill_ours
done
python3 "$HERE/sum3.py" "$OUT" | tee -a "$OUT/steps.log"
