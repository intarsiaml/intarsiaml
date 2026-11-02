#!/bin/bash
# TensorFold BLACKFOLD (head spark1 + worker spark2): one 48-token request on a ~1K-token prompt under nsys
# (cuda-graph-trace=node), CSV of the GPU trace of both ranks. Runs inside one fleetq slot that takes down both vLLMs; vLLM is
# always restored (trap). Inputs under ~/code/blackfold-tp2 are read only (mounted :ro); outputs under
# ~/intarsia-dsv41-draftdev/tfprof and $OUT. Memory: TENSORFOLD_MEMORY_RESERVE_GIB=11.6 (bf-node-prof.sh), 16K pool, the
# watcher kills our container under KILL_GIB (2.8) of MemAvailable.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-$HERE/out}; mkdir -p "$OUT"
API=http://100.120.88.104:18191
PD='$HOME/intarsia-dsv41-draftdev/tfprof'
NODE='$HOME/intarsia-dsv41-draftdev/tfprof/bf-node-prof.sh'
N=ds41-prof-dd
NS=/nsys/target-linux-sbsa-armv8/nsys
JOB=${FLEETQ_JOB:?run under fleetq run}
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$JOB; $2"; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
for h in spark1 spark2; do
  eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
  [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is -m $eo"; exit 1; }
  r $h 'docker ps -a --format "{{.Names}}" | grep -E "^(dsv41-1017-exl3-tf|ds41-native|ds41-prof-dd)$"' && { log "a TF container exists on $h"; exit 1; }
  r $h 'ss -ltn | grep -E ":18191 "' && { log "port 18191 busy on $h"; exit 1; }
  r $h 'B=$HOME/code/blackfold-tp2; set -e; ls $B/cache/rank-cache/rank*of2-*.bin $B/cache/dsv41_token_map.json $B/engine/out/zig-out/bin/tf-dsv41-lanes $B/engine/out/zig-out/native/bin/tensorfold-native $B/chat_template.native.jinja >/dev/null; ls /opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys >/dev/null' || { log "missing files on $h"; exit 1; }
  r $h "mkdir -p $PD"
  ssh -o BatchMode=yes $h "cat > $PD/bf-node-prof.sh; chmod +x $PD/bf-node-prof.sh" < "$HERE/bf-node-prof.sh"
done
r spark1 'test -s $HOME/code/blackfold-tp2/api-key' || { log "no api key file"; exit 1; }
log "preflight ok"
STOPPED=0
restore() {
  log "restore"
  kill "${WATCH:-}" 2>/dev/null
  for h in spark1 spark2; do r $h "docker logs $N > $PD/last-$h.log 2>&1; docker rm -f $N >/dev/null 2>&1; true"; scp -q $h:intarsia-dsv41-draftdev/tfprof/last-$h.log "$OUT/" 2>/dev/null; done
  if [[ $STOPPED == 1 ]]; then
    for h in spark1 spark2; do r $h 'docker start missioncntrl-guidance >/dev/null'; done
    for h in spark1 spark2; do
      for _ in $(seq 1 120); do r $h 's=$(docker inspect -f "{{.State.StartedAt}}" missioncntrl-guidance); t=$(docker logs -t --tail 2000 missioncntrl-guidance 2>&1 | grep "Application startup complete" | tail -1 | cut -d" " -f1); [[ -n $t && $t > $s ]]' && break; sleep 10; done
      log "$h vLLM $(r $h 'docker ps --filter name=missioncntrl-guidance --format {{.Status}}')"
    done
  fi
  fleetq mark-up spark1-vllm; fleetq mark-up spark2-vllm
  log restored
}
trap restore EXIT
for h in spark1 spark2; do r $h 'docker stop -t 60 missioncntrl-guidance >/dev/null'; done; STOPPED=1
log "vLLM stopped"
for h in spark1 spark2; do r $h '$HOME/code/blackfold-tp2/bf-node.sh evict'; done
( while true; do line="$(date +%T)"; for h in spark1 spark2; do a=$(ssh -n -o ConnectTimeout=5 $h "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA); line="$line $h=$a"; if [[ "$a" != NA ]] && awk "BEGIN{exit !($a < ${KILL_GIB:-2.8})}"; then ssh -n $h "docker kill $N"; line="$line KILLED-$h"; fi; done; echo "$line"; sleep 3; done ) > "$OUT/mem.log" 2>&1 & WATCH=$!
sleep 5; log "memory: $(tail -1 "$OUT/mem.log")"
T0=$(date +%s)
r spark2 "CTX=16384 TF_RESERVE_GIB=${TF_RESERVE_GIB:-10.9} $NODE worker" >/dev/null || { log "worker start failed"; exit 1; }
r spark1 "CTX=16384 TF_RESERVE_GIB=${TF_RESERVE_GIB:-10.9} $NODE head" >/dev/null || { log "head start failed"; exit 1; }
ready=0
for i in $(seq 1 240); do
  h=$(curl -s -m 3 "$API/health" || true)
  if echo "$h" | grep -q '"status": *"ok"' && ! echo "$h" | grep -q '"warming": *true'; then ready=1; break; fi
  for n in spark1 spark2; do r $n "docker ps --format '{{.Names}}' | grep -qx $N" || { log "$N stopped on $n"; r $n "docker logs --tail 30 $N 2>&1" | tee -a "$OUT/steps.log"; exit 1; }; done
  sleep 5
done
log "ready=$ready after $(( $(date +%s) - T0 )) s; $(tail -1 "$OUT/mem.log")"
[[ $ready == 1 ]] || exit 1
sleep 10
KEY="$(r spark1 'head -n 1 ~/code/blackfold-tp2/api-key')"
MODEL="$(curl -s -m 10 -H "Authorization: Bearer $KEY" "$API/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
# a ~1K-token prompt of ordinary prose (this lane's own results doc), then the continuation
python3 - "$MODEL" "${PROMPT_FILE:-$HERE/prompt.txt}" > "$OUT/body-template.json" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[1], "prompt": open(sys.argv[2]).read(), "temperature": 0}))
PY
req() { # max_tokens name
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["max_tokens"]=int(sys.argv[2]); print(json.dumps(d))' "$OUT/body-template.json" "$1" > "$OUT/body-$2.json"
  curl -s -m 300 "$API/v1/completions" -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d @"$OUT/body-$2.json" > "$OUT/resp-$2.json"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("usage"), repr(d["choices"][0]["text"][:120]))' "$OUT/resp-$2.json" | tee -a "$OUT/steps.log"
}
req 16 warm1; req 16 warm2
log "mem before trace: $(tail -1 "$OUT/mem.log")"
r spark1 "docker exec $N $NS start --session=tfh -o /out/tf-r0 --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -3 | tee -a "$OUT/steps.log"
r spark2 "docker exec $N $NS start --session=tfw -o /out/tf-r1 --force-overwrite=true --sample=none --cpuctxsw=none" 2>&1 | tail -3 | tee -a "$OUT/steps.log"
t=$(date +%s); req 48 traced; log "traced request: 48 tokens in $(( $(date +%s) - t )) s; $(tail -1 "$OUT/mem.log")"
r spark1 "docker exec $N $NS stop --session=tfh" 2>&1 | tail -3 | tee -a "$OUT/steps.log"
r spark2 "docker exec $N $NS stop --session=tfw" 2>&1 | tail -3 | tee -a "$OUT/steps.log"
# untraced decode of the same prompt for the plain tok/s of this config
t=$(date +%s); req 128 untraced; log "untraced request: 128 tokens in $(( $(date +%s) - t )) s"
for k in 0 1; do
  h=spark$((k + 1))
  r $h "cd $PD && /usr/local/bin/nsys stats --report cuda_gpu_trace --format csv --output $PD/tf-r$k $PD/tf-r$k.nsys-rep >/dev/null 2>&1; ls -la $PD/tf-r$k*" | tee -a "$OUT/steps.log"
  scp -q "$h:intarsia-dsv41-draftdev/tfprof/tf-r$k*gpu_trace*.csv" "$OUT/" 2>/dev/null
done
r spark1 "docker logs $N 2>&1 | grep -aiE 'spec|round|accept|draft' | tail -20" | tee -a "$OUT/steps.log"
log "done; $(tail -1 "$OUT/mem.log")"
