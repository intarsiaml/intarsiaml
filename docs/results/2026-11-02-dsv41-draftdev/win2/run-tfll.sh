#!/bin/bash
# TensorFold BLACKFOLD recipe (bf-node-knock.sh with side and prefetch on, reserve 10.9, 16K pool) through localeval only: the speed
# sweep and the identical-prompt dataset, the same two runs as our engine. One fleetq job over both Sparks; vLLM restored in the trap.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-$HERE/out-ll}; mkdir -p "$OUT"
API=http://100.120.88.104:18191
NODE='$HOME/intarsia-dsv41-draftdev/tfprof/bf-node-knock.sh'
PD='$HOME/intarsia-dsv41-draftdev/tfprof'
N=ds41-knock-dd
JOB=${FLEETQ_JOB:?run under fleetq run}
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$JOB; $2"; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
for h in spark1 spark2; do
  eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
  [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is -m $eo"; exit 1; }
  r $h 'docker ps -a --format "{{.Names}}" | grep -E "^(dsv41-1017-exl3-tf|ds41-native|ds41-prof-dd|ds41-knock-dd)$"' && { log "a TF container exists on $h"; exit 1; }
  r $h 'ss -ltn | grep -E ":18191 "' && { log "port 18191 busy on $h"; exit 1; }
  r $h "mkdir -p $PD"
  ssh -o BatchMode=yes $h "cat > $PD/bf-node-knock.sh; chmod +x $PD/bf-node-knock.sh" < "$HERE/bf-node-knock.sh"
done
BF_KEY="$(r spark1 'head -n 1 ~/code/blackfold-tp2/api-key')"; export BF_KEY
log "preflight ok"
STOPPED=0
stop_tf() { for h in spark1 spark2; do r $h "docker logs $N > $PD/last-$h.log 2>&1; docker rm -f $N >/dev/null 2>&1; true"; done; }
restore() {
  log "restore"
  kill "${WATCH:-}" 2>/dev/null
  stop_tf
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
T0=$(date +%s); envs="CTX=16384 TF_RESERVE_GIB=10.9 TF_SIDE=1 TF_PREFETCH=1"
r spark2 "$envs $NODE worker" >/dev/null || { log "worker start failed"; exit 1; }
r spark1 "$envs $NODE head" >/dev/null || { log "head start failed"; exit 1; }
ready=0
for i in $(seq 1 120); do
  h=$(curl -s -m 3 "$API/health" || true)
  if echo "$h" | grep -q '"status": *"ok"' && ! echo "$h" | grep -q '"warming": *true'; then ready=1; break; fi
  for n in spark1 spark2; do r $n "docker ps --format '{{.Names}}' | grep -qx $N" || { log "$N stopped on $n"; exit 1; }; done
  sleep 5
done
log "ready=$ready after $(( $(date +%s) - T0 )) s; $(tail -1 "$OUT/mem.log")"
[[ $ready == 1 ]] || exit 1
sleep 8
MODEL="$(curl -s -m 10 -H "Authorization: Bearer $BF_KEY" "$API/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
RECIPE="BLACKFOLD TP2 recipe d905cf0, TensorFold 6366d2c + patches 0001-0003, spark1 head + spark2, TF_DS_GRAPHS=1 TF_DS_HC_SIDE=1 TF_DS_L2_PREFETCH=1, context 16384, TENSORFOLD_MEMORY_RESERVE_GIB=10.9 (.scratch/tfknock/bf-node-knock.sh)"
COMMON=(--url "$API/v1" --model "$MODEL" --key-env BF_KEY --thinking off --ignore-busy --recipe "$RECIPE")
cd /opt/code/deepeval || exit 1
uv run localeval speed "${COMMON[@]}" --sizes 128,1024,4096 --gen 128 --reps 3 --label dsv41-blackfold-tf-speed-ll 2>&1 | tee "$OUT/le-speed.log" | tail -12 | tee -a "$OUT/steps.log"
uv run localeval quality "${COMMON[@]}" --dataset "$HERE/knock-rcp.jsonl" --judge none --max-tokens 160 --label dsv41-blackfold-tf-ldata-ll 2>&1 | tee "$OUT/le-ldata.log" | tail -25 | tee -a "$OUT/steps.log"
r spark1 "docker logs $N 2>&1 | grep -a 'tensorfold. done' | tail -8" > "$OUT/ldata.donelog"
log "done; $(tail -1 "$OUT/mem.log")"
