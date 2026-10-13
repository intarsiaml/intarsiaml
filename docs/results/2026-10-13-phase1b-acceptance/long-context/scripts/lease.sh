#!/usr/bin/env bash
# lease.sh <label> <expect-min> <remote script on spark1> [env assignments...]: exclusive lease taking guidance down; trap restores vLLM before release.
set -u
LABEL=$1; EXPECT=$2; REMOTE=$3; shift 3
HOST=spark1; VLLM_HOST=100.120.88.104; C=missioncntrl-guidance
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
LEASE=""; STOPPED=0
log(){ printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
cleanup(){
  trap - EXIT INT TERM; set +e
  ssh -n -o BatchMode=yes $HOST "FLEETQ_JOB=$LEASE docker kill dbg-golden-8b dbg-golden-8b-long >/dev/null 2>&1; FLEETQ_JOB=$LEASE pkill -TERM -f '[i]ntarsia-dbg/(src|mut)/target'; sleep 3; FLEETQ_JOB=$LEASE pkill -KILL -f '[i]ntarsia-dbg/(src|mut)/target'; true"
  if [ $STOPPED = 1 ]; then
    log "docker start $C"; ssh -n -o BatchMode=yes $HOST docker start $C >/dev/null
    for i in $(seq 1 180); do
      if curl -sf --max-time 5 http://$VLLM_HOST:8000/v1/models >/dev/null 2>&1; then log "vLLM answers"; STOPPED=0; break; fi; sleep 5
    done
    [ $STOPPED = 1 ] && log "WARNING: vLLM did not answer within 15 min"
  fi
  [ -n "$LEASE" ] && { log "release $LEASE"; fleetq release $LEASE >/dev/null; }
  log "done"
}
trap cleanup EXIT INT TERM
LEASE=$(fleetq acquire --host $HOST --exclusive --takes-down guidance --mem 100 --expect $EXPECT --label "$LABEL" --wait | grep -Eo '[0-9]+' | tail -1)
[ -n "$LEASE" ] || { log "no lease"; exit 1; }
export FLEETQ_JOB=$LEASE
log "lease $LEASE; docker stop $C"
STOPPED=1
ssh -n -o BatchMode=yes $HOST docker stop $C >/dev/null
for i in $(seq 1 60); do
  gb=$(( $(ssh -n -o BatchMode=yes $HOST "awk '/MemAvailable/ {print \$2}' /proc/meminfo") / 1024 / 1024 ))
  [ $gb -ge 70 ] && { log "memory freed: $gb GB"; break; }; sleep 5
done
[ $gb -ge 70 ] || { log "memory did not free"; exit 1; }
T0=$(date +%s)
ssh -n -o BatchMode=yes $HOST "$RP env $* bash $REMOTE"
log "remote script exit $?; vLLM was down $(( $(date +%s) - T0 )) s so far"
