#!/usr/bin/env bash
# driver.sh <jobfile> <expect-min> : exclusive fleetq lease taking guidance down, run remote.sh, restore vLLM in a trap.
set -uo pipefail
HOST=spark1; CONT=missioncntrl-guidance; VURL=http://100.120.88.104:8000/v1/models
D=/opt/code/intarsiaml/target/bisect-build; JOBS=$1; EXPECT=${2:-30}
SSH="ssh -o BatchMode=yes -o ServerAliveInterval=30 $HOST"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
LEASE=""; STOPPED=0
cleanup() {
  trap - EXIT INT TERM; set +e
  log "cleanup: kill our processes on $HOST"
  $SSH "pkill -TERM -f '[i]ntarsia-acc/(new|old)/target/acc/release/'; pkill -f '[r]emote.sh'; for i in \$(seq 1 20); do pgrep -f '[i]ntarsia-acc/(new|old)/target/acc/release/' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]ntarsia-acc/(new|old)/target/acc/release/'; sleep 3; true"
  scp -q -r $HOST:intarsia-acc/bis/. $D/bis-out/ 2>/dev/null
  if [[ $STOPPED == 1 ]]; then
    log "docker start $CONT"; FLEETQ_JOB=$LEASE ssh -n -o BatchMode=yes $HOST docker start $CONT >/dev/null
    for i in $(seq 1 180); do curl -sf -m 5 $VURL >/dev/null 2>&1 && { log "vLLM answers /v1/models"; break; }; sleep 5; done
  fi
  [[ -n "$LEASE" ]] && { log "release $LEASE"; fleetq release $LEASE >/dev/null; }
  log "done"; exit 0
}
trap cleanup EXIT INT TERM
mkdir -p $D/bis-out
LEASE=$(fleetq acquire --host $HOST --exclusive --takes-down guidance --mem 100 --expect $EXPECT --label "intarsiaml accuracy bisect (exclusive, vLLM down)" --wait | grep -Eo '[0-9]+' | tail -1)
[[ -n "$LEASE" ]] || { log "no lease id"; exit 1; }
log "lease $LEASE"; export FLEETQ_JOB=$LEASE
STOPPED=1; T0=$(date +%s)
ssh -n -o BatchMode=yes $HOST docker stop $CONT >/dev/null
for i in $(seq 1 60); do gb=$(( $($SSH "awk '/MemAvailable/ {print \$2}' /proc/meminfo") / 1024 / 1024 )); (( gb >= 70 )) && break; sleep 5; done
log "memory ${gb} GB available; running jobs"
(( gb >= 70 )) || { log "memory did not free"; exit 1; }
scp -q $D/remote.sh $JOBS $HOST:intarsia-acc/ && $SSH "bash intarsia-acc/remote.sh intarsia-acc/$(basename $JOBS)"
log "jobs finished after $(( ($(date +%s)-T0)/60 )) min"
