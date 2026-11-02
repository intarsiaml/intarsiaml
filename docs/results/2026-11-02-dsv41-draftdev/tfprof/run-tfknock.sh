#!/bin/bash
# TensorFold BLACKFOLD knockouts, full model, no nsys: for each config restart head (spark1) + worker (spark2) with
# TF_SIDE / TF_PREFETCH set, send the same requests (4 prompts of ~1K tokens, 160 tokens, temperature 0; 2 warm-up + 8 measured),
# report decode ms a round (client stream time first token -> last token / rounds from the head log), tokens a round, tok/s.
# One fleetq job over both Sparks (takes both vLLMs down, restored in the trap). Reserve 10.9, watcher kills ours under 2.8 GiB.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-$HERE/out}; mkdir -p "$OUT"
API=http://100.120.88.104:18191
NODE='$HOME/intarsia-dsv41-draftdev/tfprof/bf-node-knock.sh'
PD='$HOME/intarsia-dsv41-draftdev/tfprof'
N=ds41-knock-dd
JOB=${FLEETQ_JOB:?run under fleetq run}
CONFIGS=${CONFIGS:-"base:1:1 nopf:1:0 noside:0:1 neither:0:0"}   # name:TF_SIDE:TF_PREFETCH
MAXTOK=${MAXTOK:-160}
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$JOB; $2"; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
for h in spark1 spark2; do
  eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
  [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is -m $eo"; exit 1; }
  r $h 'docker ps -a --format "{{.Names}}" | grep -E "^(dsv41-1017-exl3-tf|ds41-native|ds41-prof-dd|ds41-knock-dd)$"' && { log "a TF container exists on $h"; exit 1; }
  r $h 'ss -ltn | grep -E ":18191 "' && { log "port 18191 busy on $h"; exit 1; }
  r $h 'B=$HOME/code/blackfold-tp2; set -e; ls $B/cache/rank-cache/rank*of2-*.bin $B/cache/dsv41_token_map.json $B/engine/out/zig-out/bin/tf-dsv41-lanes $B/engine/out/zig-out/native/bin/tensorfold-native $B/chat_template.native.jinja >/dev/null' || { log "missing files on $h"; exit 1; }
  r $h "mkdir -p $PD"
  ssh -o BatchMode=yes $h "cat > $PD/bf-node-knock.sh; chmod +x $PD/bf-node-knock.sh" < "$HERE/bf-node-knock.sh"
done
KEY="$(r spark1 'head -n 1 ~/code/blackfold-tp2/api-key')"
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
sleep 5; log "memory: $(tail -1 "$OUT/mem.log")"
for cfg in $CONFIGS; do
  IFS=: read name side pf <<< "$cfg"
  log "== config $name: TF_DS_HC_SIDE=$side TF_DS_L2_PREFETCH=$pf"
  T0=$(date +%s)
  envs="CTX=16384 TF_RESERVE_GIB=10.9 TF_SIDE=$side TF_PREFETCH=$pf"
  r spark2 "$envs $NODE worker" >/dev/null || { log "$name: worker start failed"; stop_tf; continue; }
  r spark1 "$envs $NODE head" >/dev/null || { log "$name: head start failed"; stop_tf; continue; }
  ready=0
  for i in $(seq 1 120); do
    h=$(curl -s -m 3 "$API/health" || true)
    if echo "$h" | grep -q '"status": *"ok"' && ! echo "$h" | grep -q '"warming": *true'; then ready=1; break; fi
    dead=0; for n in spark1 spark2; do r $n "docker ps --format '{{.Names}}' | grep -qx $N" || { log "$name: $N stopped on $n"; r $n "docker logs --tail 12 $N 2>&1 | cut -c1-300" | tee -a "$OUT/steps.log"; dead=1; }; done
    [[ $dead == 1 ]] && break
    sleep 5
  done
  log "$name: ready=$ready after $(( $(date +%s) - T0 )) s; $(tail -1 "$OUT/mem.log")"
  if [[ $ready != 1 ]]; then stop_tf; sleep 10; continue; fi
  sleep 8
  MODEL="$(curl -s -m 10 -H "Authorization: Bearer $KEY" "$API/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
  for p in 0 1; do python3 "$HERE/knock_client.py" $API "$KEY" "$MODEL" "$HERE/prompt$p.txt" 32 > /dev/null; done   # warm-up (graph capture, caches)
  : > "$OUT/$name.jsonl"
  for rep in 1 2; do for p in 0 1 2 3; do
    python3 "$HERE/knock_client.py" $API "$KEY" "$MODEL" "$HERE/prompt$p.txt" $MAXTOK >> "$OUT/$name.jsonl" || log "$name: request failed (prompt$p)"
  done; done
  log "$name: mem now $(tail -1 "$OUT/mem.log")"
  r spark1 "docker logs $N 2>&1 | grep -a 'tensorfold. done' | tail -8" > "$OUT/$name.donelog"
  sed 's/^/    /' "$OUT/$name.donelog" | cut -c1-200 | tee -a "$OUT/steps.log"
  python3 "$HERE/summarize.py" "$OUT/$name.jsonl" "$OUT/$name.donelog" | tee -a "$OUT/steps.log"
  stop_tf; sleep 12
done
log "done; $(tail -1 "$OUT/mem.log")"
