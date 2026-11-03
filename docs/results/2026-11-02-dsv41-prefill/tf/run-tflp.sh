#!/bin/bash
# run-tflp.sh: TensorFold BLACKFOLD (head spark1 + worker spark2) answers the prompts in LP_PROMPTS with 16 greedy tokens and the
# greedy text (64 tokens; NOTE: the chunk is a compile-time constant in TensorFold (`chunk_rows = 2048` in families/dsv41/model.zig), so TF_CHUNKS=512 only restarts the same 2048 build and measures run-to-run repeatability; TensorFold refuses logprobs: "logprobs are not supported by this model or backend", so the first attempt, job 8064, returned only errors), once at its served chunk (2048) and once at TF_DS_PREFILL_CHUNK=512 (TF_CHUNKS), for the chunk-dependence study
# (docs/results/2026-11-02-dsv41-prefill.md). The first prompt is asked twice (determinism). One fleetq job over both Sparks;
# vLLM always restored (trap). Adapted from run-tfprefill.sh.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-$HERE/out}; mkdir -p "$OUT"
API=http://100.120.88.104:18191
PD='$HOME/intarsia-dsv41-prefill/tf'
NODE='$HOME/intarsia-dsv41-prefill/tf/bf-node-prof.sh'
N=ds41-prof-pf
NS=/nsys/target-linux-sbsa-armv8/nsys
JOB=${FLEETQ_JOB:?run under fleetq run}
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$1" "export FLEETQ_JOB=$JOB; $2"; }
log() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
for h in spark1 spark2; do
  eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
  [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is -m $eo"; exit 1; }
  r $h 'docker ps -a --format "{{.Names}}" | grep -E "^(dsv41-1017-exl3-tf|ds41-native|ds41-prof-dd|ds41-knock-dd|ds41-prof-pf)$"' && { log "a TF container exists on $h"; exit 1; }
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
  for h in spark1 spark2; do r $h "docker logs $N > $PD/last-$h.log 2>&1; docker rm -f $N >/dev/null 2>&1; true"; scp -q $h:intarsia-dsv41-prefill/tf/last-$h.log "$OUT/" 2>/dev/null; done
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
req() { # promptfile name
  python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":open(sys.argv[2]).read(),"max_tokens":int(sys.argv[3]),"temperature":0}))' "$MODEL" "$1" "${TOK:-64}" > "$OUT/body-$2.json"
  local t=$(date +%s.%N)
  curl -s -m 300 "$API/v1/completions" -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d @"$OUT/body-$2.json" > "$OUT/lp-$2.json"
  python3 -c 'import json,sys,time; d=json.load(open(sys.argv[1])); print(sys.argv[3], "wall %.2f s" % (time.time()-float(sys.argv[2])), d.get("usage"), (d.get("error") or ""))' "$OUT/lp-$2.json" "$t" "$2" | tee -a "$OUT/steps.log"
}
LP=${LP_PROMPTS:?space-separated prompt files}
# a fresh container per TF_DS_PREFILL_CHUNK in TF_CHUNKS (default: the served 2048, then 512)
for ck in ${TF_CHUNKS:-default 512}; do
  if [[ $ck == default ]]; then CHUNKENV=""; else CHUNKENV="TF_CHUNK=$ck"; fi
  mkdir -p "$OUT/c$ck"; OUT0=$OUT; OUT=$OUT0/c$ck
  r spark2 "$CHUNKENV CTX=16384 TF_RESERVE_GIB=${TF_RESERVE_GIB:-10.9} $NODE worker" >/dev/null || { log "worker start failed"; exit 1; }
  r spark1 "$CHUNKENV CTX=16384 TF_RESERVE_GIB=${TF_RESERVE_GIB:-10.9} $NODE head" >/dev/null || { log "head start failed"; exit 1; }
  ready=0
  for i in $(seq 1 240); do
    h=$(curl -s -m 3 "$API/health" || true)
    if echo "$h" | grep -q '"status": *"ok"' && ! echo "$h" | grep -q '"warming": *true'; then ready=1; break; fi
    for n in spark1 spark2; do r $n "docker ps --format '{{.Names}}' | grep -qx $N" || { log "$N stopped on $n"; r $n "docker logs --tail 30 $N 2>&1" | tee -a "$OUT/steps.log"; exit 1; }; done
    sleep 5
  done
  log "chunk $ck ready=$ready; $(tail -1 "$OUT0/mem.log")"
  [[ $ready == 1 ]] || exit 1
  sleep 10
  KEY="$(r spark1 'head -n 1 ~/code/blackfold-tp2/api-key')"
  MODEL="$(curl -s -m 10 -H "Authorization: Bearer $KEY" "$API/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
  for f in $LP; do req "$f" "$(basename "$f" .txt)"; done
  req "$(echo $LP | cut -d' ' -f1)" "repeat-$(basename "$(echo $LP | cut -d' ' -f1)" .txt)"
  for h in spark1 spark2; do r $h "docker logs $N > $PD/last-$h-c$ck.log 2>&1; docker rm -f $N >/dev/null 2>&1; true"; scp -q $h:intarsia-dsv41-prefill/tf/last-$h-c$ck.log "$OUT0/" 2>/dev/null; done
  OUT=$OUT0
done
log "done; $(tail -1 "$OUT/mem.log")"
