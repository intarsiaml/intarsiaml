#!/usr/bin/env bash
# flipprep speed gate: ABBA of main's defaults (A) against the flip's defaults (F) on the CUDA 35B NVFP4 (spark2), --max-batch 2, fresh server per visit.
# One visit = one server; for each temperature in TEMPS (0 = greedy, 1.0) and each set in SETS (en code en-nt cjk-nt mixed-nt; -nt = thinking off)
# speed.py at 1 and 2 streams. Results: $OUT/t<temp>/speed-<visit>-<set>[-nt].json. ARMS="A.a F.a F.b A.b".
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 60 --label "flipprep: speed" -- env ARMS="A.a F.a F.b A.b" scripts/flipprep/speed.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-speed}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"; source "$S/arms.sh"
log "speed start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
for v in ${ARMS:?ARMS}; do
  arm_setup "$v" || { log "unknown arm $v"; continue; }
  if start_serve "$v" "$ENVS" --max-batch 2 $ARGS; then
    log "draft-head fallback lines in $v (must be 0): $(rsh "grep -c 'not applicable to this lm_head' ~/intarsia-flipprep/out/serve-$v.log")"
    log "vLLM requests running before $v: $(vllm_busy)"
    for temp in ${TEMPS:-0 1.0}; do
      for set in ${SETS:-en code en-nt cjk-nt mixed-nt}; do
        th=default; [[ $set == *-nt ]] && th=off
        ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$v" --streams "${STREAMS:-1,2}" --gen "${GEN:-384}" --temperature "$temp" --thinking $th --set "${set%-nt}" --out "$OUT/t$temp" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
      done
    done
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v"
  fi
done
