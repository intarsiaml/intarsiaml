#!/usr/bin/env bash
# dvocab: sweep.sh with thinking OFF (the forced tokens are the answer, so a Chinese prompt yields Chinese tokens). Same visits and arms.
# THINK=off is passed to speed.py; results land in speed-<visit>-<set>-nt.json. Run under fleetq like sweep.sh.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-sweep-nt}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "sweep (thinking off) start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
OPT="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"
for v in ${ARMS:?ARMS}; do
  arm=${v%%.*}
  case $arm in
    A)  envs="X=1"; args="" ;;
    P*) envs="$OPT"; args="--spec mtp --spec-k 2 --mtp-draft-vocab ${arm#P}" ;;
    *)  log "unknown arm $arm"; continue ;;
  esac
  if start_serve "$v" "$envs" --max-batch 2 $args; then
    log "draft-head fallback lines in the server log of $v (must be 0): $(rsh "grep -c 'not applicable to this lm_head' ~/intarsia-dvocab/out/serve-$v.log")"
    log "vLLM requests running before $v: $(vllm_busy)"
    for set in ${SETS:-cjk mixed en code}; do
      ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$v" --streams "${STREAMS:-1,2}" --gen "${GEN:-384}" --temperature "${TEMP:-1.0}" --thinking off --set "$set" --out "$OUT" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
    done
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v"
  fi
done
