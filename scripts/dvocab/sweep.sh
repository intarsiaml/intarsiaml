#!/usr/bin/env bash
# dvocab: draft-vocabulary sweep of the mtp2s opt-in set on the CUDA 35B NVFP4 (spark2), --max-batch 2, fresh server per visit.
# One visit = one server, then speed.py for every set in SETS (en code cjk mixed), at 1 and 2 streams (STREAMS).
# Visit names: <ARM>[.<tag>]; ARM A = main's defaults, P<N> = INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1
# --spec mtp --spec-k 2 --mtp-draft-vocab N (N = 0: full head).  Other knobs: TEMP (default 1.0), GEN (384), OUT (raw dir).
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 30 --label "dvocab: ..." -- scripts/dvocab/sweep.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-sweep}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "sweep start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p'); binary: $(rsh "$BIN --version | head -1")"
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
    for set in ${SETS:-en code cjk mixed}; do
      ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$v" --streams "${STREAMS:-1,2}" --gen "${GEN:-384}" --temperature "${TEMP:-1.0}" --set "$set" --out "$OUT" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
    done
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v"
  fi
done
