#!/usr/bin/env bash
# mtp2s speed A/B, greedy (temperature 0), --max-batch 2, fresh server per visit, spark2's vLLM up (idle checked, not enforced).
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 40 --label "mtp2s: ..." -- scripts/mtp2s/ab.sh
# Arms (ARMS env: space-separated visit names; a visit's arm is its name without the trailing digit):
#   A    main defaults (MTP k2 drafts while one sequence decodes; plain batched steps at 2)
#   D    A + --mtp-draft-vocab 65536 (reduced draft head; verify keeps the full head)
#   V    INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 (batched verify at 2 streams, k2)
#   VD   V + --mtp-draft-vocab 65536
#   VD1  VD at --spec-k 1 (visit names VD1a, VD1b, ...)
#   X / XD / XV / XVD: any extra arm via the ENV_<arm> / ARGS_<arm> variables
# STREAMS (default 1,2), GEN (default 384), SET (en|cjk).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-ab}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "ab start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
DV="--spec mtp --spec-k 2 --mtp-draft-vocab 65536"
ENV_A="X=1";                                              ARGS_A=""
ENV_D="X=1";                                              ARGS_D="$DV"
ENV_V="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1"; ARGS_V=""
ENV_VD="$ENV_V";                                          ARGS_VD="$DV"
ENV_VD1="$ENV_V";                                         ARGS_VD1="--spec mtp --spec-k 1 --mtp-draft-vocab 65536"
visit() { # visit <name> <arm>
  local name=$1 arm=$2 e a
  e="ENV_$arm"; a="ARGS_$arm"
  if start_serve "$name" "${!e}" --max-batch 2 ${!a}; then
    log "vLLM requests running before $name: $(vllm_busy)"
    ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$name" --streams "${STREAMS:-1,2}" --gen "${GEN:-384}" --temperature 0 --set "${SET:-en}" --out "$OUT" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
    log "vLLM requests running after $name: $(vllm_busy)"
    finish_serve "$name"
  fi
}
for v in ${ARMS:-A1 D1 V1 VD1 VD2 V2 D2 A2}; do
  case $v in
    VD1[a-z]) visit "$v" VD1 ;;
    *) visit "$v" "${v%[0-9]}" ;;
  esac
done
