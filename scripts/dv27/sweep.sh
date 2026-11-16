#!/usr/bin/env bash
# dv27: draft-vocabulary sweep on the 27B Quark MXFP4 (CUDA, spark1), fresh server per visit.
# Visit names V<N>[.<tag>] (D[.<tag>] = recipe default, no spec flags; smoke only): V<N> = `--spec mtp --spec-k 2 --mtp-draft-vocab N` (N = 0: full head; 65536 is the recipe default, passed explicitly).
# MODE=s1 (default): the shipped single-stream shape (plain `serve`, no --max-batch), speed.py at 1 stream.
# MODE=b2: `--max-batch 2` with the mtp2s opt-in env (INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1), speed.py at 2 streams (or STREAMS).
# MODE=b2n: `--max-batch 2` with INTARSIA_BATCHED_MTP=2 only (no batched verify, no verify top-k).
# CELLS: space-separated set:thinking:temperature (thinking default|off). GEN tokens (default 256). OUT raw dir.
# Run under: fleetq run --host spark1 --quiet gpu,cpu --mem 30 --expect MIN --label "dv27: ..." -- env ARMS="V65536.a V98304.a" scripts/dv27/sweep.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
MODE=${MODE:-s1}
LOGNAME_=${LOGNAME_:-sweep-$MODE}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CELLS=${CELLS:-"en:default:1 code:default:1 cjk:default:1 cjk:off:1 mixed:default:1 en:default:0 code:default:0 cjk:off:0"}
scp -q -o BatchMode=yes "$S/memwatch.sh" $HOST:intarsia-dv27/memwatch.sh  # the memory watch must exist before the first serve
log "sweep $MODE start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p'); binary: $(rsh "$BIN --version | head -1")"
for v in ${ARMS:?ARMS}; do
  arm=${v%%.*}; n=${arm#V}
  if [[ $MODE == b2n ]]; then envs="INTARSIA_BATCHED_MTP=2"; mb="--max-batch 2"; streams=${STREAMS:-2}   # batched MTP without batched verify (per-sequence verify rows, m = 3)
  elif [[ $MODE == b2 ]]; then envs="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"; mb="--max-batch 2"; streams=${STREAMS:-2}
  else envs="X=1"; mb=""; streams=${STREAMS:-1}; fi
  # arm D (smoke only): the recipe default with no spec flags at all
  if [[ $arm == D ]]; then sargs=""; else sargs="--spec mtp --spec-k 2 --mtp-draft-vocab $n"; fi
  if start_serve "$v-$MODE" "$envs" $mb $sargs; then
    log "draft-head fallback lines in the server log of $v (must be 0): $(rsh "grep -c 'not applicable to this lm_head' ~/intarsia-dv27/out/serve-$v-$MODE.log")"
    log "vLLM requests running before $v: $(vllm_busy)"
    first=1
    for cell in ${CELLS}; do
      IFS=: read -r set think temp <<<"$cell"
      ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$v-$MODE" --streams "$streams" --gen "${GEN:-256}" --temperature "$temp" --thinking "$think" --set "$set" --warmup $first --out "$OUT" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
      first=0
    done
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v-$MODE"
  fi
done
