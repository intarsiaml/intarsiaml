#!/usr/bin/env bash
# mdraft 2026-11-16 speed ABBA of the batched MTP draft on spark2 (one fleetq slot; lib.sh helpers; itl2s's load client and tables):
#   B = the base binary (main's engine, per-sequence drafts), F = the new binary at its defaults; both with INTARSIA_CB_TRACE=1
#   (the Emit lines give the server-side iteration time: prepare = drafts, verify pass), fresh server per visit, --max-batch 2.
#   ARMS "B.a F.a F.b B.b"; every visit: 2-stream cells CELLS2, then 1-stream cells CELLS1 (cell = set:temperature:thinking, as itl2s).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 60 --label "mdraft: ABBA" -- scripts/mdraft/abba.sh
#   python3 scripts/itl2s/analyze.py docs/results/2026-11-16-mdraft/raw/abba --base B
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OUT=${OUT:-$REPO/docs/results/2026-11-16-mdraft/raw/abba}; mkdir -p "$OUT"
ARMS=${ARMS:-B.a F.a F.b B.b}
CELLS2=${CELLS2:-"en:0:default code:0:default en:0:off cjk:0:off mixed:0:off en:1.0:default code:1.0:default en:1.0:off cjk:1.0:off mixed:1.0:off"}
CELLS1=${CELLS1:-"en:0:default code:0:default en:0:off en:1.0:off"}
LOGNAME_=${LOGNAME_:-abba}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
L='~/intarsia-mdraft'
log "start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models) running $(vllm_busy); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
run_cells() { # run_cells <visit> <streams> "<cells>"
  local v=$1 n=$2 c set temp th
  for c in $3; do
    set=${c%%:*}; c=${c#*:}; temp=${c%%:*}; th=${c#*:}
    rsh "python3 $L/src/scripts/itl2s/itl.py --url http://localhost:$PORT/v1 --model guidance --visit $v --set $set --temperature $temp --thinking $th --streams $n --rounds ${ROUNDS:-2} --gen ${GEN:-384} --out $L/out/cells" 2>&1 | tee -a "$LOG"
  done
}
for v in $ARMS; do
  case ${v%%.*} in
    B) BIN='~/intarsia-mdraft/bin/base' ;;
    F) BIN='~/intarsia-mdraft/bin/new' ;;
    F0) BIN='~/intarsia-mdraft/bin/new'; ENVX="INTARSIA_BATCHED_DRAFT=0" ;;
    *) log "unknown arm $v"; continue ;;
  esac
  if start_serve "$v" "INTARSIA_CB_TRACE=1 ${ENVX:-}" --max-batch 2; then
    curl -s "http://$HN:$PORT/v1/intarsia/plan" > "$OUT/plan-$v.json"
    log "vLLM requests running before $v: $(vllm_busy)"
    rsh "mkdir -p $L/out/cells; rm -f $L/out/cells/itl-$v-*; nohup bash $L/src/scripts/itl2s/vsample.sh $L/out/vllm-$v.txt > /dev/null 2>&1 < /dev/null & echo \$! > $L/out/vllm-$v.pid"
    # itl.py warms up (2 streams, the cell's temperature) before every cell
    run_cells "$v" 2 "$CELLS2"
    run_cells "$v" 1 "$CELLS1"
    rsh "kill \$(cat $L/out/vllm-$v.pid) 2>/dev/null; gzip -c $L/out/vllm-$v.txt" > "$OUT/vllm-$v.txt.gz"
    mkdir -p "$OUT/cells"; ssh -o BatchMode=yes "$HOST" "cd $L/out/cells && tar cf - itl-$v-*.json" | tar xf - -C "$OUT/cells"
    log "vLLM requests running after $v: $(vllm_busy)"
    finish_serve "$v"; gzip -f "$OUT/serve-$v.log"
  fi
  ENVX=""
done
