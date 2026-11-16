#!/usr/bin/env bash
# mhead 2026-11-16 gates of the batched MTP draft on spark2 (one fleetq slot; lib.sh helpers, one server at a time on :8195):
#   ids:   greedy ids (ids3.py: short + two 4k prompts, 256 tokens, alone and in the three concurrent pairs), --max-batch 2, arms
#            off   the new binary, --spec off                                  (the reference)
#            base  the base binary (main's engine), defaults
#            n1    the new binary, defaults (INTARSIA_BATCHED_HEAD unset)
#            n0    the new binary, INTARSIA_BATCHED_HEAD=0
#            chk   the new binary, INTARSIA_BATCHED_HEAD_CHECK=1 (every batched chained head pass is also run per sequence and compared bitwise)
#          every request of every arm must equal the reference's solo ids of the same prompt; the served /v1/intarsia/plan of base, n1, n0
#          byte-identical; base / n1 / n0 run with INTARSIA_SPEC_TRACE=1 INTARSIA_CB_TRACE=1 so spec_cmp.py can compare the per-pass
#          (position, drafted, accepted) lines of the arms and count the batched drafts
#   plan:  `plan show --json` of base vs new (knob unset and =0)
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 30 --label "mhead: gates" -- scripts/mhead/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OUT=${OUT:-$REPO/docs/results/2026-11-16-mhead/raw/gates}; mkdir -p "$OUT"
ARMS=${ARMS:-off base n1 n0 chk}
LOGNAME_=gates; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "gates start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models) running $(vllm_busy); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
NEW='~/intarsia-mhead/bin/new'; BASE='~/intarsia-mhead/bin/base'
TR="INTARSIA_SPEC_TRACE=1 INTARSIA_CB_TRACE=1"
head -c 14000 "$REPO/tools/prompts/corpus_docs.txt" > "$OUT/prompt-4k.txt"
for part in ${PARTS:-ids plan}; do
  case $part in
    ids)
      for a in $ARMS; do
        case $a in
          off)  BIN=$NEW;  e="X=1"; args="--spec off" ;;
          base*) BIN=$BASE; e="$TR"; args="" ;;
          n1*)   BIN=$NEW;  e="$TR"; args="" ;;
          chk*)  BIN=$NEW;  e="$TR INTARSIA_BATCHED_HEAD_CHECK=1"; args="" ;;
          n0*)   BIN=$NEW;  e="$TR INTARSIA_BATCHED_HEAD=0"; args="" ;;
        esac
        if start_serve "ids-$a" "$e" --max-batch 2 $args; then
          curl -s "http://$HN:$PORT/v1/intarsia/plan" > "$OUT/plan-served-$a.json"
          python3 "$S/ids3.py" "$URL" "$OUT/ids3-$a.json" "$OUT/prompt-4k.txt" 256 guidance 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids3_compare.py" "$OUT" ${ARMS%% *} ${ARMS#* } 2>&1 | tee -a "$LOG" "$OUT/ids3-compare.txt"
      set -- $ARMS; shift   # the arms after the reference: the first of them is the spec_cmp reference
      for a in "${@:2}"; do log "served plan $1 vs $a: $(cmp "$OUT/plan-served-$1.json" "$OUT/plan-served-$a.json" && echo IDENTICAL; wc -c < "$OUT/plan-served-$1.json" | tr -d ' ')"; done
      python3 "$S/spec_cmp.py" "$OUT" "$@" 2>&1 | tee -a "$LOG" "$OUT/spec-compare.txt"
      ;;
    plan)
      for spec in "base:$BASE:X=1" "new:$NEW:X=1" "new0:$NEW:INTARSIA_BATCHED_HEAD=0"; do
        n=${spec%%:*}; r=${spec#*:}; b=${r%%:*}; e=${r#*:}
        rsh "env $e $b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json > ~/intarsia-mhead/out/plan-$n.json 2> ~/intarsia-mhead/out/plan-$n.err; echo exit \$?"
      done
      log "plan show json base vs new: $(rsh "cmp ~/intarsia-mhead/out/plan-base.json ~/intarsia-mhead/out/plan-new.json && echo IDENTICAL; wc -c < ~/intarsia-mhead/out/plan-base.json" | tr '\n' ' ')"
      log "plan show json base vs new =0: $(rsh "cmp ~/intarsia-mhead/out/plan-base.json ~/intarsia-mhead/out/plan-new0.json && echo IDENTICAL" | tr '\n' ' ')"
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-mhead/out/plan-*" "$OUT/plan/"
      ;;
  esac
done
