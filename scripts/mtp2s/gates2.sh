#!/usr/bin/env bash
# mtp2s 2026-11-16 gates of the fused GDN verify launches on spark2 (one fleetq slot; lib.sh helpers, one server at a time on :8194):
#   ids:   greedy ids (ids3.py: short + two 4k prompts, 256 tokens, alone and in the three concurrent pairs), --max-batch 2, arms
#            off   the fuse binary, --spec off                              (the reference)
#            base  the base binary (main's engine), defaults
#            f1    the fuse binary, defaults (fused verify on)
#            f0    the fuse binary, INTARSIA_CUDA_GDN_VERIFY_FUSE=0
#          every request of every arm must equal the reference's solo ids of the same prompt; the served /v1/intarsia/plan of base, f1, f0 byte-identical
#   plan:  `plan show --json` of base vs fuse (knob unset and =0)
# Run under: fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 30 --label "mtp2s: gates" -- scripts/mtp2s/gates2.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export OUT=${OUT:-$REPO/docs/results/2026-11-16-mtp2s/raw/gates}; mkdir -p "$OUT"
ARMS=${ARMS:-off base f1 f0}
LOGNAME_=gates; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "gates start; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models) running $(vllm_busy); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p')"
FUSE='~/intarsia-mtp2s/bin/fuse'; BASE='~/intarsia-mtp2s/bin/base'
head -c 14000 "$REPO/tools/prompts/corpus_docs.txt" > "$OUT/prompt-4k.txt"
for part in ${PARTS:-ids plan}; do
  case $part in
    ids)
      for a in $ARMS; do
        case $a in
          off)  BIN=$FUSE; e="X=1"; args="--spec off" ;;
          base) BIN=$BASE; e="X=1"; args="" ;;
          f1)   BIN=$FUSE; e="X=1"; args="" ;;
          f0)   BIN=$FUSE; e="INTARSIA_CUDA_GDN_VERIFY_FUSE=0"; args="" ;;
        esac
        if start_serve "ids-$a" "$e" --max-batch 2 $args; then
          curl -s "http://$HN:$PORT/v1/intarsia/plan" > "$OUT/plan-served-$a.json"
          python3 "$S/ids3.py" "$URL" "$OUT/ids3-$a.json" "$OUT/prompt-4k.txt" 256 guidance 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids3_compare.py" "$OUT" ${ARMS%% *} ${ARMS#* } 2>&1 | tee -a "$LOG" "$OUT/ids3-compare.txt"
      for a in f1 f0; do log "served plan base vs $a: $(cmp "$OUT/plan-served-base.json" "$OUT/plan-served-$a.json" && echo IDENTICAL; wc -c < "$OUT/plan-served-base.json" | tr -d ' ')"; done
      ;;
    plan)
      for spec in "base:$BASE:X=1" "fuse:$FUSE:X=1" "fuse0:$FUSE:INTARSIA_CUDA_GDN_VERIFY_FUSE=0"; do
        n=${spec%%:*}; r=${spec#*:}; b=${r%%:*}; e=${r#*:}
        rsh "env $e $b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json > ~/intarsia-mtp2s/out/plan-$n.json 2> ~/intarsia-mtp2s/out/plan-$n.err; echo exit \$?"
      done
      log "plan show json base vs fuse: $(rsh "cmp ~/intarsia-mtp2s/out/plan-base.json ~/intarsia-mtp2s/out/plan-fuse.json && echo IDENTICAL; wc -c < ~/intarsia-mtp2s/out/plan-base.json" | tr '\n' ' ')"
      log "plan show json base vs fuse =0: $(rsh "cmp ~/intarsia-mtp2s/out/plan-base.json ~/intarsia-mtp2s/out/plan-fuse0.json && echo IDENTICAL" | tr '\n' ' ')"
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-mtp2s/out/plan-*" "$OUT/plan/"
      ;;
  esac
done
