#!/usr/bin/env bash
# dsubset CUDA gates on spark1 (35B NVFP4), one fleetq slot per call; PARTS picks what runs:
#   plan:   `plan show --json`, main's binary vs the lane's (knob unset): must be byte-identical; and the lane's with the knob set
#   ids:    seeded ids (ids.py: greedy + temperature 0.6 / 1.0 at 1, 2, 4 concurrent), drafting off vs the subset head:
#             ref   --spec off, --max-batch 2
#             s2    INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1 --spec mtp --spec-k 2 --mtp-draft-subset S65536, --max-batch 2
#             s1    --spec mtp --spec-k 2 --mtp-draft-subset S65536, --max-batch 1 (B = 1)
#   accept: acceptance per script, one stream (accept.py), arms F (full head), P65536, P131072, S32768, S65536 (ARMS overrides);
#           Q65536 = the 65536 prefix as a subset file (EXTRA): its counters must equal P65536's exactly (gather + id map are exact)
#   speed:  ABBA at 1 and 2 streams with the 2-stream opt-in set, prefix 131072 (P) vs subset 65536 (S), T = 1.0 seeded, 384 forced
#           tokens, thinking off, sets mixed en code zh (speed.py); VISITS overrides the order (default "P.1 S.1 S.2 P.2")
# Subset files: SUBDIR (local) is copied to ~/intarsia-dsubset/subsets/ first; arms name them sub-<rows>.txt.
# Run under: fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect <MIN> --label "dsubset: <parts>" -- env PARTS=".." scripts/dsubset/runs.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=${LOGNAME_:-runs}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SUBDIR=${SUBDIR:-$REPO/recipes/data/mtp-draft-subset}
SUB='~/intarsia-dsubset/subsets'
rsh "mkdir -p ~/intarsia-dsubset/subsets"
for f in "$SUBDIR"/*.txt; do scp -q -o BatchMode=yes "$f" "$HOST:intarsia-dsubset/subsets/sub-$(sed -n 's/^# rows //p' "$f").txt"; done
# EXTRA: more subset files copied under their own names (Q<rows> arms use prefix-<rows>.txt: the prefix as a subset file)
[[ -n "${EXTRA:-}" ]] && for f in "$EXTRA"/*.txt; do scp -q -o BatchMode=yes "$f" "$HOST:intarsia-dsubset/subsets/"; done
log "runs start [$PARTS]; vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p'); subsets: $(rsh "ls $SUB" | tr '\n' ' ')"
OPT="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"
head_args() { # head_args <arm>: the drafter head flags of an arm
  case $1 in
    F) echo "--mtp-draft-vocab 0" ;;
    P*) echo "--mtp-draft-vocab ${1#P}" ;;
    S*) echo "--mtp-draft-subset $SUB/sub-${1#S}.txt" ;;
    Q*) echo "--mtp-draft-subset $SUB/prefix-${1#Q}.txt" ;;
  esac
}
for part in ${PARTS:?PARTS}; do
  case $part in
    plan)
      P="plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json"
      rsh "~/intarsia-dsubset/bin/intarsia-main $P > ~/intarsia-dsubset/out/plan-main.json 2> ~/intarsia-dsubset/out/plan-main.err; echo exit \$?"
      rsh "~/intarsia-dsubset/bin/intarsia $P > ~/intarsia-dsubset/out/plan-lane.json 2> ~/intarsia-dsubset/out/plan-lane.err; echo exit \$?"
      rsh "~/intarsia-dsubset/bin/intarsia $P --spec mtp --mtp-draft-subset $SUB/sub-65536.txt > ~/intarsia-dsubset/out/plan-lane-subset.json 2> ~/intarsia-dsubset/out/plan-lane-subset.err; echo exit \$?"
      log "plan show json, main vs lane, knob unset: $(rsh "cmp ~/intarsia-dsubset/out/plan-main.json ~/intarsia-dsubset/out/plan-lane.json && echo IDENTICAL; wc -c < ~/intarsia-dsubset/out/plan-main.json" | tr '\n' ' ')"
      log "plan show, knob set (stderr lines naming the subset): $(rsh "grep -h 'draft-subset' ~/intarsia-dsubset/out/plan-lane-subset.err | cut -c1-200" | tr '\n' ';')"
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-dsubset/out/plan-*" "$OUT/plan/"
      ;;
    ids)
      for a in ${IDARMS:-ref s2 s1}; do
        case $a in
          ref) e="X=1"; args="--spec off --max-batch 2" ;;
          s2)  e="$OPT"; args="--spec mtp --spec-k 2 $(head_args S65536) --max-batch 2" ;;
          s1)  e="X=1"; args="--spec mtp --spec-k 2 $(head_args S65536) --max-batch 1" ;;
        esac
        if start_serve "ids-$a" "$e" $args; then
          python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$a.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids_compare.py" "$OUT" ${IDARMS:-ref s2 s1} 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
      ;;
    accept)
      for a in ${ARMS:-F P65536 P131072 S32768 S65536}; do
        if start_serve "acc-$a" "X=1" --spec mtp --spec-k 2 --max-batch 1 $(head_args $a); then
          log "draft-head lines of $a: $(rsh "grep -E 'draft-subset|draft-vocab|not applicable' ~/intarsia-dsubset/out/serve-acc-$a.log | cut -c1-200" | tr '\n' ';')"
          python3 "$S/accept.py" "$URL" guidance "$a" "$OUT" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
          finish_serve "acc-$a"
        fi
      done
      ;;
    speed)
      for v in ${VISITS:-P.1 S.1 S.2 P.2}; do
        arm=${v%%.*}; case $arm in P) h=$(head_args P131072) ;; S) h=$(head_args S65536) ;; *) h=$(head_args $arm) ;; esac
        if start_serve "sp-$v" "$OPT" --spec mtp --spec-k 2 --max-batch 2 $h; then
          log "vLLM requests running before $v: $(vllm_busy)"
          for set in ${SETS:-mixed en code zh}; do
            ( cd /opt/code/deepeval && uv run python "$S/speed.py" --url "$URL" --model guidance --arm "$v" --streams "${STREAMS:-1,2}" --gen "${GEN:-384}" --temperature "${TEMP:-1.0}" --thinking off --set "$set" --out "$OUT/speed" --metrics "http://$HN:$PORT/metrics" ) 2>&1 | tee -a "$LOG"
          done
          log "vLLM requests running after $v: $(vllm_busy)"
          finish_serve "sp-$v"
        fi
      done
      ;;
  esac
done
