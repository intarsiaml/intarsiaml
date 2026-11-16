#!/usr/bin/env bash
# mx6 CUDA gates (one fleetq slot). MODEL=27|35, HOST=spark1|spark2. PARTS (default "plan ids"):
#   plan   `plan show --json` of origin/main's binary vs the mx6 binary, with the knobs off (bare; --max-batch 2; batched MTP without batched verify): must be
#          byte-identical; and with the opt-in set on (reported: the plan grows by the batched-verify shapes)
#   repro  the main binary with the opt-in set: serves, then the first request fails with the `no kernel ... m=6` engine failure (the 27B)
#   ids    ids.py against servers R (--spec off), M (mx6, opt-in set), [MM (main, opt-in set; the 35B only)], --max-batch 2: greedy and seeded T = 1.0, B = 1, 2
# Run under: fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 50 --label "mx6: gates 27B" -- env MODEL=27 HOST=spark1 scripts/mx6/gates.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
LOGNAME_=gates-${PARTS_TAG:-all}; source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rsh "test -x $B/intarsia-main && test -x $B/intarsia-mx6" || { echo "binaries missing in $B on $HOST (scripts/mx6/build.sh has not finished)"; exit 3; }
scp -q -o BatchMode=yes "$S/memwatch.sh" $HOST:intarsia-mx6/memwatch.sh
log "gates start ($TAG on $HOST); vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:$VPORT/v1/models); earlyoom: $(rsh 'systemctl is-active earlyoom'); $(rsh 'free -g | sed -n 2p'); binaries: $(rsh "$B/intarsia-main --version | head -1; $B/intarsia-mx6 --version | head -1" | tr '\n' ' ')"
for part in ${PARTS:-plan ids}; do
  case $part in
    plan)
      OPT="INTARSIA_BATCHED_MTP=2 INTARSIA_BATCHED_VERIFY=1 INTARSIA_VERIFY_TOPK=1"
      # (no associative arrays: this runs under the Mac's bash 3.2)
      cfg() { case $1 in bare) echo "X=1|" ;; mb2) echo "X=1|--max-batch 2" ;; n) echo "INTARSIA_BATCHED_MTP=2|--max-batch 2 --spec mtp --spec-k 2" ;; on) echo "$OPT|--max-batch 2 --spec mtp --spec-k 2" ;; esac; }
      for c in ${CFGS:-bare}; do   # `plan show` takes no --max-batch / --spec (exit 2, empty files: a vacuous cmp), so only the bare plan is compared here; the served plans below cover the knobs
        v=$(cfg $c); e=${v%%|*}; a=${v#*|}
        for b in main mx6; do
          rsh "env $e $B/intarsia-$b plan show --backend cuda --model $MP --max-seq $MS --mem-budget $MB $a --json > $L/out/plan-$TAG-$c-$b.json 2> $L/out/plan-$TAG-$c-$b.err; echo plan $c $b exit \$?" | tee -a "$LOG"
        done
        log "plan show json $c ($TAG, cuda; env [$e] args [$a]): main vs mx6: $(rsh "cmp $L/out/plan-$TAG-$c-main.json $L/out/plan-$TAG-$c-mx6.json && echo IDENTICAL; wc -c < $L/out/plan-$TAG-$c-main.json; wc -c < $L/out/plan-$TAG-$c-mx6.json" 2>&1 | tr '\n' ' ')"
      done
      mkdir -p "$OUT/plan"; scp -q -o BatchMode=yes "$HOST:intarsia-mx6/out/plan-$TAG-*" "$OUT/plan/"
      ;;
    repro)
      arm_setup MM
      if start_serve "repro-main" "$ENVS" --max-batch 2 $ARGS; then
        log "first request on main with the opt-in set: HTTP $(curl -s -m 60 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{"model":"guidance","prompt":"Hello, my name is","max_tokens":16,"temperature":0}' http://$HN:$PORT/v1/completions)"
        sleep 3; log "server alive after it: $(rsh "kill -0 $SPID && echo yes || echo NO")"
        finish_serve "repro-main"
        log "engine failure line: $(grep -m1 'watchdog' "$OUT/serve-repro-main.log" | cut -c1-300)"
      fi
      ;;
    ids)
      for a in ${ARMS:-R M}; do
        arm_setup $a
        if start_serve "ids-$a" "$ENVS" --max-batch 2 $ARGS; then
          python3 "$S/ids.py" "$URL" guidance "$OUT/ids-$a.json" "http://$HN:$PORT/metrics" 2>&1 | tee -a "$LOG"
          finish_serve "ids-$a"
        fi
      done
      python3 "$S/ids_compare.py" "$OUT" ${ARMS:-R M} 2>&1 | tee -a "$LOG" "$OUT/ids-compare.txt"
      # the served plans (`GET /v1/intarsia/plan`, what `serve` really resolved with --max-batch 2): knobs off must equal main's byte for byte, the opt-in set must grow
      for a in ${PLANARMS:-RM N NM MM}; do
        arm_setup $a
        if start_serve "plan-$a" "$ENVS" --max-batch 2 $ARGS; then finish_serve "plan-$a"; fi
      done
      pl() { python3 -m json.tool --sort-keys "$OUT/plan-served-$1.json" 2>/dev/null; }
      for pair in "ids-R plan-RM" "plan-N plan-NM"; do
        set -- $pair
        if [[ -s "$OUT/plan-served-$1.json" && -s "$OUT/plan-served-$2.json" ]]; then
          cmp -s <(pl $1) <(pl $2) && log "served plan $1 (mx6) vs $2 (main), knobs off: IDENTICAL" || log "served plan $1 (mx6) vs $2 (main), knobs off: DIFFERENT"
        else log "served plan $1 vs $2: a file is missing"; fi
      done
      mm=ids-MM; [[ -s "$OUT/plan-served-$mm.json" ]] || mm=plan-MM
      if [[ -s "$OUT/plan-served-ids-M.json" && -s "$OUT/plan-served-$mm.json" ]]; then
        diff <(pl $mm) <(pl ids-M) > "$OUT/plan-served-on.diff"; log "served plan, opt-in set, main (MM) vs mx6 (M): $(wc -l < "$OUT/plan-served-on.diff") diff lines (plan-served-on.diff)"
      fi
      ;;
  esac
done
