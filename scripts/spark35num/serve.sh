#!/usr/bin/env bash
# Lane spark35: the serve arms on $HOST (one `intarsia serve` per arm): greedy ids (gen 1024) at 1k/4k/16k, needle at 32k and 64k,
# MMLU (10 per task) and GSM8K (100), as lane fp8kv ran them (scripts/fp8kv/w3.sh). Arms: main (origin/main binary, no flags: the ids gate
# against base), base, fp8, cemu, cdemu (see kl.sh). TASKS selects: greedy needle mmlu gsm8k. Needs bin/main, bin/dev, and
# $RES/data/needle-{32k,64k}.jsonl (w1.sh makes them).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ARMS=${ARMS:-"main base fp8 cemu cdemu"}; TASKS=${TASKS:-"greedy needle mmlu gsm8k"}
trap 'stop_server' EXIT INT TERM
spec() { case $1 in
  main) echo "main||" ;; base) echo "dev||" ;; fp8) echo "dev||--kv-dtype fp8" ;;
  cemu) echo "dev|INTARSIA_CUDA_GDN_STATE_BF16_EMU=1|" ;; cdemu) echo "dev|INTARSIA_CUDA_GDN_STATE_BF16_EMU=1|--kv-dtype fp8" ;;
  creal) echo "dev|INTARSIA_GDN_STATE=bf16|" ;; cdreal) echo "dev|INTARSIA_GDN_STATE=bf16|--kv-dtype fp8" ;; esac; }
# tasks per arm: main is only the ids gate; the real layout (must equal its emulation) skips the accuracy evals
arm_tasks() { case $1 in main) echo greedy ;; creal|cdreal) echo "greedy needle" ;; *) echo "$TASKS" ;; esac; }
le() { ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval quality --url "http://$HN:$PORT/v1" --model intarsia --thinking off --ignore-busy --no-unim "$@" ); }
for arm in $ARMS; do
  IFS='|' read -r bin envs extra <<< "$(spec "$arm")"
  fits 12 "serve $arm" || continue
  echo "== serve $arm $(date +%T)"
  start_server "$arm" "$bin" "$envs" $extra || continue
  R="spark35 intarsia serve --backend cuda --max-seq ${MAXSEQ:-73728} --mem-budget ${MEMB:-45} bin=$bin env='$envs' $extra"
  for t in $(arm_tasks $arm); do
    case $t in
      greedy) python3 "$H/../fp8kv/greedy.py" run --url "http://$HN:$PORT/v1" --gen 1024 --out "$RES/greedy-$arm.json" > "$RES/greedy-$arm.log" 2>&1; tail -1 "$RES/greedy-$arm.log" ;;
      needle) for n in 32k 64k; do
          le --dataset "$RES/data/needle-$n.jsonl" --judge none --max-tokens 256 --label "s35-$arm-needle-$n" --recipe "$R" > "$RES/needle-$n-$arm.out" 2>&1
          echo "needle-$n $arm: $(grep -a -E 'Contains|contains|pass|score' "$RES/needle-$n-$arm.out" | tail -3 | tr '\n' ' ' | cut -c1-200)"; done ;;
      mmlu) le --bench mmlu --limit 10 --label "s35-$arm-mmlu" --recipe "$R" > "$RES/quality-$arm-mmlu.out" 2>&1; grep -a -E 'run_id|Run id|results/|mmlu' "$RES/quality-$arm-mmlu.out" | tail -3 ;;
      gsm8k) le --bench gsm8k --limit 100 --label "s35-$arm-gsm8k" --recipe "$R" > "$RES/quality-$arm-gsm8k.out" 2>&1; grep -a -E 'run_id|Run id|results/|gsm8k' "$RES/quality-$arm-gsm8k.out" | tail -3 ;;
    esac
  done
  rsh "grep -aE 'graph|eager|spec:|row-table|kv_dtype' $L/out/serve-$arm.log | sort | uniq -c | sort -rn | head -6" > "$RES/serve-$arm-notes.txt"
  stop_server
done
echo "== serve done $(date +%T)"
