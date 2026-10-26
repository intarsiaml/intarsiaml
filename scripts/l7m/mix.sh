#!/usr/bin/env bash
# Lane l7m, LOCAL, run as the command of ONE fleetq job that holds both tatooine hosts and takes radiance down (as ab.sh): concurrency-2
# head-of-line check. Per arm the 35B server runs with `--max-batch 16`; two localeval `speed` processes run against it at the same time
# (--ignore-busy, the load is ours): L = 16384-token prompts back to back (8 generated tokens, mostly prefill), S = 1024-token prompts back to back
# (8 generated tokens) started 4 s later, so most S requests arrive while an L prefill is running. S's TTFT is the number: median, and the max
# of its reps (n = S reps per arm; pooled over the interleaved rounds the max is above the 99th percentile when n_total >= 100).
#   scripts/l7m/mix.sh "main|base||--max-batch 16" "branch|dev||--max-batch 16" ... @tag     (arms as ab.sh)
set -uo pipefail
CARD=${CARD:-1}; tag=l7mx; arms=()
for a in "$@"; do case "$a" in @*) tag=${a#@} ;; *) arms+=("$a") ;; esac; done
LOG=${LOG:-$HOME/.cache/l7m-ab}; mkdir -p "$LOG"
trap 'ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/win.sh restore"' EXIT
ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/win.sh stop"
n=0
for a in "${arms[@]}"; do
  n=$((n + 1)); IFS='|' read -r name tree envv args <<<"$a"
  lab="$tag-$n-$name"; envs=$(echo "$envv" | tr ',' ' ')
  echo "[mix $n $name] up $(date +%T)  tree=$tree env=[$envs] args=[$args]"
  ssh tatooine "env $envs CARD=$CARD MAX_SEQ=${MAX_SEQ:-17408} bash ~/intarsiaml-l7m-hip/scripts/l7m/serve.sh up $tree - $args" 2>&1 | tail -4 | tee "$LOG/$lab.up"
  if grep -q "serving after" "$LOG/$lab.up"; then
    U="--url http://tatooine:${PORT:-8296}/v1 --model intarsia --thinking off --ignore-busy --no-unim"
    (cd /opt/code/deepeval && uv run localeval speed $U --sizes 16384 --gen 8 --reps ${LREPS:-14} --label "$lab-long" \
        --recipe "mix L $lab" > "$LOG/$lab-long.out" 2>&1) &
    lp=$!
    sleep 4
    (cd /opt/code/deepeval && uv run localeval speed $U --sizes 1024 --gen 8 --reps ${SREPS:-40} --label "$lab-short" \
        --recipe "mix S $lab" > "$LOG/$lab-short.out" 2>&1)
    echo "[mix $n $name] short rc=$?"; wait $lp; echo "[mix $n $name] long rc=$?"
    grep -aE "^\| (99|10[0-9]|16[0-9]{3}) \|" "$LOG/$lab-short.out" "$LOG/$lab-long.out" | cut -c1-200
  fi
  ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/serve.sh down $lab" 2>&1 | tail -1
done
echo "[mix] done $(date +%T)"
