#!/usr/bin/env bash
# Lane l7m, LOCAL, run as the command of ONE fleetq job that holds BOTH tatooine hosts (localeval maps http://tatooine:8296 to tatooine-r9700 and
# queues a second slot there unless the job covers it; a job on tatooine-r9700-2 alone makes it queue behind everyone, which stalled the first try):
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --mem tatooine-r9700=4,tatooine-r9700-2=4 --quiet gpu,cpu --takes-down radiance-rank0,radiance-rank1 ...
# CARD=1 serves on the second card (CARD=0 the first): single-stream 35B prefill A/B with localeval `speed` at 1024/4096/16384 prompt tokens (gen 16: prefill is the question).
# Radiance is stopped first and restored by the trap. Arms are "name|tree|ENV=v,ENV=v|serve args", interleaved in the order given:
#   scripts/l7m/ab.sh "main|dev||" "macro|dev|INTARSIA_PREFILL_MACRO=8192|" ... [tag]   (tag = last argument starting with @)
set -uo pipefail
CARD=${CARD:-0}; tag=l7m; arms=()
for a in "$@"; do case "$a" in @*) tag=${a#@} ;; *) arms+=("$a") ;; esac; done
LOG=${LOG:-$HOME/.cache/l7m-ab}; mkdir -p "$LOG"
# NOWIN=1: the caller (window.sh) stopped radiance and restores it
if [ -z "${NOWIN:-}" ]; then
  trap 'ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/win.sh restore"' EXIT
  ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/win.sh stop"
fi
n=0
for a in "${arms[@]}"; do
  n=$((n + 1)); IFS='|' read -r name tree envv args <<<"$a"
  SZ=1024,4096,16384; [[ "$args" == *--max-batch* ]] && SZ=1024,4096,12288  # --max-batch 16 caps the context at 16223 (13788 with the 16384-row macro workspace): no 16k prompt fits
  lab="$tag-$n-$name"; envs=$(echo "$envv" | tr ',' ' ')
  echo "[ab $n $name] up $(date +%T)  tree=$tree env=[$envs] args=[$args]"
  ssh tatooine "env $envs CARD=$CARD MAX_SEQ=${MAX_SEQ:-17408} bash ~/intarsiaml-l7m-hip/scripts/l7m/serve.sh up $tree - $args" 2>&1 | tail -4 | tee "$LOG/$lab.up"
  if grep -q "serving after" "$LOG/$lab.up"; then
    (cd /opt/code/deepeval && uv run localeval speed --url http://tatooine:${PORT:-8296}/v1 --model intarsia --thinking off --sizes $SZ --gen ${GEN:-16} --reps 3 \
      --label "$lab" --no-unim \
      --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq ${MAX_SEQ:-17408} tree=$tree env=[$envs] args=[$args] card=$CARD") > "$LOG/$lab.out" 2>&1
    echo "[ab $n $name] localeval rc=$?"; grep -aE "^\| (99|10[0-9]|40|41|16)[0-9]* \|" "$LOG/$lab.out" | cut -c1-200
  fi
  ssh tatooine "bash ~/intarsiaml-l7m-hip/scripts/l7m/serve.sh down $lab" 2>&1 | tail -1
done
echo "[ab] done $(date +%T)"
