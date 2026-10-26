#!/usr/bin/env bash
# Lane l7, LOCAL (endor/Mac), run as the command of `fleetq run --host tatooine-r9700 --quiet gpu,cpu --takes-down radiance-rank0 ...` (so
# localeval runs inside that slot): 35B batched decode A/B, base (origin/main) against dev, interleaved, localeval speed at 1 (the sweep),
# 4, 8 and 16 streams. Radiance is stopped first and restored by the trap, even on failure.
#   scripts/l7/ab.sh [arms: "base dev base dev base dev"] [tag] [max-batch 16]   (an arm may carry env: "dev:INTARSIA_HIP_MOE_GROUPED=0")
set -uo pipefail
arms=${1:-"base dev base dev base dev"}; tag=${2:-l7ab}; mb=${3:-16}
LOG=${LOG:-$HOME/.cache/l7-ab}; mkdir -p "$LOG"
trap 'ssh tatooine "bash ~/intarsiaml-l7-hip/scripts/l7/win.sh restore"' EXIT
ssh tatooine "bash ~/intarsiaml-l7-hip/scripts/l7/win.sh stop"
n=0
for a in $arms; do
  n=$((n + 1)); tree=${a%%:*}; envv=""; [[ "$a" == *:* ]] && envv=${a#*:}
  lab="$tag-$n-$tree${envv:+-$(echo "$envv" | tr '=' '_')}"
  echo "[ab $n $tree $envv] up $(date +%T)"
  ssh tatooine "env $envv bash ~/intarsiaml-l7-hip/scripts/l7/serve.sh up $tree $mb" 2>&1 | tail -4 | tee "$LOG/$lab.up"
  if grep -q "serving after" "$LOG/$lab.up"; then
    (cd /opt/code/deepeval && uv run localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 1024 --gen 128 --reps 3 \
      --concurrency 4,8,16 --label "$lab" --no-unim \
      --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 8192 --max-batch $mb tree=$tree $envv") > "$LOG/$lab.out" 2>&1
    echo "[ab $n $tree] localeval rc=$?"; tail -14 "$LOG/$lab.out" | cut -c1-200
  fi
  ssh tatooine "bash ~/intarsiaml-l7-hip/scripts/l7/serve.sh down $lab" 2>&1 | tail -1
done
echo "[ab] done $(date +%T)"
