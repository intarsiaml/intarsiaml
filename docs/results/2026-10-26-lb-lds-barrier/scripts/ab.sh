#!/usr/bin/env bash
# Lane lb, LOCAL (endor/Mac), run as the command of `fleetq run --host tatooine-r9700 --quiet gpu,cpu --takes-down radiance-rank0 ...` (so
# localeval runs inside that slot): 35B served A/B, base (origin/main) against dev, interleaved, localeval speed at 512 / 4096 / 16384
# tokens (prefill tok/s and decode tok/s at one stream) and 8 streams (aggregate decode, 4096-token prompts). Radiance is stopped first and
# restored by the trap, even on failure.
#   scripts/lb/ab.sh [arms: "base dev base dev base dev"] [tag] [max-batch 8]
set -uo pipefail
arms=${1:-"base dev base dev base dev"}; tag=${2:-lbab}; mb=${3:-8}
LOG=${LOG:-$HOME/.cache/lb-ab}; mkdir -p "$LOG"
trap 'ssh tatooine "bash ~/intarsiaml-lb/dev/scripts/lb/win.sh restore"' EXIT
ssh tatooine "bash ~/intarsiaml-lb/dev/scripts/lb/win.sh stop"
n=0
for a in $arms; do
  n=$((n + 1)); tree=${a%%:*}; envv=""; [[ "$a" == *:* ]] && envv=${a#*:}
  lab="$tag-$n-$tree${envv:+-$(echo "$envv" | tr '=' '_')}"
  echo "[ab $n $tree $envv] up $(date +%T) radiance=$(ssh tatooine 'systemctl --user is-active radiance')"
  ssh tatooine "env $envv bash ~/intarsiaml-lb/dev/scripts/lb/serve.sh up $tree $mb" 2>&1 | tail -4 | tee "$LOG/$lab.up"
  if grep -q "serving after" "$LOG/$lab.up"; then
    (cd /opt/code/deepeval && uv run localeval speed --url http://tatooine:8294/v1 --model intarsia --thinking off --sizes 512,4096,16384 --gen 128 --reps 3 \
      --concurrency 8 --label "$lab" --no-unim \
      --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 18432 --max-batch $mb tree=$tree $envv") > "$LOG/$lab.out" 2>&1
    echo "[ab $n $tree] localeval rc=$? radiance=$(ssh tatooine 'systemctl --user is-active radiance')"; tail -14 "$LOG/$lab.out" | cut -c1-200
  fi
  ssh tatooine "bash ~/intarsiaml-lb/dev/scripts/lb/serve.sh down $lab" 2>&1 | tail -1
done
echo "[ab] done $(date +%T)"
