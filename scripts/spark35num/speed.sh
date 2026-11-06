#!/usr/bin/env bash
# Lane spark35, window 2 (one fleetq slot on $HOST, --quiet gpu,cpu): the speed ABAB, from scripts/fp8kv/le-window.sh. Per round every arm runs
# once (even rounds reversed, so with ROUNDS=2 the order is a b c d / d c b a): a fresh `intarsia serve` per arm, then localeval speed.
#   cell single: one stream, prompts 1k / 4k / 16k, gen 128, MTP default (recipe), 3 reps + warm-up
#   cell burst : (the 1-stream rep of the same call, then 4 / 8 / 16 parallel streams), 1k prompts, gen 256 (the SRV16 shape; --max-batch 16)
# Arms: base (bf16 KV, f32 state), c (INTARSIA_GDN_STATE=bf16), d (--kv-dtype fp8), cd (both). Needs bin/dev (window 1 builds it; this script
# builds it from HEAD when it is missing).
#   fleetq run --host spark2 --quiet gpu,cpu --mem 50 --expect 90 --label "spark35: speed ABAB" -- scripts/spark35num/speed.sh [rounds] [arms...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ROUNDS=${1:-2}; shift || true
ARMS=("$@"); [[ ${#ARMS[@]} -gt 0 ]] || ARMS=(base c d cd)
OUT=$RES/speed; mkdir -p "$OUT"
trap 'stop_server' EXIT INT TERM
rsh "test -x $L/bin/dev" || "$H/build.sh" "${DEV_REF:-HEAD}" dev || { echo "no dev binary"; exit 1; }
spec() { case $1 in base) echo "||" ;; c) echo "INTARSIA_GDN_STATE=bf16||" ;; d) echo "|--kv-dtype fp8|" ;; cd) echo "INTARSIA_GDN_STATE=bf16|--kv-dtype fp8|" ;; esac; }
le() { # le <label> <recipe> <sizes> <gen> <conc>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes "$3" --gen "$4" --reps 3 ${5:+--concurrency "$5"} \
      --label "$1" --no-unim --recipe "$2" ) > "$OUT/$1.out" 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' "$OUT/$1.out" | head -2 | tr '\n' ' ')" | tee -a "$OUT/index.txt"
  grep -aE 'streams: aggregate|decode|tok/s' "$OUT/$1.out" | head -8
}
run_arm() { # run_arm <round> <arm>: single-stream cell on a default server (MTP on), burst cell on --max-batch 16
  local r=$1 arm=$2 envs xargs R
  IFS='|' read -r envs xargs _ <<< "$(spec "$arm")"
  fits 18 "speed $arm r$r" || return 0
  echo "== round $r arm $arm $(date +%T)"
  if MAXSEQ=20480 MEMB=40 start_server "speed-$arm-r$r-single" dev "$envs" $xargs; then
    echo "$arm r$r single: MemAvailable $(free_gb) GB with the server up" | tee -a "$OUT/mem.txt"
    R="spark35 speed: intarsia serve --backend cuda --max-seq 20480 --mem-budget 40 bin=dev env='$envs' $xargs (MTP default)"
    le "s35-$arm-single-r$r" "$R" 1024,4096,16384 128
    stop_server
  fi
  if MAXSEQ=8192 MEMB=40 start_server "speed-$arm-r$r-burst" dev "$envs" --max-batch 16 $xargs; then
    R="spark35 speed: intarsia serve --backend cuda --max-seq 8192 --mem-budget 40 --max-batch 16 bin=dev env='$envs' $xargs"
    le "s35-$arm-burst-r$r" "$R" 1024 256 4,8,16
    stop_server
  fi
}
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm "$r" "${ARMS[k]}"; done
  else for a in "${ARMS[@]}"; do run_arm "$r" "$a"; done; fi
done
echo "== speed done $(date +%T)"
