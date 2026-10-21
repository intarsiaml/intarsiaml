#!/usr/bin/env bash
# J1 HIP no-op, step 2: ONE radiance-down window on tatooine (<= 8 min), from the Mac. Stops radiance, runs the stages
# below ON tatooine under `timeout CAP`, restores radiance in an EXIT trap on tatooine, fetches the outputs into
# docs/results/2026-10-21-j1-structured/raw/hip/ and appends the window to raw/downtime.log.
#   fleetq run --host tatooine --exclusive --takes-down radiance --mem 30 --label "J1 HIP no-op window" -- \
#     bash scripts/j1-structured/hip-window.sh
# Stages (base = ~/intarsiaml-j1-base, head = ~/intarsiaml-j1-head, built by hip-build.sh): 256 greedy ids, --ignore-eos,
#   GPU 0: Qwen3-8B short prompt base, head; then Qwen3.6-35B-A3B-NVFP4 short prompt base, head (if time is left);
#   GPU 1 (in parallel): Qwen3-8B 4k prompt base, head.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
R=$(cd "$(dirname "$0")/../.." && pwd)
RAW=$R/docs/results/2026-10-21-j1-structured/raw
mkdir -p "$RAW/hip"
CAP=${CAP:-300}
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB CAP=$CAP bash -s" <<'EOS' 2>&1 | tee "$RAW/hip/window.log"
set -uo pipefail
O=$HOME/j1-out; rm -rf "$O"; mkdir -p "$O"
now() { date '+%F %T'; }
restore() {
  for c in $(docker ps -q --filter ancestor=intarsia-hip-dev:rocm7.2.4-rust1.90.0); do docker rm -f "$c" >/dev/null 2>&1; done
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 180 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB; stopping radiance $(now)"
systemctl --user stop radiance
sleep 2
echo "[window] radiance stopped $(now)"
SHORT='Write a long story about a robot who learns to paint.'
gen() { # tree gpu tag model prompt
  local t=$1 g=$2 tag=$3 m=$4 p=$5
  INTARSIA_REPO=$HOME/intarsiaml-j1-$t HIP_VISIBLE_DEVICES=$g timeout 240 $HOME/intarsiaml-j1-$t/deploy/tatooine/dev.sh \
    /work/j1-intarsia generate -m $m --backend hip --mem-budget 29 --max-seq 8192 --chat --no-think --profile-every 0 \
    --ignore-eos --show-ids -p "$p" --max-tokens 256 > "$O/gen-$t-$tag.out" 2> "$O/gen-$t-$tag.err"
  echo "$(date +%T) gen $t $tag exit $?"
  grep -a '^ids:' "$O/gen-$t-$tag.err" > "$O/gen-$t-$tag.ids"
}
P4K="$(cat $HOME/intarsiaml-j1-head/scripts/hip-decode/prompt4k.txt)"
stages() {
  ( gen base 0 8b-short /models/Qwen3-8B "$SHORT"; gen head 0 8b-short /models/Qwen3-8B "$SHORT"
    gen base 0 35b-short /models/Qwen3.6-35B-A3B-NVFP4 "$SHORT"; gen head 0 35b-short /models/Qwen3.6-35B-A3B-NVFP4 "$SHORT" ) &
  ( gen base 1 8b-4k /models/Qwen3-8B "$P4K"; gen head 1 8b-4k /models/Qwen3-8B "$P4K" ) &
  wait
}
export -f gen; export O SHORT P4K
timeout "$CAP" bash -c "$(declare -f stages); stages"
echo "[window] stages done $(now) (exit $?)"
for tag in 8b-short 8b-4k 35b-short; do
  if [ -s "$O/gen-base-$tag.ids" ] && [ -s "$O/gen-head-$tag.ids" ]; then
    cmp -s "$O/gen-base-$tag.ids" "$O/gen-head-$tag.ids" && echo "[noop] $tag ids identical ($(md5sum < "$O/gen-base-$tag.ids" | cut -c1-8), n=$(wc -w < "$O/gen-base-$tag.ids"))" || echo "[noop] $tag ids DIFFER"
  else
    echo "[noop] $tag not run"
  fi
done
EOS
rc=${PIPESTATUS[0]}
scp -q "tatooine:j1-out/*" "$RAW/hip/" 2>/dev/null
start=$(grep -m1 'radiance stopped' "$RAW/hip/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}')
back=$(grep -m1 'radiance answers' "$RAW/hip/window.log" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
echo "$(date +%F) fleetq #$FLEETQ_JOB J1 HIP no-op radiance stopped $start answers ${back:-NEVER} (window.log in raw/hip)" >> "$RAW/downtime.log"
exit $rc
