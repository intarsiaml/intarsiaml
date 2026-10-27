#!/usr/bin/env bash
# Lane H27P (adapted from scripts/w4a8/cat-window.sh): localeval prefill / decode sweep of the 27B at tp2 for the base build and this branch, driven
# FROM THE MAC (localeval lives here), one fleetq job that holds both R9700s and takes radiance down:
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --mem 30 --expect 45 \
#     --label "h27p speed" -- env W=w5 scripts/h27p/speed-window.sh
# Arms (ARMSEQ, rounds separated by ";"): base = ~/intarsiaml-h27p/base-intarsia (origin/main, chunk 256), dev = h27p-intarsia (recipe chunk 2048 + peel 16),
# bf16 = dev with INTARSIA_TP_PARTIAL=bf16 (bf16 partials in the all-reduce: an option, not the default). A fresh `serve` per arm (MTP at its default k2),
# then `localeval speed --sizes $SIZES --gen 128 --reps 3`. MB = --max-batch (default 1; the shipped single-stream setting).
# Radiance is restored by guard.sh on tatooine when this script ends or dies.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${W:?W=<window>}"
ARMSEQ=${ARMSEQ:-"base dev;dev base;base dev"}; SIZES=${SIZES:-1024,4096,16384}; MB=${MB:-1}; MS=${MS:-20480}; PORT=8095
M27=/models/radiance/hf-cache/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=$REPO/docs/results/2026-10-27-h27p-hip-27b-prefill/raw/$W; mkdir -p "$OUT"
say() { echo "$(date +%T) $*" | tee -a "$OUT/steps.log"; }
SSH="ssh -o BatchMode=yes tatooine"
FIFO=$(mktemp -u "$REPO/h27p-out/guard.XXXXXX"); mkfifo "$FIFO"
$SSH 'mkdir -p ~/spt' < /dev/null; scp -q "$HERE/../spt/guard.sh" tatooine:spt/
$SSH 'bash ~/spt/guard.sh' > "$OUT/guard.log" 2>&1 < "$FIFO" &
GUARD=$!
exec 9> "$FIFO"
for i in $(seq 1 30); do grep -q "radiance stopped" "$OUT/guard.log" 2>/dev/null && break; sleep 1; done
say "guard: $(grep -h 'radiance stopped' "$OUT/guard.log")"
cleanup() { $SSH 'docker rm -f h27p-serve' >/dev/null 2>&1 < /dev/null; exec 9>&-; wait $GUARD; rm -f "$FIFO"; say "guard: $(grep -h 'radiance answers\|DID NOT' "$OUT/guard.log")"; }
trap cleanup EXIT

serve() { # serve <arm>
  local arm=$1 bin=h27p-intarsia envs="" mb=$MB
  [[ $arm =~ ^[a-z0-9]+[a-z]([0-9]+)$ ]] && mb=${BASH_REMATCH[1]}   # an arm name ending in digits sets --max-batch (dev16, base16, dev1)
  [[ $arm == base* ]] && bin=base-intarsia
  [[ $arm == bf16 ]] && envs="-e INTARSIA_TP_PARTIAL=bf16"
  RECIPE="HIP_VISIBLE_DEVICES=0,1 $bin serve --backend hip --model 27B --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS (lane h27p, arm $arm, docker $IMG)"
  $SSH "docker rm -f h27p-serve >/dev/null 2>&1; docker run -d --name h27p-serve --network host --device /dev/kfd --device /dev/dri \
    --group-add \$(stat -c %g /dev/kfd) --group-add \$(stat -c %g /dev/dri/renderD128) --group-add video --ipc host \
    --security-opt seccomp=unconfined --user \$(id -u):\$(id -g) -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 -e HIP_VISIBLE_DEVICES=0,1 $envs \
    -v \$HOME/intarsiaml-h27p:/work -v /data/models:/models:ro -w /work $IMG /work/$bin serve --backend hip --model $M27 \
    --served-model-name qwen38-27b-$arm --tp 2 --devices 0,1 --max-batch $mb --profile-every 0 --mem-budget 29 --max-seq $MS --no-supervisor --host 0.0.0.0 --port $PORT" \
    < /dev/null > "$OUT/docker-$arm.id" 2>&1
  for i in $(seq 1 200); do curl -sf -m 3 "http://tatooine:$PORT/health" > /dev/null 2>&1 && return 0; sleep 3; done
  $SSH 'docker logs --tail 40 h27p-serve' > "$OUT/serve-fail-$arm.log" 2>&1 < /dev/null; return 1
}
r=0
IFS=';' read -ra ROUNDS <<< "$ARMSEQ"
for order in "${ROUNDS[@]}"; do
  r=$((r + 1))
  for arm in $order; do
    say "round $r arm $arm: starting server"
    if ! serve "$arm"; then say "round $r arm $arm: server did not come up"; continue; fi
    $SSH 'docker logs h27p-serve 2>&1 | head -80' < /dev/null > "$OUT/serve-$arm-r$r.log" 2>&1
    printf '[targets.h27p-%s]\nurl = "http://tatooine:%s/v1"\nmodel = "qwen38-27b-%s"\n' "$arm" "$PORT" "$arm" > "$OUT/targets-$arm.toml"
    if [[ ${MIX:-0} == 2 ]]; then   # MIX=2: texts of A and B when B prefills beside A (mixids.py), 4k-0 and 16k-3 (6 tail rows)
      python3 "$HERE/mixids.py" "http://tatooine:$PORT" "qwen38-27b-$arm" "$REPO/scripts/cmtp/ids-prompts/4k-0.txt" "$REPO/scripts/cmtp/ids-prompts/16k-3.txt" "$REPO/scripts/cmtp/ids-prompts/4k-3.txt" > "$OUT/mixids-$arm-r$r.log" 2>&1
      say "round $r arm $arm mixids: $(tr '\n' '|' < "$OUT/mixids-$arm-r$r.log" | cut -c1-400)"
      $SSH 'docker rm -f h27p-serve' >/dev/null 2>&1 < /dev/null
      continue
    fi
    if [[ ${MIX:-0} == 1 ]]; then   # MIX=1: the mixed prefill-beside-decode latency check instead of the sweep
      python3 "$HERE/mix.py" "http://tatooine:$PORT" "qwen38-27b-$arm" "$REPO/scripts/cmtp/ids-prompts/4k-0.txt" 3 > "$OUT/mix-$arm-r$r.log" 2>&1
      say "round $r arm $arm mix: $(cat "$OUT/mix-$arm-r$r.log" | tr '\n' '|' | cut -c1-400)"
      $SSH 'docker rm -f h27p-serve' >/dev/null 2>&1 < /dev/null
      continue
    fi
    ( cd /opt/code/deepeval && LOCALEVAL_TARGETS="$OUT/targets-$arm.toml" uv run localeval speed "h27p-$arm" --sizes "$SIZES" --gen 128 --reps 3 \
        --label "h27p-$W-$arm-r$r" --recipe "$RECIPE" --wait-idle 60 ) > "$OUT/sweep-$arm-r$r.log" 2>&1 < /dev/null
    say "round $r arm $arm sweep: exit $? $(grep -aoE 'results/[^ ]+' "$OUT/sweep-$arm-r$r.log" | tail -1)"
    $SSH 'docker rm -f h27p-serve' >/dev/null 2>&1 < /dev/null
  done
done
say "done"
