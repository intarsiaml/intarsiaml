#!/usr/bin/env bash
# Lane l5r7, LOCAL, inside the fleetq job that holds tatooine-r9700-2 and takes radiance down: served prefill, merge-base build against the branch build
# (z-fast launch order default on), interleaved base dev base dev base dev, with localeval (the local-model-bench method): sizes 512, 2048, 4096, 16384
# (prompt tokens), 8 generated tokens, 3 reps + 1 warmup, one stream. usage (from the repo root):
#   NAME=<snapshot> FLEETQ_OWNER=claude-f31ccf91-l5 fleetq run --host tatooine-r9700-2 --takes-down radiance-rank1 --quiet gpu,cpu --mem 4 --expect 40 --label "..." -- scripts/l5r7/ab.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}" "${NAME:?}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$S/../../docs/results/2026-10-25-l5r7-moe-zfast/raw/ab}; mkdir -p "$OUT"
LE=/opt/code/deepeval; URL=http://tatooine:8296/v1
ssh -o BatchMode=yes tatooine 'mkdir -p ~/intarsiaml-l5-logs' && scp -q "$S/serve-win.sh" tatooine:intarsiaml-l5-logs/serve-win.sh
sv() { ssh -o BatchMode=yes tatooine "NAME=$NAME bash ~/intarsiaml-l5-logs/serve-win.sh $*" < /dev/null; }
trap 'sv restore | tee -a "$OUT/ab.log"' EXIT
i=0
for arm in ${ARMS:-base dev base dev base dev}; do
  i=$((i + 1))
  sv up $arm v$i-$arm | tail -1 | tee -a "$OUT/ab.log"
  (cd $LE && uv run localeval speed --url $URL --model intarsia --thinking off --sizes 512,2048,4096,16384 --gen 8 --reps 3 --concurrency 1 \
      --recipe "intarsia serve --backend hip --model Qwen3.6-35B-A3B-NVFP4 --max-seq 20480 --max-batch 1 (arm $arm, l5r7 zfast A/B)" \
      --label l5r7-v$i-$arm > "$OUT/v$i-$arm.txt" 2>&1; cp "$(ls -dt $LE/results/*l5r7-v$i-$arm | head -1)/result.json" "$OUT/v$i-$arm.json")
  sv down v$i-$arm | tail -1 | tee -a "$OUT/ab.log"
  echo "visit $i $arm done $(date +%T)" | tee -a "$OUT/ab.log"
done
