#!/usr/bin/env bash
# Lane cmx: full MMLU (n = 1425, 5-shot, --limit 25 per subject, thinking off, temperature 0) on the dev tree's `intarsia serve` (recipe defaults: W4A16,
# MTP k2, prefill chunk 256), as docs/results/2026-10-27-mmlu27-27b-quality.md. Mac side, inside a fleetq slot on spark1 (FLEETQ_JOB set):
#   fleetq run --host spark1 --mem 50 --expect 80 --label "cmx: MMLU n=1425 on dev" -- scripts/cmx/mmlu.sh [tree]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8193; TREE=${1:-dev}
OUT=/opt/code/intarsiaml/.worktrees/cmx/docs/results/2026-10-27-cmx-cuda-mxfp4-prefill/raw/mmlu-$TREE; mkdir -p $OUT
M=/home/nvidia/intarsiaml-cmx/models/Qwen3.8-27B-MXFP4
ssh -n -o BatchMode=yes $HOST "pkill -f 'release/intarsia serve.*--port $PORT'; sleep 3" || true
SPID=$(ssh -n -o BatchMode=yes $HOST "mkdir -p ~/intarsiaml-cmx/trees/$TREE/out; nohup ~/intarsiaml-cmx/trees/$TREE/target/release/intarsia serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name qwen3.8-27b --max-seq 8192 --mem-budget 45 --profile-every 0 > ~/intarsiaml-cmx/trees/$TREE/out/mmlu-serve.log 2>&1 < /dev/null & echo \$!")
trap 'ssh -n $HOST "kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null"' EXIT
for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && break; sleep 2; done
ssh -n $HOST "grep -aE 'prefill chunk|mxfp4 act|spec' ~/intarsiaml-cmx/trees/$TREE/out/mmlu-serve.log | head -5" | cut -c1-200 | tee $OUT/serve-head.txt
( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval quality --url http://$HN:$PORT/v1 --model qwen3.8-27b --bench mmlu --limit 25 --thinking off \
    --label cmx-mmlu-$TREE --no-unim --wait-idle 600 --recipe "intarsia serve --backend cuda tree=$TREE (cmx MXFP4 W4A16 prefill GEMM), recipe defaults" ) > $OUT/mmlu.out 2>&1
grep -aE "mmlu|run_id|results/|accuracy" $OUT/mmlu.out | head -8
