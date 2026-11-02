#!/bin/bash
# MPROF window 2: served 16-stream burst (CHUNK4K burst method: serve --max-batch 16, localeval speed --sizes 1024 --gen 512
# --reps 1 --concurrency ...), plain / cb / ce1 / kernel. usage: w2.sh OUTDIR  (inside a fleetq slot)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mnarrow; O=$1; mkdir -p $O; BIN=$W/.lane/intarsia-metal; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
PORT=18491
srv() { local tag=$1 conc=$2 gen=$3; shift 3
  local cmd="$BIN serve --backend metal --model $M35 --max-batch 16 --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  echo "## $tag $(date +%T)"; env X=1 "$@" $cmd > $O/$tag-server.out 2> $O/$tag-server.err < /dev/null & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen $gen --reps 1 \
     --concurrency $conc --thinking off --label "mprof-w2-$tag" --recipe "$* $cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err < /dev/null; echo "localeval exit $?")
  grep -A6 '^| streams' $O/$tag-localeval.out | tail -4
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
srv s-plain1 1,8,16 512
srv s-plain2 1,8,16 512
srv s-host 1,8,16 512 INTARSIA_HOST_PROFILE=1
srv s-cb 1,8,16 512 INTARSIA_METAL_PROFILE=cb:$O/s-cb.jsonl
srv s-ce1 1,8,16 512 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/s-ce1.jsonl
srv s-kernel 16 128 INTARSIA_METAL_PROFILE=kernel:$O/s-kernel.jsonl
ls -la $O
gzip -f $O/*.jsonl
echo "## done $(date +%T)"
