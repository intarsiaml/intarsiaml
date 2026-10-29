#!/bin/bash
# MMTP window 2: served A/B via localeval, 3 interleaved rounds. Run inside a fleetq slot (FLEETQ_JOB set).
# arms: mtp1 (pipelined, this branch), mtp0 (INTARSIA_METAL_MTP_GPU_ACCEPT=0 = main's MTP path, this binary),
#       off-new (spec off, this binary), off-main (spec off, main 0d02d3f36 binary)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mmtp; O=$W/.lane/${OUT:-w2}; mkdir -p $O
NEW=$W/target/release/intarsia; MAIN=$W/.lane/main-target/release/intarsia
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; PORT=18473
ROUNDS=${ROUNDS:-3}; ARMS=${ARMS:-"mtp1 mtp0 off-new off-main"}
srv() { # tag bin spec envs...
  local tag=$1 bin=$2 spec=$3; shift 3
  local cmd="$bin serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --spec $spec"
  echo "## $tag $(date +%T)"
  env X=1 "$@" $cmd > $O/$tag-server.out 2> $O/$tag-server.err &
  local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024,4096,16384 --gen 256 --reps 1 \
     --concurrency 1 --thinking off --label "mmtp-$tag" --recipe "$* $cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval exit $?")
  kill -INT $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
A=($ARMS)
for r in $(seq 1 $ROUNDS); do
  # the arm order rotates by one per round
  for i in $(seq 0 $((${#A[@]} - 1))); do
    a=${A[$(( (i + r - 1) % ${#A[@]} ))]}
    case $a in
      mtp1) srv r$r-mtp1 $NEW mtp INTARSIA_METAL_MTP_GPU_ACCEPT=1 ;;
      mtp0) srv r$r-mtp0 $NEW mtp INTARSIA_METAL_MTP_GPU_ACCEPT=0 ;;
      off-new) srv r$r-off-new $NEW off ;;
      off-main) srv r$r-off-main $MAIN off ;;
    esac
  done
done
echo "## done $(date +%T)"
