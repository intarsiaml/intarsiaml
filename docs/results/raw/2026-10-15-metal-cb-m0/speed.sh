#!/bin/zsh
# usage: speed.sh <outdir> <rounds>   (inside an exclusive endor slot): localeval speed 128/1k/4k against `intarsia serve`,
# interleaved base, new, base, new, ... (one server at a time, fresh process each)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
O=$1; ROUNDS=$2
BIN_DIR=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/bin
M=$HOME/models/Qwen3-8B
mkdir -p $O
for r in $(seq 1 $ROUNDS); do
  for v in base new; do
    if [ $v = base ]; then BIN=$BIN_DIR/intarsia-base-56ec2d5; FL=(); else BIN=$BIN_DIR/intarsia-new-${NEW_REV:?}; FL=(--max-batch 1); fi
    $BIN serve --model $M --backend metal --max-seq 8192 --port 8098 --no-ui $FL > $O/$v-r$r-serve.log 2>&1 &
    SP=$!
    for i in {1..300}; do curl -sf http://127.0.0.1:8098/health > /dev/null && break; sleep 1; done
    (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8098/v1 --model qwen3-8b --thinking off \
       --sizes 128,1024,4096 --gen 128 --reps 3 --label cb-m0-$v-r$r) > $O/$v-r$r.out 2>&1
    echo "$(date +%T) $v r$r: exit $?"
    kill $SP; wait $SP 2>/dev/null
    pkill -f "intarsia-.* serve .*--port 8098" 2>/dev/null; sleep 5
  done
done
