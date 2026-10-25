#!/usr/bin/env bash
# Lane sp, inside a fleetq slot on the box (CUDA Spark or Metal endor): the served-vs-generate split of MTP, same request body
# and /metrics counters as win9.sh does on HIP.
#   served-split.sh <cuda|metal> <sizes e.g. 128,1024,4096,16384>
# serve arms: spec off, mtp k=2, mtp k=2 with INTARSIA_SPEC_GATE=0, mtp k=2 with INTARSIA_SPEC_TRACE=1 (server log kept);
# generate arms: `--chat --no-think --temperature 0`, spec off and mtp k=2, on the same prompts.
set -uo pipefail
BE=$1; SIZES=$2
if [ "$BE" = cuda ]; then
  P=$HOME/intarsia-decfuse; BIN=$P/sp-dev/target/acc/release/intarsia; NAT=$P/sp-src-dev/sp-out; O=$P/out/sp-served; EXTRA="--mem-budget 30"
  H=$HOME/.cache/huggingface/hub; M=$H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/$(ls $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots | head -1)
  export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
  CLIENT=$P/sp-src-dev/scripts/sp/servedclient.py
else
  BIN=/opt/code/intarsiaml/.worktrees/sp-targets/dev-metal/release/intarsia; NAT=/opt/code/deepeval/tests/data/prompts/natural-v1
  O=${OUT:-/opt/code/wt-sp/docs/results/2026-10-25-sp-hip-spec/raw/served-split/metal}; EXTRA=""; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
  CLIENT=/opt/code/wt-sp/scripts/sp/servedclient.py
fi
mkdir -p $O; PORT=8394
arm() { # arm <label> <serve args> [KEY=VAL ...]
  l=$1; sa=$2; shift 2
  env "$@" $BIN serve --backend $BE $EXTRA --model $M --host 127.0.0.1 --port $PORT --served-model-name intarsia --max-seq 20480 --max-batch 1 $sa \
    > $O/serve-$l.log 2>&1 < /dev/null &
  pid=$!
  for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:$PORT/v1/models | grep -q '"data"' && break; sleep 4; done
  python3 $CLIENT $l $SIZES 3 http://127.0.0.1:$PORT $NAT > $O/served-$l.log 2>&1
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
arm off ""
arm mtp "--spec mtp --spec-k 2"
arm mtp-gate0 "--spec mtp --spec-k 2" INTARSIA_SPEC_GATE=0
arm mtp-trace "--spec mtp --spec-k 2" INTARSIA_SPEC_TRACE=1
for a in "off:--spec off" "mtp2:--spec mtp --spec-k 2"; do
  tag=${a%%:*}; args=${a#*:}
  for n in ${SIZES//,/ }; do
    $BIN generate -m $M --backend $BE $EXTRA --max-seq 20480 --profile-every 0 --ignore-eos --show-ids --max-tokens 256 --chat --no-think --temperature 0 $args \
      -p "$(cat $NAT/n$n.txt)" > /dev/null 2> $O/chat-$tag-n$n.err < /dev/null
    dl=$(grep -a 'finish=' $O/chat-$tag-n$n.err | tail -1 | sed 's/.*| decode/decode/')
    sl=$(grep -a '^spec:' $O/chat-$tag-n$n.err | tail -1)
    mspp=$(echo "$dl | $sl" | sed -E 's/decode ([0-9]+) ms.*passes=([0-9]+).*/\1 \2/' | awk 'NF==2{printf "%.1f",$1/$2}')
    echo "chat-generate $tag n$n | $(echo $dl | cut -c1-40) | $(echo $sl | cut -c1-120) | ms/pass=$mspp"
  done
done | tee $O/chat-generate.log
echo "== served"; cat $O/served-*.log
echo "== done"
