#!/usr/bin/env bash
# cvatflip gate: `serve --max-batch 8 --spec mtp --spec-k 2` (batched MTP + batched verify, the 35B's CUDA default), 8 concurrent greedy requests (the cells of scripts/mcopy/prompts.py),
# knob unset (def, the new default: on) vs INTARSIA_CUDA_VERIFY_ATTN_ROWS=0 (zero): every reply must be byte-identical. One server per arm; replies in $O/txt-<arm>/, summary at the end.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "cvatflip mb8 gate" -- ssh spark1 'bash -s' < scripts/cvatflip/mb8.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cvatflip; O=${O:-$R/out/mb8}; mkdir -p $O
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
k=0; BIN=${BIN:-$R/bin/intarsia}; PORT=${PORT:-8298}; C=$R/src/scripts/mcopy/cell.py
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
nohup bash $R/src/scripts/cvatflip/wd.sh > $O/wd.log 2>&1 < /dev/null &
WD=$!
pat() { echo "$R/bin/[a-z0-9-]+ serve.*--port $PORT"; }
stop() { pkill -f "$(pat)" 2>/dev/null; for i in $(seq 1 30); do pgrep -f "$(pat)" >/dev/null || break; sleep 1; done; }
trap 'stop; kill $WD 2>/dev/null' EXIT
for arm in def zero; do
  E=X=1; [ $arm = zero ] && E=INTARSIA_CUDA_VERIFY_ATTN_ROWS=0
  stop
  env $E INTARSIA_SPEC_TRACE=1 nohup $BIN serve --backend cuda --model $M35 --model-id m --max-batch 8 --max-seq 8192 --spec mtp --spec-k 2 --profile-every 0 --mem-budget 40 \
    --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$arm.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do pgrep -f "$(pat)" >/dev/null || { echo "serve died ($arm)"; tail -5 $O/serve-$arm.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
  echo "$arm up $(date +%T) free: $(free -g | awk 'NR==2{print $7}') GB"
  python3 $C $PORT $R/cells/chat-short.json 1 warm > /dev/null 2>&1
  rm -rf $O/txt-$arm; pids=""
  for c in chat-short code-gen reason agent-edit json-xform copy3k agent-batch chat-short; do
    python3 $C $PORT $R/cells/$c.json 1 g-$c-$((++k)) --out $O/txt-$arm > $O/$arm-$c-$k.jsonl 2>&1 &
    pids="$pids $!"
  done
  wait $pids
  curl -s http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_(spec_|graph_|decode_eager|batch)' | sed 's/{[^}]*}//' | tr '\n' ' ' > $O/metrics-$arm.txt
  echo "$arm: $(ls $O/txt-$arm | wc -l) replies; $(grep -h 'verify attention rows' $O/serve-$arm.log | sort | uniq -c | tr '\n' ';')"
  stop
done
fails=0
for f in $(ls $O/txt-def); do cmp -s $O/txt-def/$f $O/txt-zero/$f && echo "PASS $f identical ($(wc -c < $O/txt-def/$f) bytes)" || { echo "FAIL $f differs"; fails=$((fails+1)); }; done
echo "mb8 gate fails=$fails (replies def $(ls $O/txt-def | wc -l), zero $(ls $O/txt-zero | wc -l))"
