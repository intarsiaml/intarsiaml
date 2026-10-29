#!/usr/bin/env bash
# Lane bi8 GPU job 2 (spark2, --quiet gpu,cpu): after the fix. 8B CUDA gates (model batched == solo at N 2..16, engine == cold
# reference), served greedy identity vs the cold reference (solo, --max-batch 16, --max-batch 8; reps on one server), equal.py's
# 21 cells solo vs --max-batch 16, then the localeval speed A/B at 16 streams, base (origin/main + recipe cap 16) vs fix, ABAB.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"; REF=${REF:?cold reference json}
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
B=/home/nvidia/intarsiaml-bi8/bin
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
COMMON="--backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0"
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
start() { # name bin args
  SPID=$(rsh "nohup $B/$2 serve $COMMON $3 > ~/intarsiaml-bi8/out/serve-$1.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && return 0; sleep 2; done
  echo "server $1 did not come up"; stop_server; return 1
}
if [[ "${SKIP_TESTS:-0}" != 1 ]]; then
  echo "== cargo gates"
  rsh "$RPATH cd ~/intarsiaml-bi8/src && INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_batched -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|pairs bitwise|bi8 engine|assert|left|right'" > "$OUT/gates.log"
  cat "$OUT/gates.log"
fi
echo "== served greedy identity"
for arm in "solo|intarsia|" "mb16|intarsia|--max-batch 16" "mb8|intarsia|--max-batch 8"; do
  IFS='|' read -r name bin args <<< "$arm"
  start "$name" "$bin" "$args" || continue
  case $name in
    solo) ns=1,4,16; reps=3 ;; mb16) ns=1,4,8,16; reps=3 ;; mb8) ns=8,16; reps=2 ;;
  esac
  python3 "$D/greedy.py" run "http://$HN:$PORT/v1" $ns $reps "$OUT/greedy-$name.json" > /dev/null
  python3 "$D/greedy.py" compare "$REF" "$OUT/greedy-$name.json" | tee "$OUT/greedy-$name.compare" | tail -1
  if [[ $name != mb8 ]]; then python3 "$D/../htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/eq-$name.json" > /dev/null; fi
  stop_server
done
python3 "$D/../htk/equal.py" compare "$OUT/eq-solo.json" "$OUT/eq-mb16.json" | tee "$OUT/eq-compare.log" | tail -1
if [[ "${SKIP_SPEED:-0}" != 1 ]]; then
  echo "== speed A/B, 16 streams"
  for arm in base-r1 fix-r1 base-r2 fix-r2; do
    if [[ $arm == base* ]]; then bin=base; args="--recipe /home/nvidia/intarsiaml-bi8/qwen3-mb16.toml --max-batch 16"; else bin=intarsia; args="--max-batch 16"; fi
    start "$arm" "$bin" "$args" || continue
    (cd /opt/code/deepeval && uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes 1024 --gen 256 --reps 3 \
      --concurrency 16 --label "bi8-$arm" --recipe "~/intarsiaml-bi8/bin/$bin serve $COMMON $args" 2>&1 | grep -E "saved|unim|tok/s|^\|" | tail -12) | tee "$OUT/speed-$arm.log"
    stop_server
  done
fi
echo "== done"
