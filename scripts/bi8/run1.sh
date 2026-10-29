#!/usr/bin/env bash
# Lane bi8 GPU job 1 (spark2, --quiet gpu,cpu): the model-level prefix-reuse probe, then served greedy runs on one mb16 server per
# arm (reps on the same server), reuse on and INTARSIA_CB_NO_REUSE=1.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"
M=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
BIN=/home/nvidia/intarsiaml-bi8/bin/intarsia
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
arm() { # name "ENV=.." "serve args" ns reps
  local name=$1 envs=$2 args=$3 ns=$4 reps=$5
  echo "== $name env[$envs] args[$args]"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 36 --profile-every 0 $args > ~/intarsiaml-bi8/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  local up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server did not come up"; stop_server; return; }
  python3 "$D/greedy.py" run "http://$HN:$PORT/v1" "$ns" "$reps" "$OUT/$name.json"
  stop_server
}
if [[ "${SKIP_PROBE:-0}" != 1 ]]; then
  echo "== probe"
  rsh "$RPATH cd ~/intarsiaml-bi8/src && INTARSIA_BI8_PROBE=1 INTARSIA_QWEN3_DIR=$M INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_qwen3_8b_prefix_reuse_probe -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|COLD|SELFREUSE|SPLIT|CROSS|plan before|error'" > "$OUT/probe.log"
  tail -3 "$OUT/probe.log"
fi
rsh "sed 's/^max_batch = 8$/max_batch = 16/' ~/intarsiaml-bi8/src/recipes/qwen3.toml > ~/intarsiaml-bi8/qwen3-mb16.toml; grep -n '^max_batch' ~/intarsiaml-bi8/qwen3-mb16.toml"
R16="--recipe /home/nvidia/intarsiaml-bi8/qwen3-mb16.toml --max-batch 16"
arm mb16-noreuse "INTARSIA_CB_NO_REUSE=1" "$R16" 5,8,16 3
arm mb16-reuse "X=1" "$R16" 5,8,16 2
echo "== done"
