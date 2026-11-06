#!/usr/bin/env bash
# Lane spark35 (docs/results/2026-11-05-spark35-numerics.md): shared setup for build.sh / w1.sh / kl.sh / serve.sh. Source it.
# Run the scripts INSIDE one fleetq slot on $HOST (FLEETQ_JOB set); they run on the Mac and ssh to the box.
: "${FLEETQ_JOB:?run under fleetq run / fleetq acquire}"
HOST=${HOST:-spark2}
case $HOST in spark1) HN=spark-4e77 ;; spark2) HN=spark-fa00 ;; *) echo "HOST must be spark1 or spark2"; exit 2 ;; esac
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(git -C "$H" rev-parse --show-toplevel)"
L='~/intarsiaml-s35'                  # lane dir on the box: $L/src (tree), $L/bin (binaries), $L/out (logs, dumps), $L/carriers
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RES=${RES:-$REPO/docs/results/2026-11-05-spark35-numerics/raw}; mkdir -p "$RES"
PORT=${PORT:-8196}
rsh() { ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB; $*"; }
free_gb() { rsh "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo"; }
SPID=""
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 20); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; return 0; }
# start_server <name> <bin> <env assignments, space separated or empty> <extra serve args...>
start_server() {
  local name=$1 bin=$2 envs=$3; shift 3
  SPID=$(rsh "mkdir -p $L/out; nohup env $envs $L/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq ${MAXSEQ:-73728} --mem-budget ${MEMB:-45} --profile-every 0 $* > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  local i; for i in $(seq 1 180); do curl -sf "http://$HN:$PORT/health" >/dev/null && { curl -s "http://$HN:$PORT/metrics" | grep -a '^intarsia_info' > "$RES/info-$name.txt"; return 0; }; sleep 2; done
  echo "server $name did not come up"; rsh "tail -8 $L/out/serve-$name.log"; return 1
}
# window budget: w1.sh exports S35_DEADLINE (epoch seconds); the arm loops skip an arm that does not fit
left_min() { echo $(( (${S35_DEADLINE:-$(( $(date +%s) + 99999 ))} - $(date +%s)) / 60 )); }
fits() { local need=$1 what=$2; if (( $(left_min) < need )); then echo "SKIP $what: $(left_min) min left, needs $need"; return 1; fi; return 0; }

# The commit the "dev" binary is built from: $DEV_REF, else the file scripts/spark35num/DEV_REF (a sha: the lever commit, never the D default-flip commit).
DEV_REF=${DEV_REF:-$(tr -d '[:space:]' < "$H/DEV_REF")}
DEV_SHA=$(git -C "$REPO" rev-parse "$DEV_REF^{commit}") || { echo "bad DEV_REF $DEV_REF"; exit 2; }
# ensure_dev: build bin/dev from $DEV_SHA when it is not already (idempotent), then REFUSE to go on unless bin/dev.ref says exactly that commit.
ensure_dev() {
  "$H/build.sh" "$DEV_SHA" dev || { echo "dev build from $DEV_SHA failed: no fallback, this job stops"; return 1; }
  rsh "test \"\$(cat $L/bin/dev.ref 2>/dev/null)\" = $DEV_SHA" || { echo "bin/dev is not built from $DEV_SHA"; return 1; }
}
