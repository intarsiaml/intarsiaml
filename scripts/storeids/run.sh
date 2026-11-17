#!/usr/bin/env bash
# storeids: why do `serve --max-batch 8 --prefix-store on` and store off give different greedy ids? (CUDA, Qwen3.6-35B-A3B NVFP4, spark1)
# Builds origin/main (BASE) into ~/intarsiaml-storeids, then one server visit per config (see VISITS), each running scripts/storeids/ids_bench.py.
# Run under: fleetq run --host spark1 --expect 80 --mem 62 --quiet gpu,cpu --label "storeids ..." -- scripts/storeids/run.sh
# KEEP=1 keeps ~/intarsiaml-storeids at the end. ONLY="name name" runs just those visits (the build is skipped when the binary exists).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/storeids; OUT=$REPO/docs/results/2026-11-17-storeids/raw; mkdir -p "$OUT"
BASE=${BASE:-d6c2d092815faefdb9b55b0034363115c4864e15}
HOST=${HOST:-spark1}; HN=spark-4e77; [[ $HOST == spark2 ]] && HN=spark-fa00; PORT=8199
L='~/intarsiaml-storeids'; R="$L/src"; BIN="$L/bin/intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
SPID=""; WPID=""
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-storeids/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_server' EXIT; trap 'stop_server; exit 130' INT TERM

# visit name | env | extra serve args | modes
# solo = the single-sequence engine (no --max-batch): the reference. b* = --max-batch 8.
VISITS=(
"s_def||--max-seq 16384|seq,lp"
"s_spec0||--max-seq 16384 --spec off|seq,lp"
"s_spec0_nomacro|INTARSIA_PREFILL_MACRO=0|--max-seq 16384 --spec off|seq"
"s_store||--max-seq 16384 --prefix-store on|seq,lp"
"s_store_spec0||--max-seq 16384 --prefix-store on --spec off|seq"
"b_off||--max-seq 16384 --max-batch 8|burst,lp"
"b_store||--max-seq 16384 --max-batch 8 --prefix-store on|burst,lp"
"b_off_spec0||--max-seq 16384 --max-batch 8 --spec off|burst"
"b_store_spec0||--max-seq 16384 --max-batch 8 --prefix-store on --spec off|burst"
"b_off_nomacro|INTARSIA_PREFILL_MACRO=0|--max-seq 16384 --max-batch 8|burst"
"b_off_plain|INTARSIA_PREFILL_MACRO=0 INTARSIA_PACK_PREFILL=0|--max-seq 16384 --max-batch 8 --spec off|burst"
)

if [[ -z "${ONLY:-}" ]] || ! rsh "test -x $BIN"; then
  STAMP="storeids-${BASE:0:12}"
  log "build $BASE; free: $(rsh "free -g | sed -n 2p; df -h ~ | tail -1" | tr '\n' ' ')"
  rsh "mkdir -p $R $L/bin $L/out"
  rsh "[ -d $R/target ] || cp -a ~/intarsiaml-mcdiag/src/target $R/target"
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$BASE" | ssh -o BatchMode=yes "$HOST" "tar --warning=no-unknown-keyword -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "cd $R && env CARGO_TARGET_DIR=$R/target INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli -j 4 > $L/out/build.log 2>&1; grep -E '^error|Finished' $L/out/build.log | tail -5; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1" | tee -a "$OUT/run.log"
  rsh "cat $L/out/build.log" > "$OUT/build.log"
  rsh "test -x $L/bin/intarsia" || { log "build failed"; exit 2; }
fi
scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsiaml-storeids/memwatch.sh"
rsh "systemctl is-active earlyoom" | tee -a "$OUT/run.log"

start_serve() { # start_serve <name> "<ENV=..>" <serve args...>
  local name=$1 envs=$2; shift 2
  log "serve $name: env [$envs] $* ; $(rsh "free -g | sed -n 2p")"
  rsh "rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "env $envs nohup $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --mem-budget 40 --profile-every 0 $* > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $L/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 120); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 3; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $L/out/serve-$name.log" | tee -a "$OUT/run.log"; stop_server; return 1; }
  log "serve $name up; $(rsh "free -g | sed -n 2p")"
}
finish_serve() {
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/metrics-$1.txt" 2>/dev/null
  rsh "cat $L/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1"), KILLED: $(rsh "grep -c KILLED $L/out/mem-$1.txt")"
  stop_server
}

for v in "${VISITS[@]}"; do
  IFS='|' read -r name envs sargs modes <<< "$v"
  if [[ -n "${ONLY:-}" ]] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  start_serve "$name" "$envs" $sargs || continue
  python3 "$S/ids_bench.py" --url "http://$HN:$PORT" --visit "$name" --modes "$modes" --out "$OUT/ids.jsonl" 2>&1 | tee -a "$OUT/run.log"
  finish_serve "$name"
done
log "done"
if [[ -z "${KEEP:-}" && -z "${ONLY:-}" ]]; then rsh "rm -rf $L"; log "removed ~/intarsiaml-storeids"; fi
