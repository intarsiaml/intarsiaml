#!/usr/bin/env bash
# flipev2: spark2 (GB10) evidence for the herd and sjf default decisions. Phases (one fleetq job each):
#   build : origin/main CUDA CLI into ~/intarsiaml-flipev2 (CARGO_BUILD_JOBS=3, buildwatch inside build.sh)
#   hs    : `serve --max-batch 16 --prefix-store on`, 8 visits N H S B B S H N (N neither, H herd, S sjf, B both):
#           cell 2 (all visits): 8 x 16K shared + 1K unrelated short at +2 s; cell 1 (N/H visits): 16 on a shared 16K; cell 4 (N/H visits): 4 shared 16K + 2 unrelated 8K
#   age   : `serve --max-batch 4`, 7 visits F 4 8 16 16 8 4 F (F = fifo; numbers = INTARSIA_PREFILL_AGE with ORDER=shortest): cell 3, 2 decoders + 40K + 2 shorts
#   dec   : `serve --max-batch 8`, 4 visits F S S F: cell 5, 3 and 4 decoders (+ shorts)
#   clean : remove ~/intarsiaml-flipev2
# Run under: fleetq run --host spark2 --expect N --mem M --quiet gpu,cpu --label "flipev2 <phase>" -- scripts/flipev2/run.sh <phase>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/flipev2; OUT=$REPO/docs/results/2026-11-18-flipev2/raw; mkdir -p "$OUT"
HOST=${HOST:-spark2}; HN=spark-fa00; [[ $HOST == spark1 ]] && HN=spark-4e77; PORT=8198
L='~/intarsiaml-flipev2'; R="$L/src"; BIN="$L/bin/intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && { echo "usage: run.sh build|hs|age|dec|clean"; exit 1; }
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { local p; for p in "${PHASES[@]}"; do [[ $p == "$1" ]] && return 0; done; return 1; }
SPID=""; WPID=""
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-flipev2/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_server' EXIT; trap 'stop_server; exit 130' INT TERM

if has build; then
  SHA=$(git -C "$REPO" rev-parse HEAD); STAMP="flipev2-${SHA:0:12}"
  log "build $SHA; free: $(rsh "free -g | sed -n 2p; df -h ~ | tail -1; pgrep -fa 'intarsia.* serve' | head -3" | tr '\n' ' ')"
  rsh "mkdir -p $R $L/bin $L/out"
  rsh "[ -d $R/target ] || cp -a \$(ls -d ~/intarsia-ab35b/src/target ~/intarsia-batch-store/src/target 2>/dev/null | head -1) $R/target"
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOST" "tar --warning=no-unknown-keyword -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
  scp -q -o BatchMode=yes "$S/build.sh" "$HOST:intarsiaml-flipev2/build.sh"
  rsh "setsid -w bash $L/build.sh $STAMP < /dev/null > /dev/null 2>&1; tail -3 $L/out/build.log; grep -c KILLED $L/out/mem-build.txt; sort -k2 -n $L/out/mem-build.txt | head -1" | tee -a "$OUT/run.log"
  rsh "cat $L/out/build.log" > "$OUT/build.log"
  rsh "cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1" | tee -a "$OUT/run.log"
  rsh "test -x $L/bin/intarsia" || { log "build failed"; exit 2; }
  rsh "cp $L/out/mem-build.txt /dev/stdout" > "$OUT/mem-build.txt" 2>/dev/null
fi
rsh "mkdir -p $L/bench $L/out"
scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsiaml-flipev2/memwatch.sh"
scp -q -o BatchMode=yes "$S/herd_bench.py" "$S/q4.py" "$HOST:intarsiaml-flipev2/bench/"

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
  curl -sf "http://$HN:$PORT/v1/intarsia/prefix-store" > "$OUT/prefix-store-$1.json" 2>/dev/null
  rsh "cat $L/out/serve-$1.log" > "$OUT/serve-$1.log"
  scp -q -o BatchMode=yes "$HOST:intarsiaml-flipev2/out/*.json*" "$OUT/" 2>/dev/null
  log "min MemAvailable MB ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1"), KILLED: $(rsh "grep -c KILLED $L/out/mem-$1.txt")"
  stop_server
}

if has diag; then  # prefill speed of cold 16K prompts vs --max-batch and the store (spark2 ran 3.6x slower than flipev's spark1 numbers at mb16 + store)
  v=0
  for cfg in "mb16-store:--max-batch 16 --prefix-store on" "mb8-store:--max-batch 8 --prefix-store on" "mb16:--max-batch 16" "mb1:"; do
    v=$((v+1)); name="${cfg%%:*}"; sargs="${cfg#*:}"
    start_serve "diag-$name" "" --max-seq 24576 $sargs || continue
    C="rsh python3 $L/bench/herd_bench.py --url http://127.0.0.1:$PORT --arm $name --visit $name --out $L/out/diag.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1
    for seed in 701 702; do
      ( $C --mode distinct --seed $seed --n 3 --distinct-len 16448 2>&1 | tee -a "$OUT/run.log" ) &
      sleep 12   # mid-prefill: GPU power / clocks / busiest server threads
      rsh "nvidia-smi dmon -c 3 -s pu 2>&1 | tail -3; top -H -b -n1 -w 150 -p \$(pgrep -f 'intarsiaml-flipev2/bin/intarsia serve' | head -1) | sed -n 7,12p" | tee -a "$OUT/run.log"
      wait
    done
    finish_serve "diag-$name"
  done
fi

if has mixd; then  # cell 4 again with a fixed arrival order (20 ms apart): N H H N; order A = S U S S U S (flipev's), order B = the unrelated pair first
  v=0
  for arm in N H H N; do
    v=$((v+1)); name="$arm$v"
    envs=""; [[ $arm == H ]] && envs="INTARSIA_PREFIX_HERD=1"
    start_serve "mixd-$name" "$envs" --max-seq 24576 --max-batch 16 --prefix-store on || continue
    C="rsh python3 $L/bench/herd_bench.py --url http://127.0.0.1:$PORT --arm $arm --visit $name --out $L/out/mixd.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1
    for seed in 141 142 143; do $C --mode mix --seed $seed --n 4 --stagger 0.02 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --prompts mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"; done
    for seed in 151 152 153; do $C --mode mix --seed $seed --n 4 --stagger 0.02 --arrival 1,4,0,2,3,5 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --prompts mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"; done
    finish_serve "mixd-$name"
  done
fi

log "done"
if has clean; then rsh "rm -rf $L"; log "removed ~/intarsiaml-flipev2"; fi
