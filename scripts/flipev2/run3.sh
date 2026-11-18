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
DT=${DT:-800}
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

if has hs; then
  v=0
  for arm in N H S B B S H N; do
    v=$((v+1)); name="$arm$v"
    envs=""
    [[ $arm == H || $arm == B ]] && envs="INTARSIA_PREFIX_HERD=1"
    [[ $arm == S || $arm == B ]] && envs="$envs INTARSIA_PREFILL_ORDER=shortest INTARSIA_PREFILL_AGE=8"
    start_serve "hs-$name" "$envs" --max-seq 24576 --max-batch 16 --prefix-store on || continue
    C="rsh python3 $L/bench/herd_bench.py --url http://127.0.0.1:$PORT --arm $arm --visit $name --out $L/out/hs.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1   # warm, not recorded
    for seed in 111 112 113; do   # cell 2
      $C --mode hs --seed $seed 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --prompts hs --seed $seed 2>&1 | tee -a "$OUT/run.log"
    done
    if [[ $arm == N || $arm == H ]]; then
      for seed in 121 122 123; do   # cell 1
        $C --mode herd8 --n 16 --seed $seed 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --n 16 --seed $seed 2>&1 | tee -a "$OUT/run.log"
      done
      for seed in 131 132 133 134; do   # cell 4
        $C --mode mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --prompts mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"
      done
    fi
    finish_serve "hs-$name"
  done
fi

q4visits() { # q4visits <phase> <mb> <reps> "<cfgs ; separated>" <arm...>   arm = F (fifo) | number (age) | S (sjf default age)
  local ph=$1 mb=$2 reps=$3 cfgs=$4; shift 4
  local v=0 arm name envs cfg rep
  for arm in "$@"; do
    v=$((v+1)); name="$arm$v"
    envs="INTARSIA_CB_NO_REUSE=1"
    [[ $arm == S ]] && envs="$envs INTARSIA_PREFILL_ORDER=shortest INTARSIA_PREFILL_AGE=8"
    [[ $arm =~ ^[0-9]+$ ]] && envs="$envs INTARSIA_PREFILL_ORDER=shortest INTARSIA_PREFILL_AGE=$arm"
    start_serve "$ph-$name" "$envs" --max-seq 49152 --max-batch $mb || continue
    Q="rsh python3 $L/bench/q4.py"; U="http://127.0.0.1:$PORT"
    $Q solo --url $U --dec-tokens $DT --out $L/out/$ph-solo-$name.json 2>&1 | tee -a "$OUT/run.log"
    for rep in $(seq 1 $reps); do
      IFS=';' read -ra CF <<< "$cfgs"
      for cfg in "${CF[@]}"; do
        set -- $cfg
        $Q run --url $U --arm "$arm" --visit "$name" --d $1 --scn $2 --rep $rep --dec-tokens $DT --solo $L/out/$ph-solo-$name.json --out $L/out/$ph.jsonl 2>&1 | tee -a "$OUT/run.log" | cut -c1-400
        sleep 4
      done
    done
    finish_serve "$ph-$name"
  done
}
has age && q4visits age 4 3 "2 longshort" F 4 8 16 16 8 4 F
has dec && q4visits dec 8 2 "3 long;3 longshort;4 long;4 longshort" F S S F

log "done"
if has clean; then rsh "rm -rf $L"; log "removed ~/intarsiaml-flipev2"; fi
