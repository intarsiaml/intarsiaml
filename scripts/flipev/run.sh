#!/usr/bin/env bash
# flipev: build origin/main's CUDA CLI on spark2 (~/intarsiaml-flipev), then
#   phase herd: 4 visits ABBA (herd off / on), `serve --max-batch 8 --prefix-store on`: prefix sweep 4K/32K, staggered 0.5 s / 2 s, 4 shared + 2 unrelated
#   phase q4:   4 visits ABBA (sjf off / on), `serve --max-batch 4`: ITL of 2-4 decodes while a 40K prompt prefills (+ short prompts arriving)
# Run under: fleetq run --host spark2 --expect 95 --mem 62 --quiet gpu,cpu --label "flipev ..." -- scripts/flipev/run.sh [phases: build herd q4]
# KEEP=1 keeps ~/intarsiaml-flipev at the end.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/flipev; OUT=$REPO/docs/results/2026-11-17-flipev/raw; mkdir -p "$OUT"
HOST=${HOST:-spark2}; HN=spark-fa00; [[ $HOST == spark1 ]] && HN=spark-4e77; PORT=8197
L='~/intarsiaml-flipev'; R="$L/src"; BIN="$L/bin/intarsia"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && PHASES=(build pack herd q4)
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { local p; for p in "${PHASES[@]}"; do [[ $p == "$1" ]] && return 0; done; return 1; }
SPID=""; WPID=""
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-flipev/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap 'stop_server' EXIT; trap 'stop_server; exit 130' INT TERM

if has build; then
  SHA=$(git -C "$REPO" rev-parse HEAD); STAMP="flipev-${SHA:0:12}"
  log "build $SHA; free: $(rsh "free -g | sed -n 2p; df -h ~ | tail -1" | tr '\n' ' ')"
  rsh "mkdir -p $R $L/bin $L/out"
  rsh "[ -d $R/target ] || cp -a \$(ls -d ~/intarsia-ab35b/src/target ~/intarsia-batch-store/src/target 2>/dev/null | head -1) $R/target"
  rsh "rm -rf $L/stage && mkdir -p $L/stage"
  git -C "$REPO" archive "$SHA" | ssh -o BatchMode=yes "$HOST" "tar --warning=no-unknown-keyword -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "cd $R && env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=4 MAX_JOBS=4 FLASHINFER_NVCC_THREADS=1 cargo build --release --locked --features cuda -p intarsia-cli -j 4 > $L/out/build.log 2>&1; grep -E '^error|Finished' $L/out/build.log | tail -5; cp $R/target/release/intarsia $L/bin/intarsia && $L/bin/intarsia --version | head -1" | tee -a "$OUT/run.log"
  rsh "cat $L/out/build.log" > "$OUT/build.log"
  rsh "test -x $L/bin/intarsia" || { log "build failed"; exit 2; }
fi
scp -q -o BatchMode=yes "$S/memwatch.sh" "$HOST:intarsiaml-flipev/memwatch.sh"

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
  log "min MemAvailable MB ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1"), KILLED: $(rsh "grep -c KILLED $L/out/mem-$1.txt")"
  stop_server
}

if has pack; then  # 8 distinct cold 8K prompts at --max-batch 8: store off (CUDA packed prefill on) vs --prefix-store on (packing refused)
  v=0
  for arm in S T T S; do  # S = store off, T = store on
    v=$((v+1)); name="$arm$v"
    sargs=""; [[ $arm == T ]] && sargs="--prefix-store on"
    start_serve "pack-$name" "" --max-seq 16384 --max-batch 8 $sargs || continue
    C="python3 $S/herd_bench.py --url http://$HN:$PORT --arm $arm --visit $name --out $OUT/pack.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1
    for seed in 71 72; do $C --mode distinct --seed $seed 2>&1 | tee -a "$OUT/run.log"; done
    finish_serve "pack-$name"
  done
fi

if has herd; then
  v=0
  for arm in A B B A; do
    v=$((v+1)); name="$arm$v"
    envs=""; [[ $arm == B ]] && envs="INTARSIA_PREFIX_HERD=1"
    start_serve "herd-$name" "$envs" --max-seq 49152 --max-batch 8 --prefix-store on || continue
    C="python3 $S/herd_bench.py --url http://$HN:$PORT --arm $arm --visit $name --out $OUT/herd.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1   # warm, not recorded
    for seed in 21 22 23; do $C --mode herd8 --seed $seed --shared 4160 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --seed $seed --shared 4160 2>&1 | tee -a "$OUT/run.log"; done
    for seed in 31 32; do $C --mode herd8 --seed $seed --shared 32832 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --seed $seed --shared 32832 2>&1 | tee -a "$OUT/run.log"; done
    for seed in 41 42; do $C --mode herd8 --seed $seed --stagger 0.5 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --seed $seed 2>&1 | tee -a "$OUT/run.log"; done
    for seed in 51 52; do $C --mode herd8 --seed $seed --stagger 2.0 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --seed $seed 2>&1 | tee -a "$OUT/run.log"; done
    for seed in 61 62; do $C --mode mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"; $C --mode solo --prompts mix --seed $seed --n 4 2>&1 | tee -a "$OUT/run.log"; done
    finish_serve "herd-$name"
  done
fi

if has q4; then
  v=0
  for arm in A B B A; do
    v=$((v+1)); name="$arm$v"
    envs="INTARSIA_CB_NO_REUSE=1"; [[ $arm == B ]] && envs="$envs INTARSIA_PREFILL_ORDER=shortest INTARSIA_PREFILL_AGE=8"
    start_serve "q4-$name" "$envs" --max-seq 49152 --max-batch 4 || continue
    Q="python3 $S/q4.py"; U="http://$HN:$PORT"
    $Q solo --url $U --out "$OUT/q4-solo-$name.json" 2>&1 | tee -a "$OUT/run.log"
    for rep in 1 2; do
      for cfg in "2 long" "4 long" "4 longshort" "2 longshort"; do
        set -- $cfg
        $Q run --url $U --arm $name --d $1 --scn $2 --rep $rep --solo "$OUT/q4-solo-$name.json" --out "$OUT/q4.jsonl" 2>&1 | tee -a "$OUT/run.log" | cut -c1-400
        sleep 4
      done
    done
    finish_serve "q4-$name"
  done
fi
log "done"
if [[ -z "${KEEP:-}" ]] && has q4; then rsh "rm -rf $L"; log "removed ~/intarsiaml-flipev"; fi
