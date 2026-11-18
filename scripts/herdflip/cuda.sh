#!/usr/bin/env bash
# herdflip, CUDA (spark1, 35B NVFP4): the prefix herd default gates. Phases (one fleetq job each):
#   build: BASE (origin/main, pinned) and this worktree's HEAD into ~/intarsiaml-herdflip/bin/{base,new}
#          fleetq run --host spark1 --expect 45 --mem 55 --load cpu --label "herdflip cuda build" -- scripts/herdflip/cuda.sh build
#   plan : `plan show --json` base vs new (unset, INTARSIA_PREFIX_HERD=0; with and without INTARSIA_PREFIX_STORE=on); loads the weights
#   run  : one fresh `serve --max-batch 8` per VISIT, the same herd-shaped cells in each (scripts/flipev/herd_bench.py), then scripts/herdflip/compare.py
#          fleetq run --host spark1 --expect 75 --mem 62 --quiet gpu,cpu --label "herdflip cuda gates" -- scripts/herdflip/cuda.sh plan run clean
#   clean: remove ~/intarsiaml-herdflip
# vLLM stays up beside the run (budget: --mem-budget 40 + 34 resident + 10 OS < 121 GB); memwatch kills the server below 15 GB MemAvailable.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/herdflip; OUT=$REPO/docs/results/2026-11-18-herdflip/raw/cuda; mkdir -p "$OUT"
BASE=${BASE:-11cb82fe43e2b1b857e09b72f45d28378554e955}
HOST=${HOST:-spark1}; HN=spark-4e77; PORT=8199
L='~/intarsiaml-herdflip'; R="$L/src"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PHASES=("$@"); [[ ${#PHASES[@]} -eq 0 ]] && { echo "usage: cuda.sh build|plan|run|clean"; exit 1; }
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
has() { local p; for p in "${PHASES[@]}"; do [[ $p == "$1" ]] && return 0; done; return 1; }
SPID=""; WPID=""
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3"; SPID=""; }
}
trap 'stop_server' EXIT; trap 'stop_server; exit 130' INT TERM

stage() { # stage <git rev> : the tree of that rev into $R
  rsh "rm -rf $L/stage && mkdir -p $L/stage $R"
  git -C "$REPO" archive "$1" | ssh -o BatchMode=yes "$HOST" "tar --warning=no-unknown-keyword -x -C $L/stage"
  rsh "rsync -a --delete --exclude target --exclude golden --exclude tools/.venv --exclude out $L/stage/ $R/ && rm -rf $L/stage"
  rsh "cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.toml' \) -print0 | xargs -0 touch"
}
build_one() { # build_one <name> <git rev>
  log "build $1 from $2; free: $(rsh "free -g | sed -n 2p; df -h ~ | tail -1" | tr '\n' ' ')"
  stage "$2"
  scp -q -o BatchMode=yes "$S/cuda_build.sh" "$HOST:intarsiaml-herdflip/cuda_build.sh"
  rsh "setsid -w bash $L/cuda_build.sh herdflip-$1 $L/out/build-$1.log < /dev/null > /dev/null 2>&1; tail -2 $L/out/build-$1.log; grep -c KILLED $L/out/mem-build-herdflip-$1.txt; sort -k2 -n $L/out/mem-build-herdflip-$1.txt | head -1" | tee -a "$OUT/run.log"
  rsh "cp $R/target/release/intarsia $L/bin/$1 && $L/bin/$1 --version | head -1" | tee -a "$OUT/run.log"
  rsh "test -x $L/bin/$1" || { log "build $1 failed"; exit 2; }
}
if has build; then
  rsh "mkdir -p $R $L/bin $L/out"
  rsh "[ -d $R/target ] || cp -a ~/intarsia-batch-store/src/target $R/target"
  build_one base "$BASE"
  build_one new "$(git -C "$REPO" rev-parse HEAD)"
  scp -q -o BatchMode=yes "$HOST:intarsiaml-herdflip/out/build-*.log" "$OUT/"
fi

if has plan; then
  mkdir -p "$OUT/plan"
  # plan show has no --prefix-store flag; the store is the INTARSIA_PREFIX_STORE env (read by every command)
  for cfg in "base||" "new||" "new0|INTARSIA_PREFIX_HERD=0|" "base_st||INTARSIA_PREFIX_STORE=on" "new_st||INTARSIA_PREFIX_STORE=on" "new0_st|INTARSIA_PREFIX_HERD=0|INTARSIA_PREFIX_STORE=on"; do
    IFS='|' read -r name e1 e2 <<< "$cfg"; bin=${name%%_*}; bin=${bin%0}
    rsh "env $e1 $e2 $L/bin/$bin plan show --model $M35 --backend cuda --max-seq 8192 --json" > "$OUT/plan/$name.json" 2> "$OUT/plan/$name.err"
    log "plan $name (bin $bin, env [$e1 $e2]): exit $? $(wc -c < "$OUT/plan/$name.json" | tr -d ' ') bytes sha $(shasum -a 256 < "$OUT/plan/$name.json" | cut -c1-16)"
  done
  cmp "$OUT/plan/base.json" "$OUT/plan/new.json" && log "plan: base == new (byte-identical, no store)"
  cmp "$OUT/plan/base.json" "$OUT/plan/new0.json" && log "plan: base == new0 (byte-identical, no store, HERD=0)"
  cmp "$OUT/plan/base_st.json" "$OUT/plan/new_st.json" && log "plan: base == new (byte-identical, INTARSIA_PREFIX_STORE=on)"
  cmp "$OUT/plan/base_st.json" "$OUT/plan/new0_st.json" && log "plan: base == new0 (byte-identical, INTARSIA_PREFIX_STORE=on, HERD=0)"
fi

start_serve() { # start_serve <name> <bin> "<ENV=..>" <serve args...>
  local name=$1 bin=$2 envs=$3; shift 3
  log "serve $name: bin $bin env [$envs] $* ; $(rsh "free -g | sed -n 2p")"
  rsh "rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "env $envs nohup $L/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --mem-budget 40 --profile-every 0 $* > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
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

if has run; then
  rsh "mkdir -p $L/bench $L/out"
  cat > "$OUT/memwatch.sh" <<'MW'
#!/usr/bin/env bash
# usage: memwatch.sh <pid> <name>: log MemAvailable every 2 s; kill -9 <pid> (and its children) if it drops under 15 GB
pid=$1; name=$2; f=$HOME/intarsiaml-herdflip/out/mem-$name.txt
while kill -0 "$pid" 2>/dev/null; do
  a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); echo "$(date +%T) $a" >> "$f"
  if [ "$a" -lt 15360 ]; then pkill -9 -P "$pid"; kill -9 "$pid"; echo "KILLED low mem $a" >> "$f"; fi
  sleep 2
done
MW
  scp -q -o BatchMode=yes "$OUT/memwatch.sh" "$HOST:intarsiaml-herdflip/memwatch.sh"; rm -f "$OUT/memwatch.sh"
  scp -q -o BatchMode=yes "$REPO/scripts/flipev/herd_bench.py" "$HOST:intarsiaml-herdflip/bench/"
  rsh "systemctl is-active earlyoom" | tee -a "$OUT/run.log"
  # visit name | env | binary | serve args. on_* = --prefix-store on (herd acts), off_* = store off (herd inert)
  VISITS=(
    "on_new||new|--max-seq 17408 --max-batch 8 --prefix-store on"
    "on_base||base|--max-seq 17408 --max-batch 8 --prefix-store on"
    "on_zero|INTARSIA_PREFIX_HERD=0|new|--max-seq 17408 --max-batch 8 --prefix-store on"
    "on_one|INTARSIA_PREFIX_HERD=1|new|--max-seq 17408 --max-batch 8 --prefix-store on"
    "off_new||new|--max-seq 17408 --max-batch 8"
    "off_base||base|--max-seq 17408 --max-batch 8"
  )
  for v in "${VISITS[@]}"; do
    IFS='|' read -r name envs bin sargs <<< "$v"
    start_serve "$name" "$bin" "$envs" $sargs || continue
    C="rsh python3 $L/bench/herd_bench.py --url http://127.0.0.1:$PORT --arm $name --visit $name --out $L/out/herd.jsonl"
    $C --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1   # warm, not recorded
    # herd-shaped cells: 8 requests, one new 16448-token shared prefix + 200 own tokens, released at once / 0.5 s / 2 s apart; then the same prompts one at a time
    for cell in "1 0" "2 0.5" "3 2"; do
      set -- $cell
      $C --mode herd8 --n 8 --seed $1 --stagger $2 2>&1 | tee -a "$OUT/run.log"
    done
    for seed in 1 2 3; do $C --mode solo --n 8 --seed $seed 2>&1 | tee -a "$OUT/run.log"; done
    finish_serve "$name"
  done
  rsh "cat $L/out/herd.jsonl" > "$OUT/herd.jsonl"
  python3 "$S/compare.py" "$OUT/herd.jsonl" --groups on:on_new,on_base,on_zero,on_one off:off_new,off_base --engaged on_new,on_one --inert on_base,on_zero,off_new,off_base 2>&1 | tee "$OUT/compare.txt" | tee -a "$OUT/run.log"
fi

if has run; then for f in "$OUT"/serve-*.log; do echo "$(basename "$f"): $(grep -a "prefix herd" "$f" | head -1 | cut -c1-120)"; done | tee -a "$OUT/run.log"; fi
log "done"
if has clean; then rsh "rm -rf $L"; log "removed ~/intarsiaml-herdflip"; fi
