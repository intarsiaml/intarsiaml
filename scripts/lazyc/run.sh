#!/bin/sh
# lazyc gates on spark1, run ON spark1 inside a fleetq slot (FLEETQ_JOB set):  run.sh PHASE   (PHASE = time | cold | ids)
# Env: L (work dir, default ~/intarsia-lazyc), BIN, M35, M27, PORT. POSIX sh, no arrays.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
L=${L:-$HOME/intarsia-lazyc}; BIN=${BIN:-$L/bin/intarsia}; PORT=${PORT:-18700}; OUT=${OUT:-$L/out}; mkdir -p "$OUT"
M35=${M35:-/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5}
M27=${M27:-/home/nvidia/models/Qwen3.8-27B-Quark-AWQ-MXFP4}
P4=${P4:-$L/prompt-4096.txt}
COMMON="--backend cuda --served-model-name intarsia --max-seq 40000 --mem-budget 40 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
PID=""
now() { date +%s.%N; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
stop_server() {
  [ -n "$PID" ] || return 0
  kill -INT "$PID" 2>/dev/null
  i=0; while [ $i -lt 60 ] && kill -0 "$PID" 2>/dev/null; do sleep 1; i=$((i+1)); done
  kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; PID=""
}
# memory guard: below 10 GB available, kill our servers
( while :; do a=$(avail_gb); echo "$(date +%T) $a" >> "$OUT/memwatch.log"
    [ "${a:-99}" -lt 10 ] && { echo "MEM LOW $a GB: killing" >> "$OUT/memwatch.log"; pkill -9 -f "$BIN serve"; }
    sleep 2; done ) &
GUARD=$!
trap 'stop_server; kill $GUARD 2>/dev/null' EXIT INT TERM
# start NAME LAZY MODEL FLAGS [HOMEDIR]: starts a server, waits for /health, sets UP (seconds), leaves PID set
start() {
  name=$1; lazy=$2; model=$3; flags=$4; hm=${5:-$HOME}
  t0=$(now)
  HOME=$hm INTARSIA_ADMIT_LAZY=$lazy $BIN serve --model "$model" $COMMON $flags > "$OUT/$name.out" 2> "$OUT/$name.err" < /dev/null &
  PID=$!
  i=0; while [ $i -lt 1200 ]; do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 "$PID" 2>/dev/null || { echo "## $name: server died"; tail -5 "$OUT/$name.err"; PID=""; UP=-1; return 1; }
    sleep 0.2; i=$((i+1)); done
  t1=$(now); UP=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.2f", b-a}')
  echo "## $name lazy=$lazy up ${UP}s | $(grep -a '^loaded' "$OUT/$name.err" | head -1) | $(grep -a '^admission' "$OUT/$name.err" | head -1) | avail $(avail_gb) GB"
}
fresh_home() { # fresh_home DIR: a HOME with an empty admission cache and the rest of ~/.cache linked
  rm -rf "$1"; mkdir -p "$1/.cache/intarsia"
  for e in "$HOME"/.cache/*; do b=$(basename "$e"); [ "$b" = intarsia ] || ln -s "$e" "$1/.cache/$b"; done
}
phase=${1:?phase}
case $phase in
time)  # warm-up (lazy on, fills the cache for the fresh binary), then A B B A per model, 2 reps per arm
  n=0
  for mdl in 35b 27b; do
    if [ $mdl = 35b ]; then M=$M35; F="--max-batch 2"; else M=$M27; F=""; fi
    start "t-$mdl-warm" 1 "$M" "$F" && stop_server
    start "t-$mdl-warm2" 1 "$M" "$F" && stop_server
    for lz in 0 1 1 0; do n=$((n+1)); start "t-$mdl-r$n-lazy$lz" $lz "$M" "$F" && echo "TIME $mdl lazy=$lz up=$UP"; stop_server; done
    n=0
  done ;;
cold)  # fresh cache dir per arm: 35B --spec off cold, then 35B default (mixed hit/miss), then 27B cold; compare cache bytes and plans
  for lz in 0 1; do
    H=$L/home-lazy$lz; fresh_home "$H"
    start "c-35b-specoff-lazy$lz" $lz "$M35" "--max-batch 2 --spec off" "$H" && { curl -sf "http://127.0.0.1:$PORT/v1/intarsia/plan" > "$OUT/c-35b-specoff-lazy$lz.plan.json"; echo "COLD 35b-specoff lazy=$lz up=$UP"; }; stop_server
    start "c-35b-default-lazy$lz" $lz "$M35" "--max-batch 2" "$H" && { curl -sf "http://127.0.0.1:$PORT/v1/intarsia/plan" > "$OUT/c-35b-default-lazy$lz.plan.json"; echo "COLD 35b-default lazy=$lz up=$UP"; }; stop_server
    start "c-27b-default-lazy$lz" $lz "$M27" "" "$H" && { curl -sf "http://127.0.0.1:$PORT/v1/intarsia/plan" > "$OUT/c-27b-default-lazy$lz.plan.json"; echo "COLD 27b-default lazy=$lz up=$UP"; }; stop_server
  done
  echo "== admission cache files"; ls -l $L/home-lazy0/.cache/intarsia/admission $L/home-lazy1/.cache/intarsia/admission
  ok=1
  for f in "$L"/home-lazy0/.cache/intarsia/admission/*; do b=$(basename "$f")
    if cmp "$f" "$L/home-lazy1/.cache/intarsia/admission/$b"; then echo "GATE2 $b identical"; else echo "GATE2 $b DIFFERS"; ok=0; fi; done
  [ "$(ls $L/home-lazy0/.cache/intarsia/admission | wc -l)" = "$(ls $L/home-lazy1/.cache/intarsia/admission | wc -l)" ] || { echo "GATE2 file count differs"; ok=0; }
  [ $ok = 1 ] && echo "GATE2 PASS" || echo "GATE2 FAIL"
  for c in 35b-specoff 35b-default 27b-default; do
    if cmp "$OUT/c-$c-lazy0.plan.json" "$OUT/c-$c-lazy1.plan.json"; then echo "GATE2plan $c (cold/mixed) identical"; else echo "GATE2plan $c DIFFERS"; fi; done ;;
ids)  # warm cache: served plan and greedy ids, lazy 0 vs 1, three configs
  for cfg in 35b-default 35b-specoff 27b-default; do
    case $cfg in 35b-default) M=$M35; F="--max-batch 2";; 35b-specoff) M=$M35; F="--max-batch 2 --spec off";; *) M=$M27; F="";; esac
    for lz in 0 1; do
      start "i-$cfg-lazy$lz" $lz "$M" "$F" && {
        curl -sf "http://127.0.0.1:$PORT/v1/intarsia/plan" > "$OUT/i-$cfg-lazy$lz.plan.json"
        python3 "$L/ids.py" "http://127.0.0.1:$PORT/v1" "$OUT/i-$cfg-lazy$lz.ids.json" "$P4" 256 > "$OUT/i-$cfg-lazy$lz.ids.log" 2>&1; cat "$OUT/i-$cfg-lazy$lz.ids.log"; }
      stop_server
    done
    if cmp "$OUT/i-$cfg-lazy0.plan.json" "$OUT/i-$cfg-lazy1.plan.json"; then echo "GATE3 $cfg plan identical"; else echo "GATE3 $cfg plan DIFFERS"; fi
    python3 - "$OUT" "$cfg" <<'PY'
import json, sys
r, c = sys.argv[1:3]; a = json.load(open(f"{r}/i-{c}-lazy0.ids.json")); b = json.load(open(f"{r}/i-{c}-lazy1.ids.json"))
for k in a:
    same = a[k]["ids"] == b[k]["ids"]
    print(f"GATE4 {c} {k}: ids lazy0==lazy1 {same} ({len(a[k]['ids'])} tokens, prompt {a[k]['usage']['prompt_tokens']})", "" if a[k]["spec"] == b[k]["spec"] else f"spec counters differ {a[k]['spec']} vs {b[k]['spec']}")
PY
  done ;;
esac
