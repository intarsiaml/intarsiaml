#!/usr/bin/env bash
# Lane wg, ON THE MAC, as ONE fleetq command (radiance goes down inside it; wg-win.sh on tatooine restores it on every exit path):
#   fleetq run --host tatooine-r9700 --expect 28 --takes-down radiance --quiet gpu,cpu --label "wg served: 35B 1/4/8/16 streams off vs opt1 x3" -- \
#     env OUT=<dir> bash scripts/wgserve/served.sh
# Interleaved visits (pass 1: off opt1, pass 2: opt1 off, pass 3: off opt1; ARMS="main off opt1" adds the origin/main binary). Each visit:
#   start the server, wait for /health, localeval speed (1024 prompt, --gen 512, 3 reps, streams 1 + 4,8,16), stop. Window cap WINDOW_S (default 1350 s of radiance-down time, of which the last TAIL_S = 300 s are reserved for the tail below).
# Pre-step with radiance still up: compile the ablation bench (tail.sh build, cpu, ~2 min). Job budget: pre-step + WINDOW_S + restore < --expect 28 min.
# Tail (after the last visit, GPU idle): tail.sh = fp8 WMMA ablation bench (ABL=1) + model-loop B=1/4/16 for off | opt1 | fp8a8=all,bf16=off | fp8a8=off,bf16=all, x2.
# localeval runs inside this fleetq job (it inherits the job from the environment, so it does not queue again).
# Needs: stamps "tests" and "serve" fresh (wg-win.sh refuses otherwise): `remote.sh build` done after the last sync (and `build-base` for the main arm).
set -uo pipefail
OUT=${OUT:?set OUT}; mkdir -p "$OUT"
T=${T:-tatooine}; LE=/opt/code/deepeval; WINDOW_S=${WINDOW_S:-1350}; TAIL_S=${TAIL_S:-300}; ARMS=${ARMS:-"off opt1"}; PASSES=${PASSES:-3}; CONC=${CONC:-4,8,16}
LOG=$OUT/window.log; t0=$(date +%s)

say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
left() { echo $(( WINDOW_S - ($(date +%s) - t0) )); }
# pre-step, radiance still UP: the ablation bench build for the tail (a failure only skips the ablation)
ssh "$T" 'bash ~/wg-tail.sh build' </dev/null > "$OUT/tail-build.out" 2>&1
say "pre-step: $(tail -n 1 "$OUT/tail-build.out")"
t0=$(date +%s)   # the window clock starts when radiance goes down
# hold the radiance-down window on tatooine in the background (wg-win.sh stops radiance, restores it when the hold ends)
ssh "$T" "WG_NEED='tests serve' bash ~/wg-win.sh bash ~/wg-serve.sh hold $WINDOW_S" </dev/null > "$OUT/win.out" 2>&1 &
WP=$!
finish() { ssh "$T" 'bash ~/wg-serve.sh done' </dev/null >/dev/null 2>&1; wait $WP 2>/dev/null; say "window process ended"; tail -n 4 "$OUT/win.out" | tee -a "$LOG"; }
trap finish EXIT
for i in $(seq 1 60); do
  grep -q "stopping radiance" "$OUT/win.out" 2>/dev/null && break
  grep -q "REFUSING" "$OUT/win.out" 2>/dev/null && { say "REFUSED: $(cat "$OUT/win.out")"; exit 3; }
  sleep 3
done
grep -q "stopping radiance" "$OUT/win.out" || { say "window did not start"; exit 3; }
say "radiance down; arms $ARMS x $PASSES passes"
visit() {  # arm pass
  local arm=$1 name="$1-p$2"
  if (( $(left) < 200 + TAIL_S )); then say "SKIP $name: $(left) s left, tail needs $TAIL_S"; return 0; fi
  ssh "$T" "bash ~/wg-serve.sh start $arm" </dev/null > "$OUT/$name-start.out" 2>&1
  local up=0 i; for i in $(seq 1 100); do curl -sf -m 2 "http://$T:8090/health" >/dev/null 2>&1 && { up=1; break; }; sleep 3; done
  if [ $up = 0 ]; then
    say "$name: server did not come up"; ssh "$T" 'docker logs --tail 40 wg-serve 2>&1' </dev/null > "$OUT/$name-server.err" 2>&1
    ssh "$T" 'bash ~/wg-serve.sh stop' </dev/null; return 1
  fi
  say "$name: up ($(left) s left)"
  local cap=$(( $(left) - TAIL_S - 60 )); (( cap > 420 )) && cap=420
  (cd "$LE" && timeout "$cap" uv run localeval speed --url "http://$T:8090/v1" --model wg-35b --sizes 1024 --gen 512 --reps 3 \
     --concurrency "$CONC" --thinking off --label "wg-served-$name") > "$OUT/$name-localeval.out" 2>&1
  say "$name: localeval exit $? ($(left) s left)"
  local res; res=$(grep -o 'saved .*result.json' "$OUT/$name-localeval.out" | tail -1 | sed 's/^saved //')
  [ -n "$res" ] && [ -f "$res" ] && cp "$res" "$OUT/$name.result.json"
  ssh "$T" 'bash ~/wg-serve.sh stop' </dev/null; sleep 2
}
for p in $(seq 1 "$PASSES"); do
  order=($ARMS); (( p % 2 == 0 )) && order=($(echo $ARMS | tr ' ' '\n' | tac))
  for arm in "${order[@]}"; do visit "$arm" "$p"; done
done
say "visits done ($(left) s left)"
if (( $(left) > 90 )); then
  tl=$(( $(left) - 30 )); (( tl > TAIL_S )) && tl=$TAIL_S
  say "tail: ${tl} s budget"
  timeout $(( tl + 150 )) ssh "$T" "bash ~/wg-tail.sh run $tl" </dev/null > "$OUT/tail.out" 2>&1
  say "tail exit $?: $(tail -n 3 "$OUT/tail.out" | tr '\n' ' ')"
  scp -q "$T:wg-logs/tail/*" "$OUT/" 2>/dev/null
else say "tail skipped: $(left) s left"; fi
