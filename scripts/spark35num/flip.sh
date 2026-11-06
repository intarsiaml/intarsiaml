#!/usr/bin/env bash
# Lane gdn-bf16-default (user decision 2026-11-06: flip the C default, keep D opt-in): one fleetq slot on $HOST (spark2), about 30 min.
#   build first (--load cpu):  build.sh <origin/main sha> main ; build.sh <head sha> dev      (bin/main = the f32 default, bin/dev = the head with the flip)
#   fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 35 --label "gdn-bf16-default: gates + speed" -- scripts/spark35num/flip.sh
# 1. plan show (text + JSON), main vs head: the 8B and the 27B must be byte-identical; the 35B may differ only in lines that mention the GDN state
#    (the head with INTARSIA_GDN_STATE=f32 must be byte-identical to main for all three).
# 2. greedy ids (6 prompts x 256 tokens): the off switch (head, INTARSIA_GDN_STATE=f32) must equal main; the head default is reported against main.
# 3. speed, 16 streams (1k prompts, gen 256, --max-batch 16), on (head default) against off (head, INTARSIA_GDN_STATE=f32), ROUNDS rounds, the order
#    reversed every second round (on off / off on / on off ...).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
trap 'stop_server' EXIT INT TERM
O=$RES/flip; mkdir -p "$O"
ROUNDS=${ROUNDS:-3}
HUB=/home/nvidia/.cache/huggingface/hub
M8=$(rsh "ls -d $HUB/models--Qwen--Qwen3-8B/snapshots/* | head -1")
M27=${M27:-$(rsh "ls -d $HUB/models--Jundot--Qwen3.8-27B-oQ4e-mtp/snapshots/* | head -1")}
fails=0
echo "flip.sh on $HOST $(date +%T): main=$(rsh "cat $L/bin/main.ref") head=$(rsh "cat $L/bin/dev.ref"); MemAvailable $(free_gb) GB"

# ---- 1. plan show
plan() { # plan <name> <bin> <env> <model> <ext>
  local name=$1 bin=$2 envs=$3 model=$4 ext=$5
  rsh "env $envs $L/bin/$bin plan show --backend cuda --model $model --max-seq 8192 --mem-budget 40 $([[ $ext = json ]] && echo --json) < /dev/null > $L/out/flip-plan-$name.$ext 2> $L/out/flip-plan-$name.$ext.err; echo \$?"
}
for m in 8b:$M8 27b:$M27 35b:$M35; do
  tag=${m%%:*}; model=${m#*:}
  for ext in txt json; do
    rcm=$(plan "$tag-main" main "" "$model" $ext); rch=$(plan "$tag-head" dev "" "$model" $ext); rco=$(plan "$tag-off" dev "INTARSIA_GDN_STATE=f32" "$model" $ext)
    echo "plan $tag $ext: exit main=$rcm head=$rch head-off=$rco"
    if [[ $rcm != 0 || $rch != 0 || $rco != 0 ]]; then echo "FAIL plan $tag $ext did not run (see $O/flip-plan-$tag-*.err)"; fails=$((fails+1)); continue; fi
    if rsh "cmp -s $L/out/flip-plan-$tag-main.$ext $L/out/flip-plan-$tag-off.$ext"; then echo "PASS plan $tag $ext: head with INTARSIA_GDN_STATE=f32 == main (byte-identical)"; else echo "FAIL plan $tag $ext: off switch differs from main"; fails=$((fails+1)); fi
    if rsh "cmp -s $L/out/flip-plan-$tag-main.$ext $L/out/flip-plan-$tag-head.$ext"; then echo "PASS plan $tag $ext: head default == main (byte-identical)"
    else
      echo "plan $tag $ext: head default differs from main; differing lines:"
      rsh "diff $L/out/flip-plan-$tag-main.$ext $L/out/flip-plan-$tag-head.$ext | grep -a '^[<>]' | cut -c1-200" | tee "$O/plan-$tag-$ext.diff"
      if [[ $tag = 35b ]] && ! grep -a -v -i -E 'gdn|state|recurrent|slot' "$O/plan-$tag-$ext.diff" | grep -q .; then echo "PASS plan 35b $ext: differs only in GDN state lines"; else echo "FAIL plan $tag $ext: differs outside the GDN state lines"; fails=$((fails+1)); fi
    fi
  done
done
for t in 8b 27b 35b; do scp -q -o BatchMode=yes "$HOST:intarsiaml-s35/out/flip-plan-$t-*" "$O/" 2>&1 | tail -1; done

# ---- 2. greedy ids
G="$H/../fp8kv/greedy.py"; MAXSEQ=20480 MEMB=45; export MAXSEQ MEMB
arm() { # arm <name> <bin> <env>
  start_server "flip-$1" "$2" "$3" || { fails=$((fails+1)); return 1; }
  python3 "$G" run --url "http://$HN:$PORT/v1" --gen 256 --out "$O/ids-$1.json" > "$O/ids-$1.log" 2>&1; tail -1 "$O/ids-$1.log"
  stop_server
}
arm main main ""; arm off dev "INTARSIA_GDN_STATE=f32"; arm on dev ""
if python3 "$G" compare "$O/ids-main.json" "$O/ids-off.json" > "$O/cmp-main-vs-off.txt" 2>&1; then echo "PASS greedy ids: off switch == main (6 prompts x 256)"; else echo "FAIL greedy ids: off switch != main"; fails=$((fails+1)); fi
python3 "$G" compare "$O/ids-main.json" "$O/ids-on.json" > "$O/cmp-main-vs-on.txt" 2>&1; echo "greedy ids head default vs main (reported, differs by design):"; cat "$O/cmp-main-vs-on.txt"

# ---- 3. speed
MAXSEQ=8192 MEMB=40; export MAXSEQ MEMB
le() { # le <label> <recipe> <sizes> <gen> <conc>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url "http://$HN:$PORT/v1" --model intarsia --thinking off --sizes "$3" --gen "$4" --reps 3 --concurrency "$5" \
      --label "$1" --no-unim --recipe "$2" ) > "$O/$1.out" 2>&1
  echo "$1: $(grep -aE 'streams: aggregate' "$O/$1.out" | tr '\n' ' ')"
}
run_arm() { # run_arm <round> <on|off>
  local r=$1 a=$2 envs=""; [[ $a = off ]] && envs="INTARSIA_GDN_STATE=f32"
  echo "== round $r arm $a $(date +%T)"
  if start_server "flip-speed-$a-r$r" dev "$envs" --max-batch 16; then
    le "gdn-$a-burst-r$r" "gdn-bf16-default speed: head, $a (${envs:-recipe default}), --max-batch 16" 1024 256 4,8,16
    stop_server
  else fails=$((fails+1)); fi
}
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 == 0 )); then run_arm "$r" off; run_arm "$r" on; else run_arm "$r" on; run_arm "$r" off; fi
done
python3 - "$O" "$ROUNDS" <<'PY'
import re, sys
o, n = sys.argv[1], int(sys.argv[2])
agg = {}
for a in ("on", "off"):
    for r in range(1, n + 1):
        t = open(f"{o}/gdn-{a}-burst-r{r}.out", errors="replace").read()
        agg[(a, r)] = {int(s): float(v) for s, v in re.findall(r"(\d+) streams: aggregate ([\d.]+) tok/s", t)}
print("\n16-stream aggregate tok/s (on = recipe default bf16 state, off = INTARSIA_GDN_STATE=f32):")
for s in (4, 8, 16):
    on = [agg[("on", r)].get(s) for r in range(1, n + 1)]; off = [agg[("off", r)].get(s) for r in range(1, n + 1)]
    if None in on + off: print(s, "missing"); continue
    m = lambda x: sum(x) / len(x)
    print(f"{s:>2} streams: on {on} mean {m(on):.1f} | off {off} mean {m(off):.1f} | gain {100 * (m(on) / m(off) - 1):+.1f} % (per round " + ", ".join(f"{100 * (a / b - 1):+.1f}" for a, b in zip(on, off)) + ")")
PY
echo "flip.sh done $(date +%T) fails=$fails"
exit $(( fails > 0 ))
