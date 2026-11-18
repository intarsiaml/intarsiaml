#!/bin/zsh
# herdflip, Metal (endor, 35B NVFP4): the prefix herd default gates, inside a fleetq slot on endor:
#   fleetq run --host endor --expect 60 --quiet gpu,cpu --label "herdflip metal gates" -- scripts/herdflip/metal.sh
# (b) `plan show --json` base vs new (unset, INTARSIA_PREFIX_HERD=0), with and without INTARSIA_PREFIX_STORE=on
# (a) one fresh `serve --max-batch 8` per visit, the same herd-shaped cells in each (scripts/mflip/burst.py: 8 requests, one new 16448-token prefix + 200 own, 0 / 0.5 s / 2 s apart,
#     then the same prompts one at a time), then scripts/herdflip/compare.py. Default speculation (the shipped Metal MTP recipe).
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=${0:A:h:h:h}; RAW=$W/docs/results/2026-11-18-herdflip/raw/metal; mkdir -p $RAW/plan
BASE=$W/target-flip-base/intarsia-base; NEW=$W/target-flip/intarsia-new; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
PORT=${PORT:-18691}; URL=http://127.0.0.1:$PORT; S=$W/scripts/herdflip
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ "$(th)" != 0 ] && [ $w -lt ${COOL:-90} ]; do sleep 5; w=$((w+5)); done; echo "cooled ${w}s thermal $(th)"; }
show() { # show <name> <binary> <env...>
  local n=$1 b=$2; shift 2
  env X=1 "$@" $b plan show --model $M --backend metal --spec mtp --spec-k 2 --max-seq 8192 --json < /dev/null > $RAW/plan/$n.json 2> $RAW/plan/$n.err
  echo "plan $n: exit $? $(wc -c < $RAW/plan/$n.json | tr -d ' ') bytes sha $(shasum -a 256 < $RAW/plan/$n.json | cut -c1-16)"
}
show base $BASE; show new $NEW; show new0 $NEW INTARSIA_PREFIX_HERD=0
show base_st $BASE INTARSIA_PREFIX_STORE=on; show new_st $NEW INTARSIA_PREFIX_STORE=on; show new0_st $NEW INTARSIA_PREFIX_HERD=0 INTARSIA_PREFIX_STORE=on
for p in "base new" "base new0" "base_st new_st" "base_st new0_st"; do set -- ${=p}; cmp -s $RAW/plan/$1.json $RAW/plan/$2.json && echo "plan: $1 == $2 (byte-identical)" || echo "plan: $1 != $2 DIFFERS"; done
[ -s $RAW/plan/base.json ] && [ -s $RAW/plan/base_st.json ] || echo "plan FAIL: empty plan output"

# visit name | env | binary | extra serve args
VISITS=(
  "on_new|X=1|$NEW|--prefix-store on"
  "on_base|X=1|$BASE|--prefix-store on"
  "on_zero|INTARSIA_PREFIX_HERD=0|$NEW|--prefix-store on"
  "on_one|INTARSIA_PREFIX_HERD=1|$NEW|--prefix-store on"
  "off_new|X=1|$NEW|"
  "off_base|X=1|$BASE|"
)
rm -f $RAW/herd.jsonl
for v in $VISITS; do
  name=${v%%|*}; r=${v#*|}; E=${r%%|*}; r=${r#*|}; B=${r%%|*}; SA=${r#*|}
  D=$RAW/$name; mkdir -p $D; echo "== visit $name env $E $(cool)"
  env INTARSIA_CB_NO_REUSE=1 $E $B serve --backend metal --model $M --served-model-name intarsia --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT --max-seq 17408 --max-batch 8 ${=SA} > $D/server.out 2> $D/server.err < /dev/null &
  TP=$!; up=0
  for i in {1..400}; do curl -sf -m 2 $URL/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $TP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $name: $(tail -3 $D/server.err)"; kill $TP 2>/dev/null; continue; }
  C="python3 $W/scripts/mflip/burst.py --url $URL --arm $name --visit $name --cell herd --shared 16448 --own 200 --out $RAW/herd.jsonl"
  python3 $W/scripts/mflip/burst.py --url $URL --arm W --visit $name --shared 512 --own 20 --n 2 --seed 900 --max-tokens 8 --mode burst --out /dev/null > /dev/null 2>&1   # warm, not recorded
  for cell in "1 0" "2 0.5" "3 2"; do set -- ${=cell}; ${=C} --mode burst --seed $1 --n 8 --gap $2 | cut -c1-400; done
  for s in 1 2 3; do ${=C} --mode solo --seed $s --n 8 | cut -c1-400; done
  curl -sf $URL/v1/intarsia/prefix-store > $D/prefix-store.json; curl -sf $URL/metrics > $D/metrics.txt 2>/dev/null
  echo "rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"
  grep -ai "spec\|mtp\|herd\|prefill order\|prefix store" $D/server.err 2>/dev/null | head -12 > $D/startup-lines.txt
  kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; gzip -f $D/server.err
done
python3 $S/compare.py $RAW/herd.jsonl --groups on:on_new,on_base,on_zero,on_one off:off_new,off_base | tee $RAW/compare.txt
