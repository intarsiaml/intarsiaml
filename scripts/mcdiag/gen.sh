#!/usr/bin/env bash
# mcdiag on spark1: the one-chain repro in `generate` (35B NVFP4, CUDA, tp 1, greedy, --spec mtp --spec-k 2, INTARSIA_SPEC_TRACE=1).
# Arms "name:ENV=val[+ENV2=val]" (X=1: none); ROUNDS (default 2, the arm order is reversed on even rounds); CELL (code-gen); NTOK (800);
# NSYS=1 wraps the run in `nsys profile --cuda-graph-trace=node` (one .nsys-rep per arm and round, converted to sqlite in $O). SMI=1 samples
# the GPU clock / power every 200 ms beside the run. Output: $O/<arm>-r<round>.{out,err,ids,smi}.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 25 --label "mcdiag: one-chain repro" -- ssh spark1 'W=gen1 ARMS="base:X=1 probe:INTARSIA_SPEC_COPY=256+INTARSIA_SPEC_COPY_PROBE=150" bash -s' < scripts/mcdiag/gen.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-mcdiag; W=${W:?W}; O=$R/out/$W; mkdir -p "$O"
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
BIN=${BIN:-$R/bin/intarsia}; CELL=${CELL:-code-gen}; NTOK=${NTOK:-800}; ROUNDS=${ROUNDS:-2}; ARMS=${ARMS:?ARMS}
python3 $R/src/scripts/mcopy/prompts.py $R/cells > /dev/null
P=$R/cells/$CELL.txt
for r in $(seq 1 $ROUNDS); do
  order="$ARMS"; [ $((r % 2)) = 0 ] && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for spec in $order; do
    a=${spec%%:*}; e=${spec#*:}; e=${e//+/ }
    f=$O/$a-r$r; extra=""
    [ -n "${SMI:-}" ] && { nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms 200 > $f.smi 2>&1 & SP=$!; }
    pre=""; [ -n "${NSYS:-}" ] && pre="nsys profile --cuda-graph-trace=node -t cuda --stats=false -f true -o $f"
    env $e INTARSIA_SPEC_TRACE=1 $pre $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq 4096 --spec mtp --spec-k 2 --chat --no-think --temperature 0 \
      --profile-every 0 --show-ids -p "$(cat $P)" --max-tokens $NTOK < /dev/null > $f.out 2> $f.err
    [ -n "${SMI:-}" ] && kill $SP 2>/dev/null
    grep -a '^ids:' $f.err > $f.ids
    [ -n "${NSYS:-}" ] && [ -f $f.nsys-rep ] && nsys export --type sqlite --force-overwrite true -o $f.sqlite $f.nsys-rep > /dev/null 2>&1
    echo "$a r$r: $(tr ',' '\n' < $f.ids | wc -l) ids sha $(sha256sum < $f.ids | cut -c1-12) graph_steps=$(grep -ac '^spec: mtp graph step' $f.err) chains=$(grep -ac '^spec: copy token' $f.err) $(grep -a '^spec: mtp graph step' $f.err | awk '{s+=$5;n++} END{if(n) printf "mean_step=%.3f", s/n}')"
    tail -3 $f.err | cut -c1-200
  done
done
echo "gen done"
