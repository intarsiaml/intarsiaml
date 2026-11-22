#!/bin/sh
# D (timing, --quiet gpu,cpu): per-switch attribution in the real decode step (one command buffer per step), the rank routing,
# and the microbench with the rank routing on / off. Fail fast on the bitwise tests first.
cd /opt/code/intarsiaml/.worktrees/ink-dfuse
L=.scratch/D; mkdir -p $L
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; exit 1; }
cargo test --release -p intarsia-backend-metal --test inkctrl --test inkctrl_nometal4 -- --skip ink_route_stress --skip blend_tail_stress > $L/failfast.log 2>&1 || { echo "FAILFAST"; grep -E "panicked|FAILED|test result" $L/failfast.log; exit 1; }
grep "test result" $L/failfast.log
cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,4,16 --trace $L/bench_trace.jsonl > $L/bench_rank.md 2> $L/bench_rank.err
INTARSIA_INK_ROUTE_RANK=0 cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,4,16 --trace $L/bench_trace0.jsonl > $L/bench_norank.md 2> $L/bench_norank.err
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
cfg() { case $1 in off) echo "INTARSIA_INK_X=0";; route) echo "INTARSIA_INK_ROUTE_FUSED=1";; route0) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_ROUTE_RANK=0";;
  msg) echo "INTARSIA_INK_MSG_PRE=1";; tail) echo "INTARSIA_INK_BLEND_TAIL=1";; rm) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1";;
  all) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_TAIL=1";; esac; }
for rep in 1 2 3; do for m in off route route0 msg tail rm all; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cb_${m}_$rep.jsonl .scratch/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cb_${m}_$rep.out 2> $L/cb_${m}_$rep.err
  echo "$m $rep $(python3 .scratch/cbstat.py $L/cb_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_${m}_$rep.err)" >> $L/attr.txt
done; done
python3 - $L/attr.txt > $L/attr_summary.txt <<'PY'
import sys, re, statistics as st, collections
d = collections.defaultdict(list); n = {}
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) (\d) .*?dispatches/step (\d+).*?span ms/step median (\S+)", l)
    if m: d[m.group(1)].append(float(m.group(4))); n[m.group(1)] = m.group(3)
base = st.median(d["off"])
for k in ["off", "route", "route0", "msg", "tail", "rm", "all"]:
    if k in d: print(f"ATTR {k}: span median of reps {st.median(d[k]):.3f} ms/step ({st.median(d[k]) - base:+.3f} vs off), reps {d[k]}, dispatches/step {n[k]}")
PY
cat $L/attr_summary.txt; cat $L/bench_rank.md | grep -E "router|ink_route"; echo "-- rank off:"; grep -E "router|ink_route" $L/bench_norank.md
