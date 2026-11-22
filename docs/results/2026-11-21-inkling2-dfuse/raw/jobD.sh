#!/bin/sh
# D (fleetq #10964, --quiet gpu,cpu): the ONE combined slot for the load-order changes (1) INTARSIA_INK_MIX_HOIST and
# (2) INTARSIA_INK_BLEND_NORM_C (coordinator brief ~23:45) plus the per-switch attribution of the three fused switches.
#   1. build (pinned snapshot .scratch/snap2), bitwise fail-fast: tests/inkload.rs (hoist / norm_c on == off), inkctrl
#   2. rows-exact with the new switches on (tests/inkmix.rs), docs table, admission with them on, tiny model on Metal (G1)
#   3. real model bitwise: 64 greedy tokens off vs hoist / norm_c / both / best (text md5), 16-row passes (self-golden tensors)
#   4. per-switch normal-step timing: cb trace span median per config, 3 interleaved reps, 257 tokens, --profile-every 0
#   5. per-dispatch trace off vs best (per-kernel us in the stream), microbench (rank routing; ConvMix sites hoist / norm_c)
#   6. the blend tail's per-call distribution (median / p99 / max), unloaded and under a concurrent GPU load process
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
cd $S/snap2 || exit 1
[ -d target ] || cp -c -R $W/target target
L=$S/D; mkdir -p $L
date > $L/times.txt
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
echo "built $(date)" >> $L/times.txt
cargo test --release -p intarsia-backend-metal --test inkload -- --nocapture > $L/t_inkload.log 2>&1 || { echo "FAILFAST inkload"; grep -E "panicked|FAILED|test result|error" $L/t_inkload.log | head -20; exit 1; }
cargo test --release -p intarsia-backend-metal --test inkctrl --test inkctrl_nometal4 -- --skip ink_route_stress --skip blend_tail_stress > $L/failfast.log 2>&1 || { echo "FAILFAST inkctrl"; grep -E "panicked|FAILED|test result" $L/failfast.log; exit 1; }
grep -h "test result\|cases:" $L/t_inkload.log $L/failfast.log
NEW="INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1"
env $NEW cargo test --release -p intarsia-backend-metal --test inkmix > $L/t_inkmix_on.log 2>&1; echo "EXIT inkmix_on $?" >> $L/t_inkmix_on.log
cargo test --release -p intarsia-cli --test docs_env_vars > $L/t_docs.log 2>&1; echo "EXIT docs $?" >> $L/t_docs.log
for op in conv_mix sconv_res attn_res_mix loop_blend; do
  env $NEW ./target/release/intarsia kernels test --backend metal --op $op > $L/kt_$op.log 2>&1; echo "EXIT kt_$op $?" >> $L/kt_$op.log
done
env $NEW INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_metal_on.log 2>&1; echo "EXIT tiny_metal_on $?" >> $L/t_tiny_metal_on.log
echo "tests $(date)" >> $L/times.txt
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
cfg() { case $1 in off) echo "INTARSIA_INK_X=0";; route) echo "INTARSIA_INK_ROUTE_FUSED=1";; msg) echo "INTARSIA_INK_MSG_PRE=1";;
  tail) echo "INTARSIA_INK_BLEND_TAIL=1";; rm) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1";;
  hoist) echo "INTARSIA_INK_MIX_HOIST=1";; normc) echo "INTARSIA_INK_BLEND_NORM_C=1";; new) echo "$NEW";;
  best) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 $NEW";; esac; }
# 3. real model bitwise at 1 row
for m in off hoist normc new best; do
  env $(cfg $m) $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
done
# and at 16 rows: the G2-short prompts prefilled in 16-row chunks (every pass runs the 16-row paths), self-golden tensors
G2S=golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors
for m in off hoist normc new best; do
  env $(cfg $m) $S/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_$m.safetensors > $L/r16_$m.log 2>&1
done
{ r0=$(md5 -q $L/r1_off.out); for m in hoist normc new best; do a=$(md5 -q $L/r1_$m.out); echo "BITWISE 1-row decode (64 greedy tokens, text md5): $m vs off: $( [ $a = $r0 ] && echo EQUAL || echo DIFFER ) ($a vs $r0)"
  echo "BITWISE 16-row passes (G2 short, prefill chunk 16, self-golden tensor bytes): $m vs off: $(python3 -I $S/stcmp.py $L/r16_off.safetensors $L/r16_$m.safetensors)"; done; } > $L/bitwise.txt
cat $L/bitwise.txt
echo "bitwise $(date)" >> $L/times.txt
# 4. per-switch normal-step timing
for rep in 1 2 3; do for m in off route msg tail rm hoist normc new best; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cb_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cb_${m}_$rep.out 2> $L/cb_${m}_$rep.err
  echo "$m $rep $(python3 $S/cbstat.py $L/cb_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_${m}_$rep.err)" >> $L/attr.txt
done; done
python3 - $L/attr.txt > $L/attr_summary.txt <<'PY'
import sys, re, statistics as st, collections
d = collections.defaultdict(list); n = {}
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) (\d) .*?dispatches/step (\d+).*?span ms/step median (\S+)", l)
    if m: d[m.group(1)].append(float(m.group(4))); n[m.group(1)] = m.group(3)
base = st.median(d["off"])
for k in ["off", "route", "msg", "tail", "rm", "hoist", "normc", "new", "best"]:
    if k in d: print(f"ATTR {k}: span median of reps {st.median(d[k]):.3f} ms/step ({st.median(d[k]) - base:+.3f} vs off), reps {d[k]}, dispatches/step {n[k]}")
PY
cat $L/attr_summary.txt
echo "timing $(date)" >> $L/times.txt
# 5. per-dispatch trace (per-kernel us in the stream) off vs best, microbench with rank routing on / off
for m in off best; do
  env $(cfg $m) INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_${m}.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_${m}.out 2> $L/disp_${m}.err
  echo "== per-dispatch $m" >> $L/disp_summary.txt; python3 $S/disp.py $L/disp_${m}.jsonl >> $L/disp_summary.txt 2>&1
done
grep -E "^==|CONTROLLERS|conv_mix|loop_blend|attn_res|route|router" $L/disp_summary.txt
cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,4,16 --trace $L/bench_trace.jsonl > $L/bench_rank.md 2> $L/bench_rank.err
INTARSIA_INK_ROUTE_RANK=0 cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,4,16 --trace $L/bench_trace0.jsonl > $L/bench_norank.md 2> $L/bench_norank.err
grep -E "router|ink_route|conv_mix|loop_blend" $L/bench_rank.md; echo "-- rank off:"; grep -E "router|ink_route" $L/bench_norank.md
# 6. the fused blend tail's per-call time distribution (coordinator 2026-11-22: unloaded per-call time, p99 / max) and the
# same while another process keeps every GPU core busy (gpuload: 256 MB streams, 1024 x 1024-thread threadgroups per dispatch)
cargo build --release -p intarsia-backend-metal --example ink_ctrl_bench > $L/build_bench.log 2>&1
EX=./target/release/examples/ink_ctrl_bench
$S/tmo.sh 300 $EX --tail-dist 2000 --trace $L/dist_trace.jsonl > $L/tail_dist_unloaded.md 2> $L/tail_dist_unloaded.err
swiftc -O $S/gpuload.swift -o $L/gpuload > $L/gpuload_build.log 2>&1
$L/gpuload 200 > $L/gpuload.log 2>&1 &
GL=$!
sleep 2
$S/tmo.sh 300 $EX --tail-dist 2000 --trace $L/dist_trace_l.jsonl > $L/tail_dist_loaded.md 2> $L/tail_dist_loaded.err
echo "EXIT tail_dist_loaded $?" >> $L/tail_dist_loaded.md
kill $GL 2>/dev/null; wait $GL 2>/dev/null; cat $L/gpuload.log
echo "-- tail distribution, unloaded:"; cat $L/tail_dist_unloaded.md; echo "-- under gpuload:"; cat $L/tail_dist_loaded.md
echo "done $(date)" >> $L/times.txt
grep -h "EXIT\|ADMITTED\|REJECT" $L/t_inkmix_on.log $L/t_docs.log $L/kt_*.log $L/t_tiny_metal_on.log | grep -v "^ *$" | head -20
cat $L/times.txt
# ---- recovery (appended while running): `cargo test -p intarsia-cli` above rebuilt target/release/intarsia without the
# metal feature, so the model steps 2-5 failed at once ("built without the Metal backend"). Rebuild it and rerun them.
echo "recovery start $(date)" >> $L/times.txt
mkdir -p $L/first; mv $L/r1_* $L/r16_* $L/cb_* $L/attr.txt $L/attr_summary.txt $L/bitwise.txt $L/disp_* $L/kt_* $L/first/ 2>/dev/null
cargo build --release --bin intarsia --features metal > $L/build2.log 2>&1 || { echo "REBUILD FAILED"; tail -20 $L/build2.log; exit 1; }
$B --version 2>&1 | head -1
for op in conv_mix sconv_res attn_res_mix loop_blend; do
  env $NEW ./target/release/intarsia kernels test --backend metal --op $op > $L/kt_$op.log 2>&1; echo "EXIT kt_$op $?" >> $L/kt_$op.log
done
grep -h "ADMITTED\|REJECT\|EXIT" $L/kt_*.log
for m in off hoist normc new best; do
  env $(cfg $m) $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
  env $(cfg $m) $S/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_$m.safetensors > $L/r16_$m.log 2>&1
done
{ r0=$(md5 -q $L/r1_off.out); for m in hoist normc new best; do a=$(md5 -q $L/r1_$m.out); echo "BITWISE 1-row decode (64 greedy tokens, text md5): $m vs off: $( [ $a = $r0 ] && echo EQUAL || echo DIFFER ) ($a vs $r0)"
  echo "BITWISE 16-row passes (G2 short, prefill chunk 16, self-golden tensor bytes): $m vs off: $(python3 -I $S/stcmp.py $L/r16_off.safetensors $L/r16_$m.safetensors)"; done; wc -c $L/r1_off.out; } > $L/bitwise.txt
cat $L/bitwise.txt
echo "bitwise2 $(date)" >> $L/times.txt
for rep in 1 2 3; do for m in off route msg tail rm hoist normc new best; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cb_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cb_${m}_$rep.out 2> $L/cb_${m}_$rep.err
  echo "$m $rep $(python3 $S/cbstat.py $L/cb_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cb_${m}_$rep.err)" >> $L/attr.txt
done; done
python3 - $L/attr.txt > $L/attr_summary.txt <<'PY'
import sys, re, statistics as st, collections
d = collections.defaultdict(list); n = {}
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) (\d) .*?dispatches/step (\d+).*?span ms/step median (\S+)", l)
    if m: d[m.group(1)].append(float(m.group(4))); n[m.group(1)] = m.group(3)
base = st.median(d["off"])
for k in ["off", "route", "msg", "tail", "rm", "hoist", "normc", "new", "best"]:
    if k in d: print(f"ATTR {k}: span median of reps {st.median(d[k]):.3f} ms/step ({st.median(d[k]) - base:+.3f} vs off), reps {d[k]}, dispatches/step {n[k]}")
PY
cat $L/attr_summary.txt
echo "timing2 $(date)" >> $L/times.txt
for m in off best; do
  env $(cfg $m) INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_${m}.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_${m}.out 2> $L/disp_${m}.err
  echo "== per-dispatch $m" >> $L/disp_summary.txt; python3 $S/disp.py $L/disp_${m}.jsonl >> $L/disp_summary.txt 2>&1
done
grep -E "^==|CONTROLLERS|decode steps|conv_mix|loop_blend|attn_res|route|router" $L/disp_summary.txt
echo "recovery done $(date)" >> $L/times.txt
cat $L/times.txt
