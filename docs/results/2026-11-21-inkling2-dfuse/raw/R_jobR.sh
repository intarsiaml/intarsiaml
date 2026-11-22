#!/bin/sh
# R (--quiet gpu,cpu, queued after job E #11003): ink_route candidate 1 (INTARSIA_INK_ROUTE_PRELOAD) on ink/dfuse-route at
# 120cfe4b9, built in its own worktree and target dir (.scratch/route). off = best (route + msg + hoist + norm_c), on = best +
# INTARSIA_INK_ROUTE_PRELOAD=1 (the preload only acts on the fused router).
#   1. the tree is ed14676a6 and clean; build; Metal tests (the preload bitwise test, inkctrl, nometal4), admission with it on
#   2. real model bitwise off vs on: 1 row (64 greedy tokens md5), 16 rows (G2-short in 16-row chunks, self-golden tensors)
#   3. us per ink_route call: per-dispatch trace in the stream (off / on) and the microbench (ink_ctrl_bench, 48 rotating sets)
#   2b. G2 default modes (prefill + teacher-forced decode): all switches off vs best vs best + preload, self-golden compared
#   4. normal-step span off vs on, interleaved: short prompt 6 reps, 5k prompt 4 reps (cb trace, --profile-every 0)
R=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/route; S=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch
cd $R || exit 1
WANT=120cfe4b9
[ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: route worktree at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: route worktree has uncommitted changes"; git status --short; exit 1; }
[ -d target ] || { [ -d $S/snap3/target ] && cp -c -R $S/snap3/target target; }
# build hazard (coordinator, from ink-hostgap): the cloned target's outputs are newer than this tree's sources, so cargo would
# keep the clone's crates fresh and never compile the preload in; touch every tracked source so all of it rebuilds
git ls-files -z crates kernels | xargs -0 touch
L=$R/.scratch/R; mkdir -p $L
date > $L/times.txt
# coordinator guard: a backfilled start gets SIGTERM at the declared --expect; every step prints its result to the job log
# as it finishes, and the trap names the step that was cut
STEP=setup; trap 'echo "CUT (SIGTERM) during step: $STEP"; cat $L/times.txt; exit 143' TERM
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
BEST="INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1"
cfg() { case $1 in off) echo "$BEST";; on) echo "$BEST INTARSIA_INK_ROUTE_PRELOAD=1";; esac; }
# ---- 1
STEP="tests + build"
cargo test --release -p intarsia-backend-metal --test inkctrl --test inkctrl_nometal4 -- --skip ink_route_stress --nocapture > $L/failfast.log 2>&1 || { echo "FAILFAST: the preload arms are skipped; off / all-off G2 still run"; grep -E "panicked|FAILED|test result|^error" $L/failfast.log | head -20; NOPRE=1; }
# a failed test keeps the slot useful (advisor): ARMS drops the preload arm, and the off / all-off G2 comparison still runs
ARMS="off on"; [ -n "${NOPRE:-}" ] && ARMS=off
grep -h "test result\|cases" $L/failfast.log
INTARSIA_INK_ROUTE_PRELOAD=1 INK_CTRL_STRESS=4000 cargo test --release -p intarsia-backend-metal --test inkctrl stress -- --nocapture --test-threads 1 > $L/t_stress_on.log 2>&1; echo "EXIT stress_on $? $(grep -h STRESS $L/t_stress_on.log)"
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
INTARSIA_INK_ROUTE_PRELOAD=1 $B kernels test --backend metal --op ink_route > $L/kt_ink_route_on.log 2>&1; echo "EXIT kt_ink_route_on $?" >> $L/kt_ink_route_on.log
grep -h "ADMITTED\|REJECT\|EXIT" $L/kt_ink_route_on.log
echo "tests $(date)" >> $L/times.txt
# ---- 2
STEP="real-model bitwise"
G2S=golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors
if [ -z "${NOPRE:-}" ]; then
for m in off on; do
  env $(cfg $m) $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
  env $(cfg $m) $S/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_$m.safetensors > $L/r16_$m.log 2>&1
done
ok_run() { [ -s $L/$1.out ] && grep -q "finish=" $L/$1.err; }  # build hazard 2: count a match only for two real runs
{ a=$(md5 -q $L/r1_on.out); r0=$(md5 -q $L/r1_off.out); if ok_run r1_on && ok_run r1_off; then v=$( [ $a = $r0 ] && echo EQUAL || echo DIFFER ); else v="INVALID (a run failed or is empty)"; fi
  echo "BITWISE 1-row decode (64 greedy tokens, text md5, $(wc -c < $L/r1_off.out) / $(wc -c < $L/r1_on.out) bytes): preload on vs off: $v ($a vs $r0)"
  echo "SERVED (preload instance in the model run's stderr): on: $(grep -c 'served by ink_route_p' $L/r1_on.err) off: $(grep -c 'served by ink_route_p' $L/r1_off.err)"
  echo "BITWISE 16-row passes (G2 short, prefill chunk 16, self-golden tensor bytes): preload on vs off: $(python3 -I $S/stcmp.py $L/r16_off.safetensors $L/r16_on.safetensors)"; } > $L/bitwise.txt
cat $L/bitwise.txt
fi
echo "bitwise $(date)" >> $L/times.txt
# ---- 2b (added after job E): G2 in its default modes (prefill + teacher-forced decode) with every switch off, with best, and
# with best + preload, on one binary. E (#11003) had G2's decode mode at 1.109e-2 (short) / 9.205e-3 (long) with best on,
# against 2.589e-3 / 2.474e-3 in #10952 (before the ink/metal 0ee1bec29 merge); best == off is shown for greedy decode and
# 16-row passes, not for the teacher-forced decode rows: this separates the merge from the switches.
for g in short long; do
  kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
  for m in alloff $ARMS; do
    if [ $m = alloff ]; then E0="INTARSIA_INK_X=0"; else E0=$(cfg $m); fi
    STEP="G2 $g $m"
    env $E0 $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-$m-self.safetensors > $L/g2-$g-$m.log 2>&1; echo EXIT $? >> $L/g2-$g-$m.log
    echo "G2 $g $m: $(grep -h 'KL threshold' $L/g2-$g-$m.log | grep -oE '^[a-z]+|KL [0-9.e+-]+ (PASS|FAIL)' | tr '\n' ' ') $(tail -1 $L/g2-$g-$m.log)"
  done
  for m in $ARMS; do echo "G2 $g self-golden $m vs alloff: $(python3 -I $S/stcmp.py $L/g2-$g-alloff-self.safetensors $L/g2-$g-$m-self.safetensors)"; done
done > $L/g2_cmp.txt
for f in $L/g2-*-*.log; do echo "== $(basename $f .log): $(grep -h 'KL threshold' $f | grep -oE '^[a-z]+|KL [0-9.e+-]+ (PASS|FAIL)' | tr '\n' ' ') $(tail -1 $f)"; done >> $L/g2_cmp.txt
cat $L/g2_cmp.txt
echo "g2 $(date)" >> $L/times.txt
# ---- 3
STEP="per-dispatch trace + microbench"
for m in $ARMS; do
  env $(cfg $m) INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_${m}.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_${m}.out 2> $L/disp_${m}.err
  echo "== per-dispatch $m" >> $L/disp_summary.txt; python3 $S/disp.py $L/disp_${m}.jsonl >> $L/disp_summary.txt 2>&1
done
grep -E "^==|CONTROLLERS|decode steps|ink_route" $L/disp_summary.txt
cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,4,16 --trace $L/bench_trace.jsonl > $L/bench.md 2> $L/bench.err
grep -E "router|ink_route" $L/bench.md
echo "disp+bench $(date)" >> $L/times.txt
# ---- 4
STEP="step timing"
[ -n "${NOPRE:-}" ] && { echo "timing skipped (preload test failed)"; cat $L/times.txt; exit 1; }
for rep in 1 2 3 4 5 6; do for m in off on; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cbs_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cbs_${m}_$rep.out 2> $L/cbs_${m}_$rep.err
  echo "short $m $rep $(python3 $S/cbstat.py $L/cbs_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbs_${m}_$rep.err)" | tee -a $L/attr.txt
done; done
for rep in 1 2 3 4; do for m in off on; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cbl_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$(cat $S/prompt4k.txt)" --max-tokens 257 > $L/cbl_${m}_$rep.out 2> $L/cbl_${m}_$rep.err
  echo "5k $m $rep $(python3 $S/cbstat.py $L/cbl_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbl_${m}_$rep.err)" | tee -a $L/attr.txt
done; done
echo "SERVED in timing runs: on $(grep -l 'served by ink_route_p' $L/cbs_on_*.err $L/cbl_on_*.err 2>/dev/null | wc -l) of 10, off $(grep -l 'served by ink_route_p' $L/cbs_off_*.err $L/cbl_off_*.err 2>/dev/null | wc -l) of 10" >> $L/attr.txt
python3 - $L/attr.txt > $L/attr_summary.txt <<'PY'
import sys, re, statistics as st, collections
d = collections.defaultdict(list); n = {}; tps = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) (\S+) (\d) .*?dispatches/step (\d+).*?span ms/step median (\S+)", l)
    if m:
        k = (m.group(1), m.group(2)); d[k].append(float(m.group(5))); n[k] = m.group(4)
        t = re.search(r"\(([0-9.]+) tok/s\)", l)
        if t: tps[k].append(float(t.group(1)))
for p in ["short", "5k"]:
    off, on = d[(p, "off")], d[(p, "on")]
    if not off or not on: continue
    spread = max(off) - min(off)
    for k, v in (("off (best)", off), ("on (best + preload)", on)):
        kk = (p, "off" if k.startswith("off") else "on")
        print(f"| {p} | {k} | {n[kk]} | {', '.join(f'{x:.3f}' for x in v)} | {st.median(v):.3f} | {st.median(tps[kk]):.2f} |")
    print(f"DELTA {p}: median on - off = {st.median(on) - st.median(off):+.3f} ms/step; off spread {spread:.3f}; min(off) - max(on) = {min(off) - max(on):+.3f}")
PY
cat $L/attr_summary.txt
echo "done $(date)" >> $L/times.txt
cat $L/times.txt
