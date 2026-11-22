#!/bin/sh
# E (fleetq #11003, --quiet gpu,cpu): final gates of lane ink-dfuse on the code it reports (snapshot .scratch/snap3, detached
# at the ink/dfuse commit in snap3.commit: ink/metal 0ee1bec29 merged, the blend tail removed, the row-table-aware hoist).
# Coordinator rule (2026-11-22 ~01:45): best = route + msg + hoist + norm_c goes default-on if (a) best beats every off rep
# by more than the off spread, short AND 5k, and (b) G1-G3 pass with best on and greedy md5 equals off; a switch whose
# leave-one-out costs nothing beyond the spread stays opt-in.
#   1. build, fail-fast bitwise tests (inkload, inkctrl, nometal4), rows-exact with best on (inkmix), G1 (tiny inkling2 on
#      Metal with best on), batched decode with best on (inkling2_batch on Metal: row tables + hoist), tiny no-metal4 off / best
#   2. the metal binary last, admission with best on, plan show (Qwen3-8B, Llama) vs the ink/metal binary
#   3. real model bitwise off vs best (1 row: 64 greedy tokens md5; 16 rows: self-golden tensors)
#   4. G2 / G3 with best on (+ confident rows)
#   5. timing: cb-trace normal-step span, off / best / leave-one-out, short (6 interleaved reps) and 5k (4 reps)
#   6. per-dispatch trace off vs best (controller ms per token), localeval speed off vs best (serve_lib.sh teardown)
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
cd $S/snap3 || exit 1
[ -d target ] || cp -c -R $S/snap2/target target
L=$S/E; mkdir -p $L
date > $L/times.txt
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
BEST="INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1"
cfg() { case $1 in off) echo "INTARSIA_INK_X=0";; best) echo "$BEST";;
  no_route) echo "INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1";;
  no_msg) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1";;
  no_hoist) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_NORM_C=1";;
  no_normc) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_MIX_HOIST=1";; esac; }
# ---- 1. tests (cargo test first; the metal binary is built after every cargo test)
cargo test --release -p intarsia-backend-metal --test inkload --test inkctrl --test inkctrl_nometal4 -- --skip ink_route_stress > $L/failfast.log 2>&1 || { echo "FAILFAST"; grep -E "panicked|FAILED|test result|^error" $L/failfast.log | head -20; exit 1; }
grep -h "test result\|cases:" $L/failfast.log
echo "failfast $(date)" >> $L/times.txt
env $BEST cargo test --release -p intarsia-backend-metal --test inkmix > $L/t_inkmix_best.log 2>&1; echo "EXIT inkmix_best $?" >> $L/t_inkmix_best.log
env $BEST INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_metal_best.log 2>&1; echo "EXIT tiny_metal_best (G1) $?" >> $L/t_tiny_metal_best.log
INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2_batch > $L/t_batch_metal_off.log 2>&1; echo "EXIT batch_metal_off $?" >> $L/t_batch_metal_off.log
env $BEST INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2_batch > $L/t_batch_metal_best.log 2>&1; echo "EXIT batch_metal_best $?" >> $L/t_batch_metal_best.log
INTARSIA_METAL_NO_METAL4=1 INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_nom4_off.log 2>&1; echo "EXIT tiny_nom4_off $?" >> $L/t_tiny_nom4_off.log
env $BEST INTARSIA_METAL_NO_METAL4=1 INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_nom4_best.log 2>&1; echo "EXIT tiny_nom4_best $?" >> $L/t_tiny_nom4_best.log
grep -h "EXIT\|test result" $L/t_*.log
for f in nom4_off nom4_best; do echo "$f: $(grep -ho 'NoKernel("[^:]*: *[^ ]* [^ ]*' $L/t_tiny_$f.log | sort | uniq -c | head -3)"; done
echo "tests $(date)" >> $L/times.txt
# ---- 2. binary, admission, plan show
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
for op in ink_route conv_mix sconv_res attn_res_mix loop_blend; do
  env $BEST $B kernels test --backend metal --op $op > $L/kt_$op.log 2>&1; echo "EXIT kt_$op $?" >> $L/kt_$op.log
done
grep -h "ADMITTED\|REJECT\|EXIT" $L/kt_*.log
plan_cmp() {  # $1 = base binary, $2 = tag
  r=0
  for m in Qwen3-8B Llama-3.1-8B-Instruct; do
    $1 plan show --model ~/models/$m --backend metal > $L/plan_${m}_$2.out 2> $L/plan_${m}_$2.err
    $B plan show --model ~/models/$m --backend metal > $L/plan_${m}_branch.out 2> $L/plan_${m}_branch.err
    if cmp -s $L/plan_${m}_$2.out $L/plan_${m}_branch.out && cmp -s $L/plan_${m}_$2.err $L/plan_${m}_branch.err; then echo "PLAN $m vs $2: byte-identical"; else echo "PLAN $m vs $2: DIFFERS"; r=1; fi
  done
  return $r
}
if ! plan_cmp $S/bin/intarsia-base base_f55bfe8b7; then
  # the base binary predates ink/metal's merges: compare with the ink/metal tip this branch merged
  [ -d $S/snapM ] || git -C $W worktree add --detach $S/snapM 0ee1bec29 > /dev/null 2>&1
  [ -d $S/snapM/target ] || cp -c -R target $S/snapM/target
  ( cd $S/snapM && cargo build --release --bin intarsia --features metal > $L/build_base.log 2>&1 ) && plan_cmp $S/snapM/target/release/intarsia ink_metal_0ee1bec29
fi
echo "binary+admission+plan $(date)" >> $L/times.txt
# ---- 3. real model bitwise off vs best
G2S=golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors
for m in off best; do
  env $(cfg $m) $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
  env $(cfg $m) $S/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_$m.safetensors > $L/r16_$m.log 2>&1
done
{ a=$(md5 -q $L/r1_best.out); r0=$(md5 -q $L/r1_off.out); echo "BITWISE 1-row decode (64 greedy tokens, text md5, $(wc -c < $L/r1_off.out) bytes): best vs off: $( [ $a = $r0 ] && echo EQUAL || echo DIFFER ) ($a vs $r0)"
  echo "BITWISE 16-row passes (G2 short, prefill chunk 16, self-golden tensor bytes): best vs off: $(python3 -I $S/stcmp.py $L/r16_off.safetensors $L/r16_best.safetensors)"; } > $L/bitwise.txt
cat $L/bitwise.txt
echo "bitwise $(date)" >> $L/times.txt
# ---- 4. G2 / G3 with best on
for g in short long; do
  kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
  env $BEST $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-self.safetensors > $L/g2-$g.log 2>&1; echo EXIT $? >> $L/g2-$g.log
  kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
  env $BEST $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-self.safetensors > $L/g3-$g.log 2>&1; echo EXIT $? >> $L/g3-$g.log
  uv run --project $W/../ink-metal/tools python $S/confident.py $L/g2-$g-self.safetensors golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors > $L/conf-g2-$g.txt 2>&1
  uv run --project $W/../ink-metal/tools python $S/confident.py $L/g3-$g-self.safetensors /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors > $L/conf-g3-$g.txt 2>&1
done
for g in g2-short g2-long g3-short g3-long; do echo "== $g"; grep -h "^== mode\|^OVERALL\|golden gate\|^EXIT" $L/$g.log | cut -c1-120; tail -n 2 $L/conf-$g.txt; done > $L/gates.txt
cat $L/gates.txt
echo "gates $(date)" >> $L/times.txt
# ---- 5. timing: normal-step span per config, interleaved reps
CFGS="off best no_route no_msg no_hoist no_normc"
for rep in 1 2 3 4 5 6; do for m in $CFGS; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cbs_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 257 > $L/cbs_${m}_$rep.out 2> $L/cbs_${m}_$rep.err
  echo "short $m $rep $(python3 $S/cbstat.py $L/cbs_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbs_${m}_$rep.err)" >> $L/attr.txt
done; done
echo "timing short $(date)" >> $L/times.txt
for rep in 1 2 3 4; do for m in $CFGS; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cbl_${m}_$rep.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$(cat $S/prompt4k.txt)" --max-tokens 257 > $L/cbl_${m}_$rep.out 2> $L/cbl_${m}_$rep.err
  echo "5k $m $rep $(python3 $S/cbstat.py $L/cbl_${m}_$rep.jsonl fb 2>&1 | grep -E 'span|dispatches/step' | tr -s ' ' | tr '\n' ' ') $(grep -ho 'prompt=[0-9]*' $L/cbl_${m}_$rep.err | head -1) $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/cbl_${m}_$rep.err)" >> $L/attr.txt
done; done
echo "timing 5k $(date)" >> $L/times.txt
python3 - $L/attr.txt > $L/attr_summary.txt <<'PY'
import sys, re, statistics as st, collections
d = collections.defaultdict(list); n = {}; tps = collections.defaultdict(list)
for l in open(sys.argv[1]):
    m = re.match(r"(\S+) (\S+) (\d) .*?dispatches/step (\d+).*?span ms/step median (\S+)", l)
    if m:
        d[(m.group(1), m.group(2))].append(float(m.group(5))); n[(m.group(1), m.group(2))] = m.group(4)
        t = re.search(r"\(([0-9.]+) tok/s\)", l)
        if t: tps[(m.group(1), m.group(2))].append(float(t.group(1)))
for p in ["short", "5k"]:
    off = d[(p, "off")]; best = d[(p, "best")]
    if not off or not best: continue
    spread = max(off) - min(off); so = sorted(off); spread1 = so[-2] - so[0] if len(so) > 2 else spread
    print(f"== {p}: off reps {off} spread {spread:.3f} (without the slowest off rep {spread1:.3f})")
    print(f"| config | dispatches / step | reps (ms / step) | median | vs off median | vs best median | engine decode tok/s (median) |")
    print("|---|---:|---|---:|---:|---:|---:|")
    mo, mb = st.median(off), st.median(best)
    for k in ["off", "best", "no_route", "no_msg", "no_hoist", "no_normc"]:
        v = d[(p, k)]
        if v: print(f"| {k} | {n[(p, k)]} | {', '.join(f'{x:.3f}' for x in v)} | {st.median(v):.3f} | {st.median(v) - mo:+.3f} | {st.median(v) - mb:+.3f} | {st.median(tps[(p, k)]) if tps[(p, k)] else float('nan'):.2f} |")
    gap = min(off) - max(best)
    print(f"RULE(a) {p}: min(off) - max(best) = {gap:.3f} ms vs off spread {spread:.3f} -> {'PASS' if gap > spread else 'FAIL'}; "
          f"median(best) vs min(off) - spread: {mb:.3f} vs {min(off) - spread:.3f} -> {'PASS' if mb < min(off) - spread else 'FAIL'}; "
          f"(without the slowest off rep: gap {gap:.3f} vs {spread1:.3f} -> {'PASS' if gap > spread1 else 'FAIL'})")
    for k in ["no_route", "no_msg", "no_hoist", "no_normc"]:
        v = d[(p, k)]
        if v: c = st.median(v) - mb; print(f"LOO {p} {k}: costs {c:+.3f} ms vs best -> {'NEEDED (> spread)' if c > spread else 'within the off spread'} (spread {spread:.3f}, {spread1:.3f} w/o slowest)")
PY
cat $L/attr_summary.txt
# ---- 6. per-dispatch trace off vs best, localeval
for m in off best; do
  env $(cfg $m) INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_${m}.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_${m}.out 2> $L/disp_${m}.err
  echo "== per-dispatch $m" >> $L/disp_summary.txt; python3 $S/disp.py $L/disp_${m}.jsonl >> $L/disp_summary.txt 2>&1
done
grep -E "^==|CONTROLLERS|decode steps" $L/disp_summary.txt
echo "disp $(date)" >> $L/times.txt
. $S/serve_lib.sh
HASH=$(git -C $S/snap3 rev-parse --short HEAD)
for m in off best; do
  serve_start $L/serve_$m.log $(cfg $m) || continue
  if [ $m = best ]; then R="endor: env $BEST intarsia serve --backend metal --max-batch 1 --max-seq 20480 (intarsiaml ink/dfuse $HASH: ink/metal 0ee1bec29 + lane switches route / msg / hoist / norm_c on)"; else R="endor: intarsia serve --backend metal --max-batch 1 --max-seq 20480 (intarsiaml ink/dfuse $HASH, defaults = lane switches off = the ink/metal 0ee1bec29 graph)"; fi
  ( cd /opt/code/deepeval && uv run localeval speed --url $SRV_URL --model inkling-2-mini --sizes 128,5120 --label endor-intarsia-metal-dfuse-e-$m --recipe "$R" ) > $L/localeval_$m.log 2>&1
  echo "localeval $m exit $? ($SRV_URL)" >> $L/times.txt
  serve_stop
done
lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -i intarsia | head -3
tail -25 $L/localeval_off.log; tail -25 $L/localeval_best.log
echo "done $(date)" >> $L/times.txt
cat $L/times.txt
