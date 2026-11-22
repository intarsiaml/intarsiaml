#!/bin/sh
# F (fleetq #11145, --quiet gpu,cpu, after job R #11111): the ink/dfuse tip at 00c81dda7 on Metal, snapshot .scratch/snap5 =
# the default-on flip 2f952f697 + ink/metal dbec19036 merged (adds tests/inkling2_mix_decode_cpu.rs). Retargeted from snap4
# (6318887b5, the flip before the merge; the old script is jobF.v1.sh).
#   1. tests with the new defaults and with all four switches off: the Metal crate (inkload, inkctrl, inkctrl_nometal4,
#      inkmix), the models' Inkling tests on Metal (inkling2 = G1 tiny, inkling2_batch, inkling2_mix_decode_cpu). A failure is
#      recorded, not fatal: the model checks still run.
#   2. binary, admission of the Inkling ops with the defaults, plan show (Qwen3-8B, Llama-3.1-8B) vs the base binary
#   3. defaults == E's best: greedy md5 + dispatches per step for no env / all =1 / all =0 / each =0, 16-row self-golden
#   4. G1-G3: G2 both modes and G3 prefill with the defaults (+ confident rows), self-goldens vs E's best (byte compare)
W=/opt/code/intarsiaml/.worktrees/ink-dfuse; S=$W/.scratch
cd $S/snap5 || exit 1
WANT=00c81dda7
[ "$(git rev-parse --short=9 HEAD)" = $WANT ] || { echo "ABORT: snap5 at $(git rev-parse --short=9 HEAD), not $WANT"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "ABORT: snap5 has uncommitted changes"; git status --short; exit 1; }
[ -d target ] || cp -c -R $S/snap3/target target
# build hazard (coordinator): a cloned target's outputs can be newer than this tree's sources; touch them all so the flip
# and the merge are compiled in
git ls-files -z crates kernels | xargs -0 touch
L=$S/F; mkdir -p $L
date > $L/times.txt
# coordinator guard: a backfilled start gets SIGTERM at the declared --expect; every step prints its result to the job log
# as it finishes, and the trap names the step that was cut
STEP=setup; trap 'echo "CUT (SIGTERM) during step: $STEP"; cat $L/times.txt; exit 143' TERM
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
ALL0="INTARSIA_INK_ROUTE_FUSED=0 INTARSIA_INK_MSG_PRE=0 INTARSIA_INK_MIX_HOIST=0 INTARSIA_INK_BLEND_NORM_C=0"
# ---- 1. tests (cargo test first; the metal binary is built after every cargo test)
t() { tag=$1; shift; STEP="test $tag"; env "$@" > $L/t_$tag.log 2>&1; rc=$?
  echo "EXIT $tag $rc: $(grep -h 'test result' $L/t_$tag.log | awk '{p+=$4; f+=$6} END {print p+0 " passed, " f+0 " failed"}')"
  grep -h "panicked\|FAILED" $L/t_$tag.log | head -6; }
STEP=tests
{
t metal_defaults INTARSIA_INK_X=0 cargo test --release -p intarsia-backend-metal --test inkload --test inkctrl --test inkctrl_nometal4 --test inkmix -- --skip ink_route_stress
t metal_alloff $ALL0 cargo test --release -p intarsia-backend-metal --test inkload --test inkctrl --test inkctrl_nometal4 --test inkmix -- --skip ink_route_stress
t models_defaults INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch --test inkling2_mix_decode_cpu -- --nocapture
t models_alloff $ALL0 INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 --test inkling2_batch --test inkling2_mix_decode_cpu -- --nocapture
} 2>&1 | tee $L/tests.txt
echo "tests $(date)" >> $L/times.txt
# ---- 2. binary, admission, plan show
STEP="build + admission + plan"
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build.log; exit 1; }
for op in ink_route conv_mix sconv_res attn_res_mix loop_blend; do
  $B kernels test --backend metal --op $op > $L/kt_$op.log 2>&1; echo "EXIT kt_$op $?" >> $L/kt_$op.log
done
grep -h "ADMITTED\|REJECT\|EXIT" $L/kt_*.log
for m in Qwen3-8B Llama-3.1-8B-Instruct; do
  $S/bin/intarsia-base plan show --model ~/models/$m --backend metal > $L/plan_${m}_base.out 2> $L/plan_${m}_base.err
  $B plan show --model ~/models/$m --backend metal > $L/plan_${m}_tip.out 2> $L/plan_${m}_tip.err
  if cmp -s $L/plan_${m}_base.out $L/plan_${m}_tip.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_tip.err; then v=byte-identical; else v=DIFFERS; fi
  echo "PLAN $m, base f55bfe8b7 vs tip $WANT: $v (stdout $(wc -l < $L/plan_${m}_tip.out) lines)"
done | tee $L/plan.txt
echo "binary+admission+plan $(date)" >> $L/times.txt
# ---- 3. the defaults == E's best
STEP="env matrix + 16-row"
cfg() { case $1 in defaults) echo "INTARSIA_INK_X=0";; all1) echo "INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_MIX_HOIST=1 INTARSIA_INK_BLEND_NORM_C=1";;
  all0) echo "$ALL0";;
  route0) echo "INTARSIA_INK_ROUTE_FUSED=0";; msg0) echo "INTARSIA_INK_MSG_PRE=0";; hoist0) echo "INTARSIA_INK_MIX_HOIST=0";; normc0) echo "INTARSIA_INK_BLEND_NORM_C=0";; esac; }
for m in defaults all1 all0 route0 msg0 hoist0 normc0; do
  env $(cfg $m) INTARSIA_METAL_PROFILE=cb:$L/cb_$m.jsonl $S/tmo.sh 300 $B generate --model $CK --backend metal --max-seq 4096 --prompt "$P" --max-tokens 64 --temperature 0 --profile-every 0 > $L/r1_$m.out 2> $L/r1_$m.err
  ok=$( [ -s $L/r1_$m.out ] && grep -q "finish=" $L/r1_$m.err && echo ok || echo INVALID-RUN )
  echo "$m: $ok md5 $(md5 -q $L/r1_$m.out) bytes $(wc -c < $L/r1_$m.out) $(python3 $S/cbstat.py $L/cb_$m.jsonl fb 2>&1 | grep -o 'dispatches/step [0-9]*') $(grep -ho 'decode [0-9]* ms ([0-9.]* tok/s)' $L/r1_$m.err)" | tee -a $L/flip.txt
done
echo "E best md5 (ac2371c81, env on): $(md5 -q $S/E/r1_best.out); E off md5: $(md5 -q $S/E/r1_off.out)" | tee -a $L/flip.txt
G2S=golden/oracle/inkling2-mini-oracle-w4a16-short-top20.safetensors
$S/tmo.sh 900 $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --prefill-chunk 16 --golden $G2S --max-mean-kl 1 --no-ref-noise --allow-dirty --write-self-golden $L/r16_defaults.safetensors > $L/r16_defaults.log 2>&1
echo "16-row self-golden, defaults vs E best: $(python3 -I $S/stcmp.py $S/E/r16_best.safetensors $L/r16_defaults.safetensors)" | tee -a $L/flip.txt
echo "flip $(date)" >> $L/times.txt
# ---- 4. G1-G3 (G1 = models_defaults above: inkling2 on Metal); G2 both modes, G3 prefill, with the defaults
gsum() {  # $1 = g2-short ...: the run's verdict and its self-golden against E's best (and R's all-off when there)
  { echo "== $1"; grep -h "^== mode\|^OVERALL\|golden gate\|^EXIT" $L/$1.log | cut -c1-120; tail -n 2 $L/conf-$1.txt
    echo "self-golden vs E best ($1): $(python3 -I $S/stcmp.py $S/E/$1-self.safetensors $L/$1-self.safetensors 2>&1)"
    [ -f $S/route/.scratch/R/$1-alloff-self.safetensors ] && echo "self-golden vs R all-off ($1): $(python3 -I $S/stcmp.py $S/route/.scratch/R/$1-alloff-self.safetensors $L/$1-self.safetensors 2>&1)"; } | tee -a $L/gates.txt; }
for g in short long; do
  STEP="G2 $g"
  kl=$( [ $g = short ] && echo 4e-3 || echo 6e-3 )
  $B accuracy --model $CK --backend metal --max-seq 4096 --golden golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g2-$g-self.safetensors > $L/g2-$g.log 2>&1; echo EXIT $? >> $L/g2-$g.log
  uv run --project $W/../ink-metal/tools python $S/confident.py $L/g2-$g-self.safetensors golden/oracle/inkling2-mini-oracle-w4a16-$g-top20.safetensors > $L/conf-g2-$g.txt 2>&1
  gsum g2-$g
  STEP="G3 $g"
  kl=$( [ $g = short ] && echo 1.05e-2 || echo 1.65e-2 )
  $B accuracy --model $CK --backend metal --max-seq 4096 --mode prefill --golden /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors --max-mean-kl $kl --no-ref-noise --allow-dirty --write-self-golden $L/g3-$g-self.safetensors > $L/g3-$g.log 2>&1; echo EXIT $? >> $L/g3-$g.log
  uv run --project $W/../ink-metal/tools python $S/confident.py $L/g3-$g-self.safetensors /opt/code/intarsiaml/golden/inkling2-mini-vllm-$g-top20.safetensors > $L/conf-g3-$g.txt 2>&1
  gsum g3-$g
done
echo "done $(date)" >> $L/times.txt
cat $L/times.txt
