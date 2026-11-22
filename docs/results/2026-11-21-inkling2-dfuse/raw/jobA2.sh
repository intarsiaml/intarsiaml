#!/bin/sh
# A2: build, the whole Metal crate's tests, stress under a concurrent GPU load, admission, G1 (tiny model) off / on / no-metal4, plan show
cd /opt/code/intarsiaml/.worktrees/ink-dfuse
L=.scratch/A; mkdir -p $L
cargo build --release --bin intarsia --features metal > $L/build_a2.log 2>&1 || { echo "BUILD FAILED"; tail -30 $L/build_a2.log; exit 1; }
cargo test --release -p intarsia-backend-metal --no-fail-fast -- --skip ink_route_stress --skip blend_tail_stress > $L/t_metal_all.log 2>&1; echo "EXIT metal_all $?" >> $L/t_metal_all.log
INTARSIA_INK_BLEND_TAIL=1 cargo test --release -p intarsia-backend-metal --test inkmix > $L/t_metal_inkmix_tail.log 2>&1; echo "EXIT metal_inkmix_tail $?" >> $L/t_metal_inkmix_tail.log
for cs in 32 1024; do
  INTARSIA_INK_BLEND_CS=$cs cargo test --release -p intarsia-backend-metal --test inkctrl blend_tail_is > $L/t_tail_cs$cs.log 2>&1; echo "EXIT tail_cs$cs $?" >> $L/t_tail_cs$cs.log
done
.scratch/tgbw > $L/load.log 2>&1 &
LP=$!
INK_CTRL_STRESS=12000 .scratch/tmo.sh 900 cargo test --release -p intarsia-backend-metal --test inkctrl stress -- --nocapture --test-threads 1 > $L/t_stress.log 2>&1; echo "EXIT stress $?" >> $L/t_stress.log
wait $LP
for op in ink_route attn_res_mix moe_route; do
  ./target/release/intarsia kernels test --backend metal --op $op > $L/kt_$op.log 2>&1; echo "EXIT kt_$op $?" >> $L/kt_$op.log
done
for op in conv_mix loop_blend; do
  INTARSIA_INK_BLEND_TAIL=1 ./target/release/intarsia kernels test --backend metal --op $op > $L/kt_${op}_tail.log 2>&1; echo "EXIT kt_${op}_tail $?" >> $L/kt_${op}_tail.log
done
cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_cpu.log 2>&1; echo "EXIT tiny_cpu $?" >> $L/t_tiny_cpu.log
INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_metal.log 2>&1; echo "EXIT tiny_metal $?" >> $L/t_tiny_metal.log
ON="INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_TAIL=1"
env $ON INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_metal_on.log 2>&1; echo "EXIT tiny_metal_on $?" >> $L/t_tiny_metal_on.log
env $ON INTARSIA_METAL_NO_METAL4=1 INTARSIA_INKLING2_BACKEND=metal cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_metal_nom4.log 2>&1; echo "EXIT tiny_metal_nom4 $?" >> $L/t_tiny_metal_nom4.log
env $ON cargo test --release -p intarsia-models --test inkling2 > $L/t_tiny_cpu_on.log 2>&1; echo "EXIT tiny_cpu_on $?" >> $L/t_tiny_cpu_on.log
# plan show for other models: this branch vs the ink/metal f55bfe8b7 binary (.scratch/bin/intarsia-base)
for m in Qwen3-8B Llama-3.1-8B-Instruct; do
  for b in base branch; do
    bin=$( [ $b = base ] && echo .scratch/bin/intarsia-base || echo ./target/release/intarsia )
    $bin plan show --model ~/models/$m --backend metal > $L/plan_${m}_$b.out 2> $L/plan_${m}_$b.err
  done
  cmp -s $L/plan_${m}_base.out $L/plan_${m}_branch.out && cmp -s $L/plan_${m}_base.err $L/plan_${m}_branch.err && echo "PLAN $m: byte-identical" || echo "PLAN $m: DIFFERS"
done
grep -h "STRESS" $L/t_stress.log
grep -h "EXIT" $L/t_metal_all.log $L/t_metal_inkmix_tail.log $L/t_tail_cs*.log $L/t_stress.log $L/kt_*.log $L/t_tiny_*.log
