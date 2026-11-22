#!/bin/sh
# lane ink-d1row job Q (--quiet gpu,cpu): kernel benches (pipelined vs plain), measurement B, engine decode tok/s and
# step period (short and 5k prompt, A/B per change), per-dispatch traces, localeval speed-plain (plain vs pipelined).
cd /opt/code/intarsiaml/.worktrees/ink-d1row || exit 1
S=.scratch/Q
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
X=./target/release/examples
if [ ! -f .scratch/L/build.ok ]; then
  echo "job L build missing: building"
  cargo build --release --bin intarsia --features metal > $S/build.log 2>&1 && cargo build --release -p intarsia-backend-metal --examples >> $S/build.log 2>&1 || { echo "build FAILED"; exit 1; }
fi
git rev-parse --short HEAD > $S/head.txt
# 1. kernel benches
$X/inkmoe_bench nv 1 4 16 > $S/moe_nv_a.txt 2>&1; echo "moe nv a $?"
$X/inkmoe_bench mx 1 4 > $S/moe_mx.txt 2>&1; echo "moe mx $?"
INKMOE_PROBE=1 $X/inkmoe_bench nv 1 > $S/moe_nv_probe.txt 2>&1; echo "moe probe $?"
INKMOE_DNF_THREADS=256 $X/inkmoe_bench nv 1 > $S/moe_nv_dnf256.txt 2>&1; echo "moe dnf256 $?"
$X/inkmoe_bench nv 1 4 16 > $S/moe_nv_b.txt 2>&1; echo "moe nv b $?"
$X/ink_attn_bench > $S/attn_table.txt 2>&1; echo "attn table $?"
$X/ink_attn_bench --short > $S/attn_short.txt 2>&1; echo "attn short $?"
$X/ink_attn_bench --short --cold --layers 12 > $S/attn_short_cold.txt 2>&1; echo "attn short cold $?"
# 2. measurement B
$X/ink_mix_bench reread > $S/measB.txt 2>&1; echo "measB $?"
$X/ink_attn_bench --pf 84 --layers 12 > $S/pf84.txt 2>&1; echo "pf 84 $?"
$X/ink_attn_bench --pf 512 --layers 12 > $S/pf512.txt 2>&1; echo "pf 512 $?"
INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_FIRST_COMMIT=0 $X/ink_mix_bench --reps 20 reread > $S/measB_cb1.txt 2>&1; echo "measB cb1 $?"
# 3. engine decode: step period from the cb trace (profile-every 0), A/B per change, ABBA
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
G="$B generate --model $CK --backend metal --max-seq 8192 --temperature 0 --profile-every 0"
i=0
for cfg in 0:0 1:1 1:0 0:1 1:gu 1:dn 1:dn 1:gu 0:1 1:0 1:1 0:0; do
  a=${cfg%%:*}; m=${cfg##*:}; i=$((i+1))
  INTARSIA_INK_ATTN_PIPE=$a INTARSIA_INK_MOE_PIPE=$m INTARSIA_METAL_PROFILE=cb:$S/gen_short_${cfg}_$i.jsonl $G --prompt "$P" --max-tokens 256 > $S/gen_short_${cfg}_$i.out 2> $S/gen_short_${cfg}_$i.err
  echo "short attn=$a moe=$m: $(grep -i finish $S/gen_short_${cfg}_$i.err | tail -1)"
done
for k in 1 2; do
  i=$((i+1))
  INTARSIA_INK_ATTN_PD=8 INTARSIA_METAL_PROFILE=cb:$S/gen_short_pd8_$i.jsonl $G --prompt "$P" --max-tokens 256 > $S/gen_short_pd8_$i.out 2> $S/gen_short_pd8_$i.err
  echo "short pd8: $(grep -i finish $S/gen_short_pd8_$i.err | tail -1)"
done
for cfg in 0:0 1:1 1:1 0:0; do
  a=${cfg%%:*}; m=${cfg##*:}; i=$((i+1))
  INTARSIA_INK_ATTN_PIPE=$a INTARSIA_INK_MOE_PIPE=$m INTARSIA_METAL_PROFILE=cb:$S/gen_5k_${cfg}_$i.jsonl $G --prompt "$(cat .scratch/prompt5k.txt)" --max-tokens 128 > $S/gen_5k_${cfg}_$i.out 2> $S/gen_5k_${cfg}_$i.err
  echo "5k attn=$a moe=$m: $(grep -i finish $S/gen_5k_${cfg}_$i.err | tail -1)"
done
# 4. per-dispatch traces (one command buffer per launch, not waited on), plain vs pipelined
for v in 0 1; do
  INTARSIA_INK_ATTN_PIPE=$v INTARSIA_INK_MOE_PIPE=$v INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_FIRST_COMMIT=0 INTARSIA_METAL_PROFILE=cb:$S/cb1_$v.jsonl $G --prompt "$P" --max-tokens 65 > /dev/null 2> $S/cb1_$v.err
  echo "cb1 $v $?"
done
# 5. localeval speed-plain through intarsia serve, plain then pipelined
for v in 0 1; do
  INTARSIA_INK_ATTN_PIPE=$v INTARSIA_INK_MOE_PIPE=$v $B serve --model $CK --backend metal --host 127.0.0.1 --port 8099 --max-batch 1 --max-seq 20480 --served-model-name inkling-2-mini > $S/serve_$v.log 2>&1 &
  SP=$!
  n=0; until curl -sf http://127.0.0.1:8099/v1/models > /dev/null; do sleep 1; n=$((n+1)); kill -0 $SP 2>/dev/null || { echo "server died"; break; }; [ $n -gt 300 ] && break; done
  (cd /opt/code/deepeval && uv run localeval suite --url http://127.0.0.1:8099/v1 --model inkling-2-mini --tier adhoc --cells speed-plain --thinking default --label endor-intarsia-metal-d1row-pipe$v --tokenizer-repo thinkingmachines/Inkling-2-Mini --max-context 20480 --max-num-seqs 1 --lane plain --recipe "endor: INTARSIA_INK_ATTN_PIPE=$v INTARSIA_INK_MOE_PIPE=$v intarsia serve --backend metal --max-batch 1 --max-seq 20480 (intarsiaml ink/d1row $(cat /opt/code/intarsiaml/.worktrees/ink-d1row/$S/head.txt))") > $S/localeval_$v.log 2>&1
  echo "localeval pipe$v exit $?"; grep -E "run_id|unim|decode" $S/localeval_$v.log | tail -6
  kill $SP; wait $SP
done
echo done
