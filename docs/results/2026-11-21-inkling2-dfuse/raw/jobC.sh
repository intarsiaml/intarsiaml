#!/bin/sh
# C (timing, --quiet gpu,cpu): lane switches all off (= the ink/metal graph) vs all on, same binary.
# off = the defaults (the ink/metal graph), on = the three switches set to 1.
#   1. cb traces of greedy decode (normal steps: wall, GPU busy, dispatches), short prompt, twice each, interleaved
#   2. profiled steps (--profile-every 16) for the per-op table
#   3. 5k-prompt decode tok/s (generate)
#   4. intarsia serve + localeval speed (sizes 128, 5120), off and on
cd /opt/code/intarsiaml/.worktrees/ink-dfuse
L=.scratch/C; mkdir -p $L
cargo build --release --bin intarsia --features metal > $L/build.log 2>&1 || { echo "BUILD FAILED"; exit 1; }
cargo test --release -p intarsia-backend-metal --test inkctrl -- --skip ink_route_stress --skip blend_tail_stress > $L/failfast.log 2>&1 || { echo "FAILFAST inkctrl tests failed"; tail -30 $L/failfast.log; exit 1; }
# in-stream microbench of 76 and the blend tail at the MTP-verify / batch row counts (coordinator condition 5)
cargo run --release -p intarsia-backend-metal --example ink_ctrl_bench -- --rows 1,2,3,4,16 --trace $L/bench_trace.jsonl > $L/ctrl_bench.md 2> $L/ctrl_bench.err
CK=/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a
B=./target/release/intarsia
P="Write a short story about a lighthouse keeper who finds a message in a bottle. It"
ON="INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_TAIL=1"
G="$B generate --model $CK --backend metal --max-seq 8192 --temperature 0"
run() { m=$1; shift; if [ $m = on ]; then env INTARSIA_INK_ROUTE_FUSED=1 INTARSIA_INK_MSG_PRE=1 INTARSIA_INK_BLEND_TAIL=1 "$@"; else "$@"; fi; }
date > $L/summary.txt
for rep in 1 2; do for m in off on; do
  run $m env INTARSIA_METAL_PROFILE=cb:$L/cb_${m}_$rep.jsonl .scratch/tmo.sh 300 $G --profile-every 0 --prompt "$P" --max-tokens 129 > $L/cb_${m}_$rep.out 2> $L/cb_${m}_$rep.err
  echo "== $m rep $rep: $(grep -h 'finish=' $L/cb_${m}_$rep.err)" >> $L/summary.txt
  python3 .scratch/cbstat.py $L/cb_${m}_$rep.jsonl fb >> $L/summary.txt 2>&1
done; done
for m in off on; do
  run $m env INTARSIA_METAL_PROFILE=cb:$L/prof_${m}.jsonl .scratch/tmo.sh 300 $G --profile-every 16 --prompt "$P" --max-tokens 49 > $L/prof_${m}.out 2> $L/prof_${m}.err
done
for m in off on; do
  run $m env INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$L/disp_${m}.jsonl .scratch/tmo.sh 300 $G --profile-every 0 --prompt "$P" --max-tokens 33 > $L/disp_${m}.out 2> $L/disp_${m}.err
  echo "== per-dispatch $m" >> $L/summary.txt; python3 .scratch/disp.py $L/disp_${m}.jsonl >> $L/summary.txt 2>&1
done
for m in off on; do echo "== per-op $m" >> $L/summary.txt; python3 .scratch/ctrl.py $L/prof_${m}.jsonl >> $L/summary.txt 2>&1; done
for rep in 1 2; do for m in off on; do
  run $m .scratch/tmo.sh 300 $G --profile-every 0 --prompt "$(cat .scratch/prompt4k.txt)" --max-tokens 128 > $L/g5k_${m}_$rep.out 2> $L/g5k_${m}_$rep.err
  echo "5k $m rep $rep: $(grep -h 'finish=' $L/g5k_${m}_$rep.err)" >> $L/summary.txt
done; done
for m in off on; do
  run $m $B serve --model $CK --backend metal --host 127.0.0.1 --port 8099 --max-batch 1 --max-seq 20480 --served-model-name inkling-2-mini > $L/serve_$m.log 2>&1 &
  SP=$!
  n=0; until curl -sf http://127.0.0.1:8099/v1/models > /dev/null; do sleep 1; n=$((n+1)); [ $n -gt 300 ] && break; kill -0 $SP 2>/dev/null || break; done
  if [ $m = on ]; then R="endor: env $ON intarsia serve --backend metal --max-batch 1 --max-seq 20480 (intarsiaml ink/dfuse $(git rev-parse --short HEAD), lane switches on)"; else R="endor: intarsia serve --backend metal --max-batch 1 --max-seq 20480 (intarsiaml ink/dfuse $(git rev-parse --short HEAD), defaults = switches off = the ink/metal graph)"; fi
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:8099/v1 --model inkling-2-mini --sizes 128,5120 --label endor-intarsia-metal-dfuse-$m --recipe "$R" ) > $L/localeval_$m.log 2>&1
  echo "localeval $m exit $?" >> $L/summary.txt
  kill $SP; wait $SP
done
date >> $L/summary.txt
cat $L/ctrl_bench.md $L/summary.txt
