#!/usr/bin/env bash
# Lane bi8 GPU job 3 (spark2, --quiet gpu,cpu): the shared-path gates. CUDA rows_exact; batch_cuda_real on the 35B (model gate
# and engine == local engine); `plan show` base vs fix byte-compared for the 8B, the 35B and Gemma-4; the 35B served-token
# identity (#224's equal.py 21 cells, 1/4/16 streams) base vs fix at --max-batch 16.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
H=spark2; HN=spark-fa00; PORT=8197
D=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:?}; mkdir -p "$OUT"
HUB=/home/nvidia/.cache/huggingface/hub
M8=$HUB/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
M35=$HUB/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
MG=/home/nvidia/models/Gemma-4-26B-A4B-NVFP4
B=/home/nvidia/intarsiaml-bi8/bin
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
SPID=""
rsh() { ssh -n -o BatchMode=yes $H "$@"; }
stop_server() { [[ -n "$SPID" ]] && { rsh "kill $SPID 2>/dev/null; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 2"; SPID=""; }; }
trap stop_server EXIT INT TERM
echo "== plan show"
rsh "mkdir -p ~/intarsiaml-bi8/out/plan; f=0; for mm in 8:$M8 35:$M35 g:$MG; do m=\${mm%%:*}; M=\${mm#*:}; for t in base intarsia; do $B/\$t plan show --backend cuda --model \$M --max-seq 8192 --mem-budget 40 --json < /dev/null > ~/intarsiaml-bi8/out/plan/\$m-\$t.json 2>/dev/null; $B/\$t plan show --backend cuda --model \$M --max-seq 8192 --mem-budget 40 < /dev/null > ~/intarsiaml-bi8/out/plan/\$m-\$t.txt 2>/dev/null; done; for k in json txt; do if [ -s ~/intarsiaml-bi8/out/plan/\$m-base.\$k ] && cmp -s ~/intarsiaml-bi8/out/plan/\$m-base.\$k ~/intarsiaml-bi8/out/plan/\$m-intarsia.\$k; then echo \"PASS plan \$m \$k byte-identical (\$(wc -c < ~/intarsiaml-bi8/out/plan/\$m-intarsia.\$k) B)\"; else echo \"FAIL plan \$m \$k\"; f=\$((f+1)); fi; done; done; echo plan fails=\$f" | tee "$OUT/plan.log"
echo "== rows_exact"
rsh "$RPATH cd ~/intarsiaml-bi8/src && INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test rows_exact 2>&1 | grep -aE '^test result|FAILED|panicked' | tail -6" | tee "$OUT/rows_exact.log"
echo "== batch_cuda_real 35B"
rsh "$RPATH cd ~/intarsiaml-bi8/src && INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_35b_batched -- --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|pairs bitwise'" | tee "$OUT/batch35.log"
echo "== 35B served identity, base vs fix, --max-batch 16"
for bin in base intarsia; do
  SPID=$(rsh "nohup $B/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name intarsia --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 > ~/intarsiaml-bi8/out/ident35-$bin.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 150); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
  [[ $up == 1 ]] || { echo "server $bin did not come up"; stop_server; continue; }
  python3 "$D/../htk/equal.py" run --url "http://$HN:$PORT/v1" --out "$OUT/eq35-$bin.json" > /dev/null
  stop_server
done
python3 "$D/../htk/equal.py" compare "$OUT/eq35-base.json" "$OUT/eq35-intarsia.json" | tee "$OUT/eq35-compare.log" | tail -1
echo "== done"
