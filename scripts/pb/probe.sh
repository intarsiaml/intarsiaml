#!/usr/bin/env bash
# Lane pb: packed-prefill bit-identity on CUDA (35B NVFP4 tp1), spark1, Mac side, inside one fleetq slot:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 80 --label "pb: packed prefill bit-identity probe + fix" -- scripts/pb/probe.sh [steps]
# steps (default "build probe serve"):
#   build: rsync this tree to ~/intarsiaml-pb/src (cargo target seeded from lane bm's), build the CLI (~/intarsiaml-pb/bin/pb) and
#          the batch_cuda_real test binary
#   probe: real_35b_packed_short_segment_probe and real_35b_macro_trailing_chunk_probe, once per env of PROBE_ENVS (';'-separated)
#   serve: scripts/pb/ids.py (lane bm's ids.py) 6 concurrent greedy prompts vs the solo spec-off ids, RUNS times per binary in
#          BINS (default: main 602a1fcfe = ~/intarsiaml-bm/bin/main, and this tree), INTARSIA_CB_TRACE=1
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8196
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$REPO/docs/results/2026-11-05-pb-packed-bitid/raw; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $HOST "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
DISABLE=moe_experts_mma_bf16,moe_experts_w4a16_bf16
PROBE_ENVS=${PROBE_ENVS:-"INTARSIA_PACK_MIN_ROWS=16;INTARSIA_PACK_MIN_ROWS=16 INTARSIA_CUDA_DISABLE=$DISABLE;"}
SPID=""
stop() { [[ -n "$SPID" ]] && { rsh "kill $SPID; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"; SPID=""; }; }
trap stop EXIT
for step in ${1:-build probe serve}; do case $step in
build)
  echo "== build $(date +%T)"
  rsh "mkdir -p ~/intarsiaml-pb/src ~/intarsiaml-pb/out ~/intarsiaml-pb/bin; test -d ~/intarsiaml-pb/src/target || cp -a ~/intarsiaml-bm/src/target ~/intarsiaml-pb/src/target"
  rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .worktrees --exclude out --exclude 'docs/results/*/raw' "$REPO/" "$HOST:intarsiaml-pb/src/"
  rsh "cd ~/intarsiaml-pb/src && CARGO_BUILD_JOBS=12 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -4 && cp target/release/intarsia ~/intarsiaml-pb/bin/pb"
  rsh "cd ~/intarsiaml-pb/src && CARGO_BUILD_JOBS=12 cargo test -p intarsia-models --release --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -4"
  ;;
probe)
  IFS=';' read -ra ENVS <<< "$PROBE_ENVS"; [[ "$PROBE_ENVS" == *";" ]] && ENVS+=("")
  n=0
  for e in "${ENVS[@]}"; do
    n=$((n+1)); echo "== probe $n [${e:-default}] $(date +%T)"
    rsh "cd ~/intarsiaml-pb/src && env $e INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real -- --ignored --exact real_35b_packed_short_segment_probe real_35b_macro_trailing_chunk_probe --test-threads=1 --nocapture 2>&1" > $O/probe-$n.log
    echo "env: ${e:-default}" >> $O/probe-$n.log
    grep -aE '^(PROBE|MACROPROBE)|^test |test result|panicked' $O/probe-$n.log | tail -80
  done
  ;;
serve)
  git -C "$REPO" show origin/lane-bm/batched-mtp:docs/results/2026-11-05-batched-mtp/raw/gates-cuda2/ids-solo-off.json > $O/ids-solo-off.json
  for bin in ${BINS:-main pb}; do
    case $bin in main) B=~/intarsiaml-bm/bin/main;; pb) B=~/intarsiaml-pb/bin/pb;; esac
    for r in $(seq 1 ${RUNS:-6}); do
      name=$bin-r$r; echo "== serve $name $(date +%T)"
      SPID=$(rsh "nohup env INTARSIA_CB_TRACE=1 ${SERVE_ENV:-} $B serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 --max-batch 16 > ~/intarsiaml-pb/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
      up=0; for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && { up=1; break; }; sleep 2; done
      [[ $up == 1 ]] && python3 $REPO/scripts/pb/ids.py http://$HN:$PORT/v1 m $O/ids-$name.json 6
      curl -sf http://$HN:$PORT/metrics | grep -E '^intarsia_(pack|prefill)' > $O/metrics-$name.txt
      stop
      rsh "cat ~/intarsiaml-pb/out/serve-$name.log" > $O/serve-$name.log
      [[ $up == 1 ]] || { echo "server $name did not come up"; tail -5 $O/serve-$name.log; continue; }
      python3 - $O/ids-solo-off.json $O/ids-$name.json <<'PY' | tee -a $O/ids.txt
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
same = [x == y for x, y in zip(a, b)]
print(sys.argv[2].split('/')[-1], "identical per sequence:", same, "ALL" if all(same) and len(a) == len(b) else "MISMATCH")
PY
      grep -aE "cb-trace.*(Admitted|Chunk)" $O/serve-$name.log | head -16
    done
  done
  ;;
esac; done
echo "== done $(date +%T)"
