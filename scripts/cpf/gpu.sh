#!/usr/bin/env bash
# Lane pf (cuda-pack-flip; scripts/cpf, scripts/pf is an older lane): GPU stages on spark1, Mac side, inside ONE fleetq slot (after scripts/cpf/build.sh all dev):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 110 --label "pf: CUDA pack gates" -- scripts/cpf/gpu.sh [stage ...]
# stages (default all, in this order): kern model ids plan kernels speed
#   kern    tests/packed_rows.rs (CUDA): registry-selected kernel per segment vs per packed total, every row byte for byte
#   model   batch_cuda_real real_35b_forced_packs_are_bitwise_solo_prefill: the fix (default) and the old rule (INTARSIA_PACK_MIN_ROWS=16)
#   ids     the batched-MTP lane's 6-prompt greedy gate (scripts/cpf/ids.py): solo spec-off reference, the old rule (repro, CB trace),
#           then the fix 8 times, each with its pack passes and Pack trace events
#   plan    plan show json, dev vs ~/intarsiaml-bm/bin/main;  kernels  kernels test --backend cuda --quick
#   speed   localeval speed ABBA at 16 streams: old rule vs fix (the 16-stream rows use the largest of SIZES, default 24,1024;
#           SIZES=24 TAG=short measures 16 streams of 24-token prompts, bodies in the 16..32 range this lane stops packing)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
HOST=spark1; HN=spark-4e77; PORT=8196
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$REPO/docs/results/2026-11-05-cuda-pack-flip/raw; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
P='~/intarsiaml-pf'; BIN=${BIN:-dev}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes $HOST "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
SPID=""
stop() { [[ -n "$SPID" ]] && { rsh "kill $SPID; for i in \$(seq 1 15); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"; SPID=""; }; }
trap stop EXIT
serve() { # serve <log-name> <env> <args...>
  local name=$1 envs=$2; shift 2
  SPID=$(rsh "nohup env $envs $P/bin/$BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name m --max-seq 8192 --mem-budget 40 --profile-every 0 $* > $P/out/$name.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && return 0; sleep 2; done
  echo "server $name did not come up"; rsh "tail -5 $P/out/$name.log"; stop; return 1
}
STAGES=${*:-kern model ids plan kernels speed}
has() { [[ " $STAGES " == *" $1 "* ]]; }
rsh "mkdir -p $P/out"

if has kern; then
  echo "== kern $(date +%T)"; mkdir -p $O/kern
  rsh "cd $P/src && INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-backend-cuda --release --test packed_rows -- --test-threads=1 --nocapture" > $O/kern/packed_rows.log 2>&1
  grep -aE '^test |test result|PACKROWS.*DIFFERS' $O/kern/packed_rows.log | head -60
fi

if has model; then
  echo "== model $(date +%T)"; mkdir -p $O/model
  for arm in fix old16; do
    env=""; [[ $arm == old16 ]] && env="INTARSIA_PACK_MIN_ROWS=16"
    rsh "cd $P/src && env $env INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_35b_forced_packs -- --ignored --test-threads=1 --nocapture" > $O/model/forced-$arm.log 2>&1
    echo "-- $arm"; grep -aE '^test |test result|FORCEDPACK (pack_min|gate prompt|.*DIFFERS|[0-9]+ packs ran)' $O/model/forced-$arm.log | head -40
  done
  # lane pb's probe: a trailing 16..40-row chunk after two 2048-row chunks, macro pass vs the plain schedule (fix, and main's rule)
  for arm in fix old16; do
    env=""; [[ $arm == old16 ]] && env="INTARSIA_PACK_MIN_ROWS=16"
    rsh "cd $P/src && env $env INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 cargo test -p intarsia-models --release --test batch_cuda_real real_35b_macro_trailing_chunk_probe -- --ignored --test-threads=1 --nocapture" > $O/model/macro-probe-$arm.log 2>&1
    echo "-- macro probe $arm"; grep -aE '^test |test result|MACROPROBE' $O/model/macro-probe-$arm.log | head -20
  done
fi

ids_run() { # ids_run <name> <env> <conc> <args...>
  local name=$1 envs=$2 conc=$3; shift 3
  echo "== ids $name $(date +%T)"
  serve ids-$name "INTARSIA_CB_TRACE=1 $envs" "$@" || return
  python3 $REPO/scripts/cpf/ids.py http://$HN:$PORT/v1 m $O/ids/ids-$name.json $conc
  curl -sf http://$HN:$PORT/metrics | grep -E '^intarsia_batch_(pack_passes|packed_chunks|prefill_chunks)' > $O/ids/ids-$name.metrics
  stop
  rsh "grep -aE 'cb-trace.*(Admitted|PrefillStart|Chunk|Pack)' $P/out/ids-$name.log | sed 's/cb-trace t=[0-9]* //'" > $O/ids/trace-$name.txt
  python3 $REPO/scripts/cpf/cmp.py $O/ids/ids-solo-off.json $O/ids/ids-$name.json $O/ids/ids-$name.metrics $O/ids/trace-$name.txt | tee -a $O/ids/ids.txt
}
if has ids; then
  mkdir -p $O/ids
  ids_run solo-off "" 1 --max-batch 1 --spec off
  for r in 1 2 3 4 5 6; do ids_run old16-r$r "INTARSIA_PACK_MIN_ROWS=16" 6 --max-batch 16; done
  for r in 1 2 3 4 5 6 7 8; do ids_run fix-r$r "" 6 --max-batch 16; done
fi

if has plan; then
  echo "== plan $(date +%T)"; mkdir -p $O/plan
  for b in "$P/bin/$BIN:dev" "~/intarsiaml-bm/bin/main:bm-main"; do
    rsh "${b%%:*} --version | head -1; ${b%%:*} plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json" > $O/plan/plan-${b##*:}.json 2> $O/plan/plan-${b##*:}.err
  done
  (cmp <(tail -n +2 $O/plan/plan-dev.json) <(tail -n +2 $O/plan/plan-bm-main.json) && echo "plan show: dev == bm main (byte-identical json)" || echo "plan show: DIFFERS") | tee $O/plan/plan.txt
fi

if has kernels; then
  echo "== kernels test --quick $(date +%T)"; mkdir -p $O/kernels
  rsh "$P/bin/$BIN kernels test --backend cuda --quick" > $O/kernels/kernels.log 2>&1; echo "kernels exit $?" | tee $O/kernels/exit.txt
  grep -cE "NOT ADMITTED|FAILED" $O/kernels/kernels.log; tail -3 $O/kernels/kernels.log
fi

le() { # le <label> <recipe>
  ( cd /opt/code/deepeval && FLEETQ_JOB=$FLEETQ_JOB uv run localeval speed --url http://$HN:$PORT/v1 --model m --thinking off --sizes ${SIZES:-24,1024} --gen 128 --reps 3 --concurrency 16 \
      --label "$1" --recipe "$2" ) > $O/speed/$1.out 2>&1
  echo "$1: $(grep -aE 'run_id|Run id|results/' $O/speed/$1.out | head -2 | tr '\n' ' ')" | tee -a $O/speed/index.txt
  grep -aE 'streams: aggregate|aggregate' $O/speed/$1.out | head -4
}
if has speed; then
  mkdir -p $O/speed
  for r in 1 2; do
    arms="old16 fix"; (( r == 2 )) && arms="fix old16"
    for a in $arms; do
      env=""; [[ $a == old16 ]] && env="INTARSIA_PACK_MIN_ROWS=16"
      echo "== speed r$r $a $(date +%T)"
      serve speed-$a${TAG:+-$TAG}-r$r "$env" --max-batch 16 || continue
      le "pf-$a${TAG:+-$TAG}-r$r" "intarsia serve --backend cuda --max-seq 8192 --mem-budget 40 --max-batch 16 bin=$BIN env='$env' (lane pf)"
      curl -sf http://$HN:$PORT/metrics | grep -E '^intarsia_batch_(pack_passes|packed_chunks|prefill_chunks)' > $O/speed/metrics-$a${TAG:+-$TAG}-r$r.txt
      stop
    done
  done
fi
echo "== done $(date +%T)"
