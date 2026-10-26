#!/usr/bin/env bash
# Lane dfh G0 on CUDA (spark1, after scripts/dc/gates.sh g0): flags off, the lane head against its base (origin/main merge base):
# plan show (JSON, text) + 256 greedy ids, Qwen3-8B and Qwen3.6-35B-A3B NVFP4, short and ~3.5k prompts. Run ON spark1:
#   build:  fleetq run --host spark1 --load cpu --expect 30 -- ssh spark1 'bash -s build' < scripts/dfh/cuda-g0.sh
#   g0:     fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 30 -- ssh spark1 'bash -s g0' < scripts/dfh/cuda-g0.sh
# Trees ~/intarsiaml-dfh-{dev,base} (HOST=spark1 scripts/dfh/sync.sh), targets ~/intarsiaml-dfh-target-{dev,base}.
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
H=/home/nvidia/.cache/huggingface/hub
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
O=$HOME/intarsiaml-dfh-out/g0; mkdir -p $O
case "${1:?build|g0}" in
build)
  for t in base dev; do
    (cd $HOME/intarsiaml-dfh-$t && CARGO_TARGET_DIR=$HOME/intarsiaml-dfh-target-$t CARGO_BUILD_JOBS=12 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
    echo "build $t done $(date +%T)"
  done ;;
g0)
  PD=$HOME/intarsiaml-dfh-dev/scripts/tp2-hip/noop/prompts; fails=0
  for t in base dev; do
    B=$HOME/intarsiaml-dfh-target-$t/release/intarsia
    for m in 8 35; do
      if [ $m = 8 ]; then M=$M8; else M=$M35; fi
      for p in short 4k; do
        $B generate -m $M --backend cuda --mem-budget 40 --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
          -p "$(cat $PD/$p.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $O/gen-$m-$p-$t.err
        grep -a '^ids:' $O/gen-$m-$p-$t.err > $O/gen-$m-$p-$t.ids
      done
      $B plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 --json < /dev/null > $O/plan-$m-$t.json 2> /dev/null
      $B plan show --backend cuda --model $M --max-seq 8192 --mem-budget 40 < /dev/null > $O/plan-$m-$t.txt 2> /dev/null
    done
  done
  for m in 8 35; do
    for p in short 4k; do cmp -s $O/gen-$m-$p-base.ids $O/gen-$m-$p-dev.ids && [ -s $O/gen-$m-$p-dev.ids ] && echo "PASS ids $m $p identical ($(tr ',' '\n' < $O/gen-$m-$p-dev.ids | wc -l) ids)" || { echo "FAIL ids $m $p"; fails=$((fails+1)); }; done
    for k in json txt; do
      if diff <(grep -v 'stamp\|head=\|code-tree' $O/plan-$m-base.$k) <(grep -v 'stamp\|head=\|code-tree' $O/plan-$m-dev.$k) > $O/plan-$m.$k.diff && [ -s $O/plan-$m-dev.$k ]; then
        echo "PASS plan $m $k identical but the build stamp ($(wc -c < $O/plan-$m-dev.$k) B)"; else echo "FAIL plan $m $k ($(wc -l < $O/plan-$m.$k.diff) diff lines)"; fails=$((fails+1)); fi
    done
  done
  echo "g0 fails=$fails" ;;
esac
