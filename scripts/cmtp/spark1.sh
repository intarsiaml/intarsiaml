#!/usr/bin/env bash
# Lane cmtp (35B --spec mtp default on CUDA): build and gates ON spark1, scratch ~/intarsiaml-cmtp only.
#   src/ = this worktree (rsync), base/ = origin/main (git archive), targets target-{dev,base}, out/ = raw logs.
#   fleetq run --host spark1 --load cpu --expect 45 --label "cmtp: build" -- ssh spark1 'bash -s build' < scripts/cmtp/spark1.sh
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cmtp: ids" -- ssh spark1 'bash -s ids' < scripts/cmtp/spark1.sh
set -uo pipefail
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
R=$HOME/intarsiaml-cmtp
H=/home/nvidia/.cache/huggingface/hub
M35=$(ls -d $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
M8=$(ls -d $H/models--Qwen--Qwen3-8B/snapshots/* | head -1)
O=${O:-$R/out}; mkdir -p "$O"
BIN=${BIN:-$R/target-dev/release/intarsia}
case "${1:?build|ids}" in
build)
  for t in ${TREES:-dev base}; do
    src=$R/src; [ $t = base ] && src=$R/base
    (cd $src && CARGO_TARGET_DIR=$R/target-$t CARGO_BUILD_JOBS=${JOBS:-12} cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
    echo "build $t done $(date +%T) $(md5sum $R/target-$t/release/intarsia | cut -c1-12)"
  done ;;
ids)
  # ids <name>...: for each prompt file name (short-0 .. 16k-5) greedy 256 ids, --spec off vs --spec mtp --spec-k ${K:-2}; ARMS overrides the arm list
  shift; P=$R/src/scripts/dfh/ids-prompts; fails=0
  for n in "$@"; do
    case $n in short*) MS=4096 ;; 4k*) MS=8192 ;; *) MS=20480 ;; esac
    for arm in ${ARMS:-off mtp}; do
      case $arm in off) A="--spec off" ;; mtp) A="--spec mtp --spec-k ${K:-2}" ;; auto) A="" ;; esac
      $BIN generate -m $M35 --backend cuda --mem-budget 40 --max-seq $MS $A --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat $P/$n.txt)" --max-tokens ${NTOK:-256} < /dev/null > $O/gen-$n-$arm.out 2> $O/gen-$n-$arm.err
      grep -a '^ids:' $O/gen-$n-$arm.err > $O/gen-$n-$arm.ids
    done
    pt=$(grep -a -o 'prompt[ _a-z]*[=:] *[0-9]*' $O/gen-$n-off.err | head -1)
    st=$(grep -a '^spec:' $O/gen-$n-mtp.err | tail -1 | cut -c1-120)
    if cmp -s $O/gen-$n-off.ids $O/gen-$n-mtp.ids && [ -s $O/gen-$n-off.ids ]; then echo "PASS ids $n ($pt) $(tr ',' '\n' < $O/gen-$n-off.ids | wc -l) ids identical | $st"; else echo "FAIL ids $n ($pt) differ or empty | $st"; fails=$((fails+1)); fi
  done
  echo "ids fails=$fails" ;;
esac
