#!/bin/bash
# usage: cc-checks.sh <cuda|hip> <outdir> <variant: base|head|head-nofusion> [steps...]
# Runs inside a fleetq slot (FLEETQ_JOB set by caller).
MODE=$1; OUT=$2; V=$3; shift 3
STEPS=${*:-"accshort acclong gen-short gen-4k optable"}
mkdir -p $OUT
TREE=${V%%-*}
ENVX=""
[ "$V" = head-nofusion ] && ENVX="INTARSIA_NO_FUSION=1"
if [ $MODE = cuda ]; then
  RUN="env $ENVX /tmp/intarsia-cc-$TREE-target/release/intarsia"
  M=/home/nvidia/models/Qwen3-8B; G=/home/nvidia/intarsiaml-golden
else
  export INTARSIA_REPO=/tmp/intarsia-hip-$TREE INTARSIA_TARGET_VOL=ccheck-target-$TREE HIP_VISIBLE_DEVICES=1
  RUN="/tmp/intarsia-hip-head/deploy/tatooine/dev.sh env $ENVX /target/release/intarsia"
  M=/models/Qwen3-8B; G=/models/golden
fi
for S in $STEPS; do
  case $S in
    kernels) $RUN kernels test --backend $MODE --model $M > $OUT/$V-kernels.out 2> $OUT/$V-kernels.err; echo "$V kernels exit $?";;
    accshort) $RUN accuracy --model $M --backend $MODE --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $OUT/$V-acc-short.out 2> $OUT/$V-acc-short.err; echo "$V accshort exit $?";;
    acclong) $RUN accuracy --model $M --backend $MODE --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $OUT/$V-acc-long.out 2> $OUT/$V-acc-long.err; echo "$V acclong exit $?";;
    gen-short|gen-4k) P=${S#gen-}; $RUN generate --model $M --backend $MODE --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids --prompt "$(cat /tmp/prompt-$P.txt)" > $OUT/$V-gen-$P.out 2> $OUT/$V-gen-$P.err; echo "$V gen-$P exit $?";;
    optable) $RUN generate --model $M --backend $MODE --max-seq 8192 --profile-every 1 --op-table --temperature 0 --max-tokens 65 --ignore-eos --prompt "$(cat /tmp/prompt-short.txt)" > $OUT/$V-optable.out 2> $OUT/$V-optable.err; echo "$V optable exit $?";;
  esac
done
echo "ALLDONE $V" 
