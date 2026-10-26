#!/usr/bin/env bash
# Lane sk: build / run the CUDA test binaries of a tree on spark2 (tree = ~/intarsiaml-sk-<tree>, built by build.sh), one fleetq slot each:
#   fleetq run --host spark2 --load cpu --mem 16 --expect 20 --label "sk: build tests" -- scripts/sk/gpu-tests.sh <tree> build
#   fleetq run --host spark2 --mem 76 --expect 30 --label "sk: ..." -- scripts/sk/gpu-tests.sh <tree> run <outname> <test-binary> <filter> [-- extra test args] [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=$1; STEP=$2; shift 2; HOST=${HOST:-spark2}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
P='$HOME/intarsiaml-sk-'$TREE
if [[ $STEP == build ]]; then
  ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd $P && { test -d target/test || cp -a target/acc target/test; }; CARGO_BUILD_JOBS=8 CARGO_TARGET_DIR=target/test cargo test --release -p intarsia-models --test batch_cuda_real --no-run 2>&1 | grep -E '^error|Finished|Executable' | tail -5"
  exit 0
fi
OUTN=$1; BIN=$2; FILTER=$3; shift 3
ENVS=(); EXTRA=(); INCL="--include-ignored"
for a in "$@"; do if [[ $a == --exclude-ignored ]]; then INCL=""; elif [[ $a == *=* && $a != --* ]]; then ENVS+=("$a"); else EXTRA+=("$a"); fi; done
RAW=/opt/code/intarsiaml/.worktrees/sk/docs/results/2026-10-25-sk-spark-prefill/raw; mkdir -p "$RAW"
ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; cd $P && B=\$(ls -t target/test/release/deps/$BIN-* | grep -v '\\.d\$' | head -1); echo \"binary \$B\"
  env INTARSIA_QWEN36_DIR=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5 INTARSIA_REQUIRE_GPU=1 ${ENVS[*]:-X=1} timeout 3000 \$B $FILTER $INCL --nocapture --test-threads=1 ${EXTRA[*]:-} 2>&1" > "$RAW/$OUTN.log"
grep -aE 'test result|MACRO |panicked|FAILED|passed|^test ' "$RAW/$OUTN.log" | tail -25
