#!/bin/bash
# runs locally inside the fleetq slot on spark1 (FLEETQ_JOB set); SPARK_DIR=~/intarsia-hrr
set -uo pipefail
cd /opt/code/intarsiaml/.worktrees/hrr
export SPARK_DIR=~/intarsia-hrr BASE_REF=origin/main GREEDY=1
H=spark1; D='$HOME/intarsia-hrr'
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4 FLASHINFER_NVCC_THREADS=1;'
rsh() { ssh -n -o BatchMode=yes $H "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
STAMP="$(git rev-parse --short=9 HEAD) dirty=$(git diff --quiet HEAD -- crates kernels && echo no || echo yes) $(date +%m%d-%H%M)"
rsh "mkdir -p $D/src $D/bin $D/logs $D/probe"
rsync -a --no-t --checksum --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude .scratch --exclude .worktrees ./ $H:intarsia-hrr/src/
echo "stamp: $STAMP"
rsh "set -o pipefail; cd $D/src && INTARSIA_BUILD_STAMP='$STAMP' cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -3" || { echo BUILD FAILED; exit 1; }
rsh "cp $D/src/target/release/intarsia $D/bin/intarsia && echo '$STAMP' > $D/bin/STAMP"
scripts/dsv41-tp2/plandiff-spark1.sh 2>&1 | tee .scratch/plandiff-run.log
scp -q .scratch/qa/cuda_repro.sh .scratch/hrr_reuse_probe.py docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt $H:intarsia-hrr/probe/
rsh "cd $D/probe && mkdir -p docs/results/raw/2026-10-15-metal-longctx-profile/prompts && mv prompt-16384.txt docs/results/raw/2026-10-15-metal-longctx-profile/prompts/"
rsh "$D/probe/cuda_repro.sh base $D/src-base/target/release/intarsia INTARSIA_X=1"
rsh "$D/probe/cuda_repro.sh branch-default $D/bin/intarsia INTARSIA_X=1"
rsh "$D/probe/cuda_repro.sh branch-exact $D/bin/intarsia INTARSIA_PREFIX_REUSE=exact"
mkdir -p .scratch/cuda-out; scp -q "$H:intarsia-hrr/probe/probe-*.txt" "$H:intarsia-hrr/probe/probe-*.json" .scratch/cuda-out/
echo CUDA_NOOP_DONE
