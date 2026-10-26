#!/usr/bin/env bash
# Lane dc: build and run on spark1 (sources rsynced from this worktree; all state under ~/intarsiaml-dc-*).
#   scripts/dc/spark1.sh build   <test|cli>...     rsync + cargo build --release (run under: fleetq run --host spark1 --load cpu ...)
#   scripts/dc/spark1.sh probe                     G1 probe: tests/verify16_cuda_real.rs on the real 8B (fleetq run --quiet gpu,cpu --mem 20)
# Everything runs on spark1 through ssh; this script is started on the machine that holds the fleetq slot.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
SRC='~/intarsiaml-dc-src'; TGT='~/intarsiaml-dc-target'
ENVP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export CARGO_TARGET_DIR=$HOME/intarsiaml-dc-target'
H=/home/nvidia/.cache/huggingface/hub
M8='$(ls -d '$H'/models--Qwen--Qwen3-8B/snapshots/* | head -1)'
case "${1:?}" in
  build)
    rsync -a --delete --exclude target --exclude .git --exclude .worktrees --exclude tools/.venv "$HERE/" "spark1:intarsiaml-dc-src/" || exit 1
    shift
    for what in "$@"; do
      case $what in
        fx) ssh spark1 "$ENVP; cd $SRC && CARGO_BUILD_JOBS=8 cargo test --release --no-run -p intarsia-models --test dflash_fixture 2>&1 | grep -E '^error|Finished|Executable' -A6 | tail -12" ;;
        test) ssh spark1 "$ENVP; cd $SRC && CARGO_BUILD_JOBS=8 cargo test --release --no-run -p intarsia-models --test verify16_cuda_real 2>&1 | grep -E '^error|Finished|Executable' | tail -5" ;;
        cli)  ssh spark1 "$ENVP; cd $SRC && CARGO_BUILD_JOBS=8 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -5" ;;
      esac
    done ;;
  fxrun)  # rsync the reference fixtures (dc-ref/out/raw_*), run tests/dflash_fixture.rs on CUDA
    rsync -a /opt/code/intarsiaml/.worktrees/dc-ref/out/raw_math2 /opt/code/intarsiaml/.worktrees/dc-ref/out/raw_code1 /opt/code/intarsiaml/.worktrees/dc-ref/out/raw_chat2 spark1:intarsiaml-dc-fx/ || exit 1
    D=$H/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/4080a604bb608e59fb5240db1fa347357a4e3332
    ssh spark1 "$ENVP; cd $SRC && INTARSIA_QWEN3_DIR=$M8 INTARSIA_DFLASH_DIR=$D INTARSIA_DFLASH_FIXTURES=\$HOME/intarsiaml-dc-fx INTARSIA_DFLASH_TEST_BACKEND=cuda ${PROBE_ENV:-} cargo test --release -p intarsia-models --test dflash_fixture -- --nocapture --test-threads=1 2>&1 | grep -aE 'math2|code1|chat2|test result|panicked|error|dflash:|test '" ;;
  probe)
    ssh spark1 "$ENVP; cd $SRC && INTARSIA_QWEN3_DIR=$M8 INTARSIA_REQUIRE_GPU=1 ${PROBE_ENV:-} cargo test --release -p intarsia-models --test verify16_cuda_real -- --nocapture --test-threads=1 2>&1 | grep -aE 'PROBE|DIFF|checked|plan notes|test result|panicked|error'" ;;
esac
# fetch   : download the DCOCO checkpoint into the HF hub layout on spark1 (run under: fleetq run --host spark1 --load net,disk)
if [ "${1:-}" = fetch ]; then
  ssh spark1 'set -e; R=TD-Labs/Qwen3-8B-DCOCO; SHA=4080a604bb608e59fb5240db1fa347357a4e3332
    D=/home/nvidia/.cache/huggingface/hub/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/$SHA; mkdir -p $D
    for f in config.json README.md model.safetensors; do curl -sfL -C - -o $D/$f https://huggingface.co/$R/resolve/$SHA/$f; done
    ls -l $D; echo "want 4591038706 bytes"; sha256sum $D/model.safetensors
    curl -sIL https://huggingface.co/$R/resolve/$SHA/model.safetensors | grep -i "x-linked-etag"'
fi
# basesync: export origin/main (this repo) to spark1:~/intarsiaml-dc-base
if [ "${1:-}" = basesync ]; then
  git -C "$HERE" archive origin/main | ssh spark1 'rm -rf ~/intarsiaml-dc-base && mkdir -p ~/intarsiaml-dc-base && tar -x -C ~/intarsiaml-dc-base' && echo synced "$(git -C "$HERE" rev-parse --short origin/main)"
fi
# gates: run scripts/dc/gates.sh <args> on spark1 from the rsynced tree
if [ "${1:-}" = gates ]; then shift; ssh spark1 "GATE='${GATE:-}' NTOK='${NTOK:-}' MODE='${MODE:-}' K='${K:-}' ZERO='${ZERO:-}' bash ~/intarsiaml-dc-src/scripts/dc/gates.sh $*"; fi
