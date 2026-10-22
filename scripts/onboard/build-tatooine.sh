#!/usr/bin/env bash
# Lane `on`: build the HIP CLI of THIS worktree on tatooine in the dev image (CPU only, non-exclusive slot).
#   fleetq run --host tatooine --mem 0 --load cpu --label "on: HIP build" -- scripts/onboard/build-tatooine.sh
# Tree: ~/intarsiaml-on (own cargo target volume); the binary is copied to ~/intarsiaml-on/on-intarsia (runs inside the image).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP=${INTARSIA_STAMP:-$("$REPO/scripts/tree-stamp.sh")}
echo "stamp: $STAMP"
ssh -n -o BatchMode=yes tatooine "mkdir -p ~/intarsiaml-on"
rsync -a --delete --exclude target --exclude .git --exclude 'golden/*.safetensors' --exclude tools/.venv --exclude .claude --exclude on-intarsia --exclude onboard-runs "$REPO/" tatooine:intarsiaml-on/
# the canonical cached goldens (their reference-noise sidecars in golden/ref_noise bind to these file hashes; the older copies
# in /models/golden on the box have other hashes, which makes `acc-short` fail with "ref_noise was measured on golden hash ...")
rsync -a /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors /opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors tatooine:intarsiaml-on/golden/
ssh -n -o BatchMode=yes tatooine "export FLEETQ_JOB=${FLEETQ_JOB:-}; export INTARSIA_REPO=\$HOME/intarsiaml-on INTARSIA_TARGET_VOL=intarsia-cargo-target-on; D=\$INTARSIA_REPO/deploy/tatooine/dev.sh; \$D env INTARSIA_BUILD_STAMP='$STAMP' CARGO_BUILD_JOBS=\${JOBS:-20} cargo build --release -p intarsia-cli --features hip 2>&1 </dev/null | tail -5; \$D cp /target/release/intarsia /work/on-intarsia </dev/null; if [ -n '${TEST:-}' ]; then \$D env CARGO_TARGET_DIR=/tmp/t CARGO_BUILD_JOBS=\${JOBS:-20} cargo test --release -p intarsia-cli --lib 2>&1 </dev/null | grep -E '^test result|FAILED|panicked|^error' | tail -8; fi; ls -la \$INTARSIA_REPO/on-intarsia"
