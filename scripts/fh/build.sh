#!/usr/bin/env bash
# Lane fh, phase B build: the branch on tatooine (CPU only, no GPU, radiance stays up). Run from the repo root on endor, after the branch is pushed:
#   fleetq run --host tatooine --load cpu --expect 45 --label "fh build" -- scripts/fh/build.sh lane-fh/hip-pack-default
# Puts a clone at ~/intarsiaml-g2b-fh on that commit and builds it with scripts/35b-tp2-hip/build.sh (binary ~/intarsiaml-g2b-fh/g2b-intarsia).
set -euo pipefail
BR=${1:?branch}; S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE=$(git -C "$S/.." remote get-url origin)
SHA=$(git -C "$S/.." rev-parse HEAD)
ssh -o BatchMode=yes tatooine "set -e; [ -d ~/intarsiaml-g2b-fh/.git ] || git clone -q $REMOTE ~/intarsiaml-g2b-fh; cd ~/intarsiaml-g2b-fh && git fetch -q origin $BR && git checkout -q --detach FETCH_HEAD && git rev-parse HEAD"
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=${FLEETQ_JOB:-} TREES='fh=head=${SHA:0:9}_branch=$BR' bash -s" < "$S/../35b-tp2-hip/build.sh"
# warm the test binaries window.sh's gates run, so the GPU window does not compile (CPU only; HIP_VISIBLE_DEVICES unset: nothing runs)
ssh -o BatchMode=yes tatooine "cd ~/intarsiaml-g2b-fh && export INTARSIA_REPO=\$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-fh FLEETQ_JOB=${FLEETQ_JOB:-} && deploy/tatooine/dev.sh cargo test --release -p intarsia-models --test batch_cuda_real --no-run && deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --test packed_rows --test fp8_f8 --no-run"
