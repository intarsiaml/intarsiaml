#!/usr/bin/env bash
# Lane fh, phase B build: this worktree on tatooine (CPU only, no GPU, radiance stays up). tatooine cannot reach GitHub, so the tree is rsynced.
# Run from the repo root on endor:
#   FLEETQ_JOB=<cpu slot> scripts/fh/build.sh            (e.g. fleetq acquire --host tatooine --load cpu,disk --expect 50 ...)
# Puts the tree at ~/intarsiaml-g2b-fh, starts its cargo target volume as a copy of intarsia-cargo-target-g2b-dev (incremental build),
# builds it with scripts/35b-tp2-hip/build.sh (binary ~/intarsiaml-g2b-fh/g2b-intarsia) and compiles the test binaries window.sh's gates run.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$(cd "$S/../.." && pwd)"
SHA=$(git -C "$T" rev-parse --short=9 HEAD); BR=$(git -C "$T" rev-parse --abbrev-ref HEAD)
export FLEETQ_JOB=${FLEETQ_JOB:?a tatooine cpu slot}
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .scratch --exclude 'g2b-intarsia' "$T/" tatooine:intarsiaml-g2b-fh/
ssh -n -o BatchMode=yes tatooine "docker volume inspect intarsia-cargo-target-g2b-fh >/dev/null 2>&1 || { docker volume create intarsia-cargo-target-g2b-fh >/dev/null && docker run --rm -v intarsia-cargo-target-g2b-dev:/s:ro -v intarsia-cargo-target-g2b-fh:/d alpine sh -c 'cp -a /s/. /d/ && chmod 0777 /d'; }"
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB TREES='fh=head=${SHA}_branch=$BR' bash -s" < "$S/../35b-tp2-hip/build.sh"
# warm the test binaries window.sh's gates run, so the GPU window does not compile (nothing runs: --no-run)
ssh -n -o BatchMode=yes tatooine "cd ~/intarsiaml-g2b-fh && export INTARSIA_REPO=\$PWD INTARSIA_TARGET_VOL=intarsia-cargo-target-g2b-fh FLEETQ_JOB=$FLEETQ_JOB && deploy/tatooine/dev.sh cargo test --release -p intarsia-models --test batch_cuda_real --no-run 2>&1 | tail -3 && deploy/tatooine/dev.sh cargo test --release -p intarsia-backend-hip --test packed_rows --test fp8_f8 --no-run 2>&1 | tail -3"
