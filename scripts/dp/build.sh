#!/usr/bin/env bash
# Lane dp: this worktree on tatooine (CPU only, no GPU, radiance stays up). tatooine cannot reach GitHub, so the tree is rsynced.
# Run from the repo root on endor, inside a tatooine cpu slot:
#   fleetq run --host tatooine --load cpu --expect 40 --label "..." -- scripts/dp/build.sh
# Everything lives in ONE folder on tatooine, ~/intarsiaml-dp: the tree, the cargo target dir (~/intarsiaml-dp/target, bind-mounted at /target;
# seeded once from the docker volume intarsia-cargo-target-g2b-dev, read-only, so the build is incremental) and the binary
# ~/intarsiaml-dp/dp-intarsia (it needs the image's ROCm libs, so it runs inside the image as /work/dp-intarsia). The crates.io cache is the
# existing shared volume intarsia-cargo-target-registry (read by the build, not created here).
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$(cd "$S/../.." && pwd)"
SHA=$(git -C "$T" rev-parse --short=9 HEAD); BR=$(git -C "$T" rev-parse --abbrev-ref HEAD)
: "${FLEETQ_JOB:?a tatooine cpu slot}"
rsync -a --delete --exclude .git --exclude target --exclude .worktrees --exclude .scratch --exclude dp-intarsia --exclude logs "$T/" tatooine:intarsiaml-dp/
ssh -o BatchMode=yes tatooine "STAMP='head=${SHA} branch=$BR' bash -s" <<'REMOTE'
set -euo pipefail
D=$HOME/intarsiaml-dp; IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p "$D/logs"
if [ ! -d "$D/target" ]; then
  docker run --rm --user 0 -v intarsia-cargo-target-g2b-dev:/s:ro -v "$D":/w "$IMG" sh -c "mkdir -p /w/target && cp -a /s/. /w/target/ && chown -R $(id -u):$(id -g) /w/target"
fi
echo "== build $STAMP $(date +%T)"
set +e
docker run --rm --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e CARGO_TARGET_DIR=/target -e CARGO_BUILD_JOBS=20 \
  -e INTARSIA_BUILD_STAMP="$STAMP" -v "$D":/work -v "$D/target":/target -v intarsia-cargo-target-registry:/opt/cargo/registry -w /work "$IMG" \
  sh -c 'cargo build --release -p intarsia-cli --features hip && cp /target/release/intarsia /work/dp-intarsia' > "$D/logs/build.log" 2>&1
rc=$?
echo "build exit $rc $(date +%T)"; grep -E '^(error|warning: unused)|Finished' "$D/logs/build.log" | tail -5
[ $rc -eq 0 ] || exit $rc
ls -la "$D/dp-intarsia"; du -sh "$D"
REMOTE
