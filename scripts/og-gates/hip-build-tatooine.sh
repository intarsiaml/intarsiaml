#!/usr/bin/env bash
# CPU-only slot on tatooine: sync this tree to ~/og-hip, seed an own cargo target volume (intarsia-cargo-target-og, a copy of
# the newest round's volume so only changed crates rebuild) and build the HIP release CLI in the dev image. No GPU, radiance stays up.
#   fleetq run --host tatooine --mem 0 --load cpu --expect 45 --label "og: ..." -- scripts/og-gates/hip-build-tatooine.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host tatooine}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST=tatooine; VOL=intarsia-cargo-target-og; SEED=${SEED:-intarsia-cargo-target-r10}
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
rsh() { ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh "mkdir -p ~/og-hip/og-data ~/og-hip/og-out"
rsync -a --delete --exclude target --exclude .git --exclude golden --exclude tools/.venv --exclude .claude --exclude docs/results \
  --exclude og-data --exclude og-out "$REPO/" "$HOST:og-hip/"
scp -q /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors "$HOST:og-hip/og-data/"
echo "stamp: $("$REPO/scripts/tree-stamp.sh" 2>/dev/null | head -1)  tree HEAD $(git -C "$REPO" rev-parse --short HEAD)"
rsh "docker volume inspect $VOL >/dev/null 2>&1 && echo 'volume $VOL exists' || { docker volume inspect $SEED >/dev/null 2>&1 && docker volume create $VOL >/dev/null && docker run --rm --user 0 -v $SEED:/from:ro -v $VOL:/to $IMG bash -c 'cp -a /from/. /to/ && chmod 0777 /to' && echo 'seeded $VOL from $SEED'; }"
rsh "cd ~/og-hip && INTARSIA_REPO=\$HOME/og-hip INTARSIA_TARGET_VOL=$VOL deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip 2>&1 | grep -E '^(error|warning: unused)|Finished|-->' | head -30"
rsh "cd ~/og-hip && INTARSIA_REPO=\$HOME/og-hip INTARSIA_TARGET_VOL=$VOL deploy/tatooine/dev.sh bash -c 'ls -la /target/release/intarsia && /target/release/intarsia --version'"
rsh "df -h / | tail -1; docker system df -v 2>/dev/null | grep -E 'intarsia-cargo-target-(og|r10) ' | head"
