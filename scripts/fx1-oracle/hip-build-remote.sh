#!/usr/bin/env bash
# ON tatooine (CPU slot, no GPU): build lane-fx1 for HIP in the dev image, own cargo target volume seeded from m3's,
# then the 1 % / 0.5 % row-parallel scale mutant overlays of the 35B (symlinks resolve inside the container).
set -euo pipefail
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
if ! docker volume inspect intarsia-cargo-target-fx1 >/dev/null 2>&1; then
  docker volume create intarsia-cargo-target-fx1 >/dev/null
  docker run --rm -v intarsia-cargo-target-m3:/a:ro -v intarsia-cargo-target-fx1:/b --user 0 $IMG bash -c 'cp -a /a/. /b/ && chmod 0777 /b' < /dev/null
fi
cd ~/intarsiaml-fx1 && mkdir -p bench-out fx1out mutants
export INTARSIA_REPO=$HOME/intarsiaml-fx1 INTARSIA_TARGET_VOL=intarsia-cargo-target-fx1
deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip < /dev/null 2>&1 | grep -E '^error|Finished' | head
deploy/tatooine/dev.sh /target/release/intarsia --version < /dev/null
for f in 1.01 1.005; do
  LINK_ROOT=/models/Qwen3.6-35B-A3B-NVFP4 python3 scripts/fx1-oracle/make_scale_mutant.py /data/models/intarsia/Qwen3.6-35B-A3B-NVFP4 mutants/mut$f $f < /dev/null
done
ls /data/models/intarsia/golden/ | grep -i 35b || true
