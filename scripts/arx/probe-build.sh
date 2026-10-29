#!/usr/bin/env bash
# Lane ARX, ON tatooine inside a fleetq job (--load cpu): build lane PRB's prefix-reuse probe (crates/intarsia-models/tests/prefix_reuse_real.rs)
# of the branch tree for HIP -> ~/intarsiaml-arx/arx-bin/prefix_reuse_real-dev
set -uo pipefail
R=$HOME/intarsiaml-arx
export INTARSIA_REPO=$R INTARSIA_TARGET_VOL=intarsia-cargo-arx
cd "$R"
deploy/tatooine/dev.sh bash -c "
  exe=\$(cargo test --release --no-run -p intarsia-models --test prefix_reuse_real 2>&1 < /dev/null | tee /work/arx-test-probe.log | grep -aoE '/target/release/deps/prefix_reuse_real-[0-9a-f]+' | tail -1)
  [ -n \"\$exe\" ] || { echo BUILD-FAILED; grep -aE '^error' -A12 /work/arx-test-probe.log | head -60; exit 3; }
  cp \$exe /work/arx-bin/prefix_reuse_real-dev && echo probe binary" < /dev/null
