#!/usr/bin/env bash
# Lane PF27, ON tatooine inside a fleetq job (--load cpu): build the release CLI (and the mxfp4_a8 test binary) of ~/intarsiaml-pf27
# in the lane's cargo volume intarsia-cargo-pf27 (seeded once from intarsia-cargo-glr, as scripts/asa/build.sh seeds from ldf).
#   bash ~/intarsiaml-pf27/scripts/pf27/build.sh <tag>      -> ~/intarsiaml-pf27/pf27-<tag>-cli, ~/intarsiaml-pf27/pf27-bin/mxfp4_a8-<tag>
# CLI=0 skips the CLI, TESTS=0 skips the test binary.
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-pf27 INTARSIA_TARGET_VOL=intarsia-cargo-pf27
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
tag=${1:?tag}
cd "$INTARSIA_REPO"; mkdir -p pf27-bin
if ! docker volume inspect "$INTARSIA_TARGET_VOL" > /dev/null 2>&1; then
  docker volume create "$INTARSIA_TARGET_VOL" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-glr:/a:ro -v "$INTARSIA_TARGET_VOL":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $INTARSIA_TARGET_VOL"
fi
# rsync keeps the Mac mtimes: a tree switch (base <-> dev) can look older than the last build to cargo. Force the kernel library to rebuild.
touch kernels/hip/*.hip kernels/hip/*.h crates/intarsia-backend-hip/build.rs crates/intarsia-backend-hip/src/*.rs
if [[ "${CLI:-1}" == 1 ]]; then
  echo "== $tag: build release cli --features hip $(date +%T)"
  deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip > pf27-bin/build-cli-$tag.log 2>&1 < /dev/null
  echo "cli rc=$? $(date +%T)"; grep -aE "^error|Finished" -A6 pf27-bin/build-cli-$tag.log | head -30
  deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/pf27-$tag-cli" < /dev/null && echo "copied to pf27-$tag-cli"
fi
if [[ "${TESTS:-1}" == 1 ]]; then
  deploy/tatooine/dev.sh bash -c "
    exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test mxfp4_a8 2>&1 < /dev/null | tee /work/pf27-bin/build-test-$tag.log | grep -aoE '/target/release/deps/mxfp4_a8-[0-9a-f]+' | tail -1)
    [ -n \"\$exe\" ] || { echo BUILD-FAILED; grep -aE '^error' -A8 /work/pf27-bin/build-test-$tag.log | head -40; exit 3; }
    cp \$exe /work/pf27-bin/mxfp4_a8-$tag && echo test binary mxfp4_a8-$tag \$(sha256sum /work/pf27-bin/mxfp4_a8-$tag | cut -c1-16)" < /dev/null
fi
if [[ "${ISA:-1}" == 1 ]]; then
  bash scripts/pf27/isa.sh mxfp4_a8_gemm | tail -2
  python3 scripts/pf27/resusage.py pf27-out/isa/mxfp4_a8_gemm.res.txt gemm_kernel || true
  python3 scripts/pf27/resusage.py pf27-out/isa/mxfp4_a8_gemm.res.txt direct_kernel || true
fi
echo "== $tag done $(date +%T)"
