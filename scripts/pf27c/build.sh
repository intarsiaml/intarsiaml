#!/usr/bin/env bash
# Lane PF27c (from ARX), ON tatooine inside a fleetq job (--load cpu): build the release CLI and the test binaries of one tree in its own volume.
#   bash ~/intarsiaml-pf27c/scripts/pf27c/build.sh dev|base
# -> ~/intarsiaml-pf27c/arx-<tag>-cli, ~/intarsiaml-pf27c/pf27c-bin/<test>-<tag>. CLI=0 / TESTS=0 skip, TBINS lists the test targets.
set -uo pipefail
tag=${1:?dev|base}
R=$HOME/intarsiaml-pf27c
if [[ $tag == base ]]; then T=$R/pf27c-base-tree; V=intarsia-cargo-pf27c-base; else T=$R; V=intarsia-cargo-pf27c; fi
export INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$V
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p $R/pf27c-bin
if ! docker volume inspect "$V" > /dev/null 2>&1; then
  docker volume create "$V" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-glr:/a:ro -v "$V":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $V from intarsia-cargo-glr"
fi
cd "$T"
# rsync keeps the Mac mtimes: force the kernel library and the HIP crate to rebuild
touch kernels/hip/*.hip kernels/hip/*.h crates/intarsia-backend-hip/build.rs crates/intarsia-backend-hip/src/*.rs
if [[ "${CLI:-1}" == 1 ]]; then
  echo "== $tag: build release cli $(date +%T)"
  deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip > $R/pf27c-bin/build-cli-$tag.log 2>&1 < /dev/null
  echo "cli rc=$? $(date +%T)"; grep -aE "^error|^warning: unused|Finished" -A6 $R/pf27c-bin/build-cli-$tag.log | head -40
  deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/pf27c-$tag-cli.tmp" < /dev/null && mv $T/pf27c-$tag-cli.tmp $R/pf27c-$tag-cli && echo "cli -> pf27c-$tag-cli"
fi
if [[ "${TESTS:-1}" == 1 ]]; then
  for t in ${TBINS:-mxfp4_a8 comm ar_norm ar_push}; do
    deploy/tatooine/dev.sh bash -c "
      exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test $t 2>&1 < /dev/null | tee /work/pf27c-test-$t.log | grep -aoE '/target/release/deps/$t-[0-9a-f]+' | tail -1)
      [ -n \"\$exe\" ] || { echo BUILD-FAILED $t; grep -aE '^error' -A12 /work/pf27c-test-$t.log | head -60; exit 3; }
      cp \$exe /work/pf27c-test-$t.bin && echo test binary $t-$tag" < /dev/null
    [[ -f $T/pf27c-test-$t.bin ]] && mv $T/pf27c-test-$t.bin $R/pf27c-bin/$t-$tag
    mv $T/pf27c-test-$t.log $R/pf27c-bin/build-test-$t-$tag.log 2>/dev/null
  done
fi
echo "== $tag done $(date +%T)"
