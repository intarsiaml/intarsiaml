#!/usr/bin/env bash
# Lane NQ27 (from ARX), ON tatooine inside a fleetq job (--load cpu): build the release CLI and the test binaries of one tree in its own volume.
#   bash ~/intarsiaml-nq27/scripts/nq27/build.sh dev|base
# -> ~/intarsiaml-nq27/arx-<tag>-cli, ~/intarsiaml-nq27/nq27-bin/<test>-<tag>. CLI=0 / TESTS=0 skip, TBINS lists the test targets.
set -uo pipefail
tag=${1:?dev|base}
R=$HOME/intarsiaml-nq27
if [[ $tag == base ]]; then T=$R/nq27-base-tree; V=intarsia-cargo-nq27-base; else T=$R; V=intarsia-cargo-nq27; fi
export INTARSIA_REPO=$T INTARSIA_TARGET_VOL=$V
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
mkdir -p $R/nq27-bin
if ! docker volume inspect "$V" > /dev/null 2>&1; then
  docker volume create "$V" > /dev/null
  docker run --rm --user 0 -v intarsia-cargo-glr:/a:ro -v "$V":/b "$IMG" sh -c 'cp -a /a/. /b/ && chmod 0777 /b' && echo "seeded $V from intarsia-cargo-glr"
fi
cd "$T"
# rsync keeps the Mac mtimes: force the kernel library and the HIP crate to rebuild
touch kernels/hip/*.hip kernels/hip/*.h crates/intarsia-backend-hip/build.rs crates/intarsia-backend-hip/src/*.rs
if [[ "${CLI:-1}" == 1 ]]; then
  echo "== $tag: build release cli $(date +%T)"
  deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip > $R/nq27-bin/build-cli-$tag.log 2>&1 < /dev/null
  echo "cli rc=$? $(date +%T)"; grep -aE "^error|^warning: unused|Finished" -A6 $R/nq27-bin/build-cli-$tag.log | head -40
  deploy/tatooine/dev.sh sh -c "cp /target/release/intarsia /work/nq27-$tag-cli.tmp" < /dev/null && mv $T/nq27-$tag-cli.tmp $R/nq27-$tag-cli && echo "cli -> nq27-$tag-cli"
fi
if [[ "${TESTS:-1}" == 1 ]]; then
  for t in ${TBINS:-mxfp4_a8 comm ar_norm ar_push}; do
    deploy/tatooine/dev.sh bash -c "
      exe=\$(cargo test --release --no-run -p intarsia-backend-hip --test $t 2>&1 < /dev/null | tee /work/nq27-test-$t.log | grep -aoE '/target/release/deps/$t-[0-9a-f]+' | tail -1)
      [ -n \"\$exe\" ] || { echo BUILD-FAILED $t; grep -aE '^error' -A12 /work/nq27-test-$t.log | head -60; exit 3; }
      cp \$exe /work/nq27-test-$t.bin && echo test binary $t-$tag" < /dev/null
    [[ -f $T/nq27-test-$t.bin ]] && mv $T/nq27-test-$t.bin $R/nq27-bin/$t-$tag
    mv $T/nq27-test-$t.log $R/nq27-bin/build-test-$t-$tag.log 2>/dev/null
  done
fi
echo "== $tag done $(date +%T)"
