#!/usr/bin/env bash
# ON tatooine inside a fleetq --load cpu job (lane ink-hip-vrow): (0) with SAVE_T=1, copy the volume's current test executables (branch
# ink/hip-verify verify2 = k5, CPU job #12183) to .scratch/bin/t-verify2/ so window 7 can run #534's gates on them; (1) build the vrow
# tree (.scratch/vrow-src) as intarsia-$NAME with the test targets ink_mix (hip) and verify_rows_real (models).
set -uo pipefail
LANE="$HOME/intarsiaml-ink-hip-verify"
cd "$LANE"; mkdir -p .scratch/bin/t-verify2 logs
export INTARSIA_REPO=$LANE INTARSIA_TARGET_VOL=intarsia-cargo-target-ink-hip-verify INTARSIA_CONTAINER_NAME=ink-hip-verify-build
DEV="$LANE/deploy/tatooine/dev.sh"
if [ -n "${SAVE_T:-}" ]; then
  for t in verify_rows_real inkling2 inkling2_verify_softcap inkling2_dspark inkling2_dspark_cut; do
    exe=$(grep -aoE "/target/release/deps/${t}-[0-9a-f]+" logs/build-mtest-$t.log | tail -1)
    $DEV cp "$exe" /work/.scratch/bin/t-verify2/$t < /dev/null && echo "saved $t <- $exe $(md5sum .scratch/bin/t-verify2/$t | cut -c1-12)"
  done
  docker rm -f ink-hip-verify-build >/dev/null 2>&1
fi
HTESTS="${HTESTS:-ink_mix}" MTESTS="${MTESTS:-verify_rows_real}" LIBTESTS="${LIBTESTS:-}" bash .scratch/build.sh "${NAME:-vrow1}" .scratch/vrow-src
