#!/usr/bin/env bash
# Lane rocm102-bisect (ON tatooine, CPU only, fleetq --load cpu): per image, build the CLI with the kernel archive unbundled
# (INTARSIA_HIP_KERNELS_NOBUNDLE=1, build.rs), capture its final link command, and copy out the binary and build.rs's objects.
set -u
. "$(dirname "$0")/lib.sh"
mkdir -p "$B"
for a in ${ARMS:-r7 r10}; do
  t0=$(date +%s)
  dk "$a" bash -c '
    set -e; a=$1; export INTARSIA_HIP_KERNELS_NOBUNDLE=1 INTARSIA_BUILD_STAMP="$2"
    cargo build --release --locked -p intarsia-cli --features hip
    mkdir -p /target/rustc-tmp; rm -rf /target/rustc-tmp/*
    # the extra rustc args rebuild only the CLI crate; save-temps (TMPDIR in the volume) keeps the link inputs for re-links
    TMPDIR=/target/rustc-tmp cargo rustc --release --locked -p intarsia-cli --features hip --bin intarsia -- --print link-args -C save-temps > /work/.bisect/link-$a.raw
    grep -a "\"cc\"" /work/.bisect/link-$a.raw | tail -1 > /work/.bisect/link-$a.txt
    od=$(grep -ao "\"-L\" \"[^\"]*intarsia-backend-hip-[^\"]*/out\"" /work/.bisect/link-$a.txt | head -1 | cut -d" " -f2 | tr -d "\"")
    echo "OUT_DIR $od"; echo "$od" > /work/.bisect/outdir-$a.txt
    mkdir -p /work/.bisect/objs/$a /work/.bisect/bins/$a; rm -f /work/.bisect/objs/$a/*.o
    cp "$od"/*.o /work/.bisect/objs/$a/; cp /target/release/intarsia /work/.bisect/bins/$a/intarsia
    ls /work/.bisect/objs/$a | wc -l; hipcc --version | head -2' _ "$a" "$STAMP" < /dev/null > "$O/mkbins-$a.log" 2>&1
  log "mkbins $a exit $? ($(( $(date +%s) - t0 )) s): $(grep -aE '^OUT_DIR|^[0-9]+$|HIP version|error' "$O/mkbins-$a.log" | tr '\n' '|' | cut -c1-300)"
done
