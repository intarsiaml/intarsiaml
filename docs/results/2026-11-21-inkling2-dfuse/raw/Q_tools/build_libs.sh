#!/bin/sh
# Probe metallibs with the backend's flags (build.rs: -std=metal3.1 -O3 -fno-fast-math, Xcode.app's toolchain).
# usage: build_libs.sh SRC_DIR OUT_DIR
set -e
S=$1; O=$2; mkdir -p "$O"
[ -n "$DEVELOPER_DIR" ] || { [ -d /Applications/Xcode.app/Contents/Developer ] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer; }
one() {
    n=$1; f=$2; shift 2
    xcrun -sdk macosx metal -c -std=metal3.1 -O3 -fno-fast-math "$@" -o "$O/$n.air" "$S/$f"
    xcrun -sdk macosx metallib "$O/$n.air" -o "$O/$n.metallib"
}
one ship ship.metal
one p0 probe.metal
one p_nopro probe.metal -DPR_NOPRO=1
one p_nomath probe.metal -DPR_NOMATH=1
one p_skel probe.metal -DPR_NOPRO=1 -DPR_NOMATH=1
one p_la1 probe.metal -DPR_NB=1
one p_la2 probe.metal -DPR_NB=2
one p_la4 probe.metal -DPR_NB=4
one p_early2 probe.metal -DPR_NB=2 -DPR_EARLY=1
one p_early4 probe.metal -DPR_NB=4 -DPR_EARLY=1
echo "libs ok: $(ls "$O"/*.metallib | wc -l)"
