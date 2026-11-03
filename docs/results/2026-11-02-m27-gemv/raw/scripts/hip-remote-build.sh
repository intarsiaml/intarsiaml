#!/bin/bash
# ON tatooine (stdin of ssh bash -s): build one tree into its own bind-mounted target volume. args: name
set -euo pipefail
name=$1
IMG=intarsia-hip-dev:rocm7.2.4-rust1.90.0
R=$HOME/intarsia-m27-gemv/$name
V=intarsia-cargo-m27gemv-$name
RV=intarsia-cargo-m27gemv-registry
D=/data/intarsia-m27gemv-target-$name
mkdir -p $D && chmod 0777 $D
docker volume inspect $V >/dev/null 2>&1 || docker volume create --driver local -o type=none -o o=bind -o device=$D $V >/dev/null
if ! docker volume inspect $RV >/dev/null 2>&1; then
  docker volume create $RV >/dev/null
  docker run --rm --user 0 -v intarsia-cargo-target-registry:/src:ro -v $RV:/dst $IMG bash -c 'cp -a /src/. /dst/ && chmod -R a+rwX /dst' 2>&1 | tail -2
fi
cd $R
INTARSIA_REPO=$R INTARSIA_TARGET_VOL=$V INTARSIA_REGISTRY_VOL=$RV deploy/tatooine/dev.sh cargo build --release --locked --features hip -p intarsia-cli 2>&1 < /dev/null | tail -4
