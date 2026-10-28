#!/usr/bin/env bash
# Lane LDF, ON tatooine inside a fleetq job (--load cpu, no GPU): build the LDS micro-benchmark (scripts/ldf/ldsbench.hip) into ~/intarsiaml-ldf/ldf-bin/ldsbench,
# and ask the assembler whether gfx1201 has the LDS transpose loads (gfx950's ds_read_b64_tr_*, gfx1250's ds_load_tr*).
#   bash ~/intarsiaml-ldf/scripts/ldf/micro.sh
set -uo pipefail
export INTARSIA_REPO=$HOME/intarsiaml-ldf INTARSIA_TARGET_VOL=intarsia-cargo-ldf
cd "$INTARSIA_REPO"; mkdir -p ldf-bin bench-out
deploy/tatooine/dev.sh bash -c '
  cd /work && /opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1201 -o ldf-bin/ldsbench scripts/ldf/ldsbench.hip -Wno-unused-value > bench-out/ldsbench.log 2>&1; echo "hipcc rc=$?"; grep -a error bench-out/ldsbench.log | head -20
  for ins in "ds_read_b64_tr_b8 v[0:1], v2" "ds_read_b64_tr_b16 v[0:1], v2" "ds_load_tr8_b64 v[0:1], v2" "ds_load_tr16_b128 v[0:3], v2" "ds_read_b128 v[0:3], v2" "ds_load_b128 v[0:3], v2"; do
    r=$(echo "$ins" | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn -mcpu=gfx1201 -show-encoding 2>&1 | grep -aE "error|encoding" | head -1)
    echo "asm gfx1201: [$ins] -> $r"
  done' < /dev/null
ls -la ldf-bin/ldsbench
