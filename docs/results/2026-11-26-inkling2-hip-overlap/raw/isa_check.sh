#!/usr/bin/env bash
# Same-ISA check (lane ink-hip-overlap; coordinator's Q1 answer), ON tatooine inside a fleetq --load cpu job: the device assembly of
# the HIP translation units this lane changed, from main's tree (.scratch/main-src, origin/main a4c8f3b19) and this lane's tree, with
# build.rs's flags; then isa_cmp.py per kernel: which kernels' instructions are byte-identical and which differ.
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-bpf
O=$R/logs/isa; mkdir -p "$O"
TUS="ink_mix.hip ink_mix_rows.hip ink_mix_segs.hip ink_attn.hip ink_attn_rows.hip ink_route_fused.hip ik_runtime.hip"
cat > "$O/asm.sh" <<'IN'
set -u
for t in $TUS; do
  for side in main br; do
    src=/work/kernels/hip; [ $side = main ] && src=/work/.scratch/main-src/kernels/hip
    [ -f $src/$t ] || { echo "no $side $t"; continue; }
    /opt/rocm/bin/hipcc --cuda-device-only -S -O3 -std=c++17 -fPIC -Wall -Wno-unused-function --offload-arch=gfx1201 -I $src/.. -I $src \
      -o /work/logs/isa/$side-${t%.hip}.s $src/$t 2> /work/logs/isa/$side-${t%.hip}.err || echo "hipcc failed: $side $t"
  done
done
echo asm done
IN
cd "$R" && INK_CONTAINER=ink-hip-bpf-isa HIP_VISIBLE_DEVICES=0 bash .scratch/run.sh env TUS="$TUS" bash /work/logs/isa/asm.sh
python3 "$R/.scratch/isa_cmp.py" "$O" $TUS
