#!/usr/bin/env bash
# Lane lmhead-fp8-default: copy the Qwen3.8-27B Quark MXFP4 checkpoint (18 GB, the recipe's checkpoint, bf16 lm_head) from this Mac to spark1 ~/lmh-cuda/models.
# Resumable (rsync --append: the partial file is a clean prefix; checksum resume was CPU-bound) with keepalives: the first attempt stalled at 4.6 GB with the TCP stream idle.
#   fleetq run --host spark1 --load net,disk --mem 0 --expect 30 --label "..." -- scripts/lmhead/ccopy.sh
set -uo pipefail
D=spark1:lmh-cuda/models/Qwen3.8-27B-Quark-AWQ-MXFP4/; SRC=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/lmh-cuda/models'
for try in 1 2 3 4 5 6; do
  rsync -a --append --timeout=60 -e "ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3" --exclude '*.orig*' "$SRC" "$D" && break
  echo "rsync try $try failed ($?), resuming"; sleep 5
done
ssh -n -o BatchMode=yes spark1 'ls -la ~/lmh-cuda/models/Qwen3.8-27B-Quark-AWQ-MXFP4/model.safetensors'
