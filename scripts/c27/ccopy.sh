#!/usr/bin/env bash
# Lane c27: copy the Qwen3.8-27B Quark MXFP4 checkpoint (18 GB) from the Mac to spark1 ~/intarsia-c27/models (resumable).
#   fleetq run --host spark1 --load net,disk --mem 0 --expect 30 --label "c27 copy 27B" -- scripts/c27/ccopy.sh
set -uo pipefail
D=spark1:intarsia-c27/models/Qwen3.8-27B-Quark-AWQ-MXFP4/; SRC=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/
ssh -n -o BatchMode=yes spark1 'mkdir -p ~/intarsia-c27/models ~/intarsia-c27/out'
for try in 1 2 3 4 5 6; do
  rsync -a --append --timeout=60 -e "ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3" --exclude '*.orig*' "$SRC" "$D" && break
  echo "rsync try $try failed ($?), resuming"; sleep 5
done
ssh -n -o BatchMode=yes spark1 'ls -la ~/intarsia-c27/models/Qwen3.8-27B-Quark-AWQ-MXFP4/model.safetensors'
