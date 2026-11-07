#!/usr/bin/env bash
# Lane d2r: copy the Qwen3.8-27B Quark MXFP4 checkpoint (18 GB) from the Mac to spark1 ~/intarsia-d2r/models, then spark1 -> spark2 (resumable).
#   fleetq run --host spark1,spark2 --load net,disk --mem spark1=0,spark2=0 --expect 20 --label "d2r copy 27B" -- scripts/d2r/prep.sh
set -uo pipefail
M=Qwen3.8-27B-Quark-AWQ-MXFP4; SRC=/Users/josef/models/$M/
for h in spark1 spark2; do ssh -n -o BatchMode=yes $h 'mkdir -p ~/intarsia-d2r/models ~/intarsia-d2r/bin ~/intarsia-d2r/out'; done
for try in 1 2 3 4 5 6; do
  rsync -a --append --timeout=60 -e "ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3" --exclude '*.orig*' "$SRC" spark1:intarsia-d2r/models/$M/ && break
  echo "rsync try $try failed ($?), resuming"; sleep 5
done
ssh -n -o BatchMode=yes spark1 "ssh -o BatchMode=yes spark2 mkdir -p intarsia-d2r/models/$M && rsync -a --append --timeout=60 ~/intarsia-d2r/models/$M/ spark2:intarsia-d2r/models/$M/"
for h in spark1 spark2; do ssh -n -o BatchMode=yes $h "ls -la ~/intarsia-d2r/models/$M/model.safetensors; df -h ~ | tail -1"; done
