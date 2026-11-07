#!/usr/bin/env bash
# Lane q2s: copy the Qwen3.8-27B Quark MXFP4 checkpoint (18 GB) from the Mac to spark1 ~/intarsia-q2s/models, then spark1 -> spark2 (resumable).
#   fleetq run --host spark1,spark2 --load net,disk --mem spark1=0,spark2=0 --expect 20 --label "q2s copy 27B" -- scripts/q2s/prep.sh
set -uo pipefail
M=Qwen3.8-27B-Quark-AWQ-MXFP4; SRC=/Users/josef/models/$M/
for h in spark1 spark2; do ssh -n -o BatchMode=yes $h 'mkdir -p ~/intarsia-q2s/models ~/intarsia-q2s/bin ~/intarsia-q2s/out'; done
for try in 1 2 3 4 5 6; do
  rsync -a --append --timeout=60 -e "ssh -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3" --exclude '*.orig*' "$SRC" spark1:intarsia-q2s/models/$M/ && break
  echo "rsync try $try failed ($?), resuming"; sleep 5
done
ssh -n -o BatchMode=yes spark1 "ssh -o BatchMode=yes spark2 mkdir -p intarsia-q2s/models/$M && rsync -a --append --timeout=60 ~/intarsia-q2s/models/$M/ spark2:intarsia-q2s/models/$M/"
for h in spark1 spark2; do ssh -n -o BatchMode=yes $h "ls -la ~/intarsia-q2s/models/$M/model.safetensors; df -h ~ | tail -1"; done
