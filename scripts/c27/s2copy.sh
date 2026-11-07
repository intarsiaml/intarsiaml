#!/usr/bin/env bash
# Lane c27: copy the 27B checkpoint, the src tree, and the built binary (target-src/release/intarsia) from spark1 to spark2 ~/intarsia-c27, over the Spark-Spark link.
#   fleetq run --host spark1,spark2 --load net,disk --mem spark1=0,spark2=0 --expect 15 --label "c27 copy to spark2" -- scripts/c27/s2copy.sh
set -uo pipefail
T=${TREE:-src}
ssh -n -o BatchMode=yes spark2 'mkdir -p ~/intarsia-c27/models ~/intarsia-c27/target-'$T'/release ~/intarsia-c27/xout'
ssh -n -o BatchMode=yes spark1 "rsync -a --partial -e ssh ~/intarsia-c27/models/ 10.100.152.3:intarsia-c27/models/ && rsync -a -e ssh ~/intarsia-c27/target-$T/release/intarsia 10.100.152.3:intarsia-c27/target-$T/release/ && rsync -a --exclude target -e ssh ~/intarsia-c27/$T/ 10.100.152.3:intarsia-c27/$T/"
ssh -n -o BatchMode=yes spark2 'ls -la ~/intarsia-c27/models/*/model.safetensors ~/intarsia-c27/target-'$T'/release/intarsia'
