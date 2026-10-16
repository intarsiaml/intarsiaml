#!/bin/zsh
# usage: withclock.sh <outprefix> <cmd...>   runs cmd with macmon logging GPU clock to <outprefix>.macmon.jsonl
out=$1; shift
macmon pipe -i 200 > "$out.macmon.jsonl" 2>/dev/null &
MP=$!
sleep 1
"$@" > "$out.txt" 2>&1
rc=$?
kill $MP 2>/dev/null
exit $rc
