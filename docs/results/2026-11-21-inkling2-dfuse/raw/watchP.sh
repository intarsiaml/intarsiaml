#!/bin/sh
# watch #11305 (job P): poll fleetq every 3 min while queued; once it runs, check every 30 s that it does not end at once
# and that the build step finishes (times.txt "build" within 15 min); report then exit (a second watch follows the end)
J=11305; O=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/jobP.out; T=/opt/code/intarsiaml/.worktrees/ink-dfuse/.scratch/P/times.txt
st() { fleetq show $J --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d.get('exit_code'), d.get('deadline'))" 2>/dev/null || echo unknown; }
while [ "$(st | cut -d' ' -f1)" = queued ]; do sleep 180; done
echo "LEFT QUEUE $(date '+%H:%M'): $(st)"
end=$(( $(date +%s) + 900 ))
while [ $(date +%s) -lt $end ]; do
  s=$(st | cut -d' ' -f1)
  if [ "$s" != running ]; then echo "ENDED $(date '+%H:%M'): $(st)"; grep -v "waiting\|has your owner\|If this work" $O | tail -15; exit 0; fi
  if grep -q "^build " $T 2>/dev/null; then echo "PAST BUILD $(date '+%H:%M'): $(st)"; cat $T; exit 0; fi
  sleep 30
done
echo "BUILD NOT DONE after 15 min $(date '+%H:%M'): $(st)"; grep -v "waiting\|has your owner\|If this work" $O | tail -8
