#!/usr/bin/env bash
# postchain free-memory watchdog, runs ON spark1: every 20 s, if available memory < 15 GB kill every process of THIS lane (cwd or command line under
# ~/intarsiaml-postchain: cargo, rustc, nvcc, cicc, ptxas, intarsia servers) and log it. Start: nohup bash wd.sh > ~/intarsiaml-postchain/out/wd.log & ; stop: kill the pid.
LANE=$HOME/intarsiaml-postchain; LIM=${LIM:-15}
while :; do
  a=$(free -g | awk 'NR==2{print $7}')
  if [ "${a:-99}" -lt "$LIM" ]; then
    echo "$(date +%T) available ${a} GB < $LIM: killing lane processes"
    for p in $(pgrep -f 'cargo|rustc|nvcc|cicc|ptxas|cc1plus|intarsia|python3|taskset'); do
      [ "$p" = "$$" ] && continue
      if [ "$(readlink /proc/$p/cwd 2>/dev/null)" != "${LANE}" ] && ! readlink /proc/$p/cwd 2>/dev/null | grep -q "^$LANE" && ! tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "$LANE"; then continue; fi
      echo "kill $p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-80)"; kill -9 $p
    done
  fi
  sleep 20
done
