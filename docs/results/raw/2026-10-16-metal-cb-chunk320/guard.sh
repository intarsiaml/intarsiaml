# sourced by the slot scripts: hard deadline handling (coordinator: nothing of ours running or holding endor after 11:20 EDT)
SOFT=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-16 10:55" +%s)   # no new phase / round after this
HARD=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-16 11:12" +%s)   # kill our own work at this time
past_soft() { [ $(date +%s) -ge $SOFT ]; }
start_watchdog() {  # kills this script (its EXIT trap stops the server) at HARD
  local left=$(( HARD - $(date +%s) )); (( left < 1 )) && left=1
  ( sleep $left; echo "WATCHDOG: hard deadline reached, terminating $$"; kill -TERM $$ ) &
  WDPID=$!
}
