#!/usr/bin/env bash
# Lane wg, ON tatooine inside a fleetq job that takes radiance down (--takes-down radiance). Runs one command with radiance stopped and
# ALWAYS restores radiance afterwards (trap, also on TERM/INT: a fleetq --expect deadline sends SIGTERM), waiting until it answers.
#   win.sh <command...>            e.g.  win.sh bash ~/wg-bench.sh run
#   win.sh --sha                   print the sha of the synced tree
#   win.sh --stamp NAME            (called by prep.sh / bench.sh build after a SUCCESSFUL build) record the sha of the synced tree for NAME
# Radiance is stopped only if every stamp in $WG_NEED (default "tests bench-intarsiaml-wg-var") equals the sha of ~/intarsiaml-wg now:
# a window never runs on a tree that did not build (the kernels and the crates are compiled only on tatooine).
set -uo pipefail
now() { date '+%F %T'; }
LOG=$HOME/wg-logs; mkdir -p "$LOG"
tree_sha() { (cd "$HOME/intarsiaml-wg" && find kernels crates scripts/wg Cargo.toml Cargo.lock -type f -not -path '*/target/*' 2>/dev/null | LC_ALL=C sort | xargs sha1sum | sha1sum | cut -c1-16); }
if [ "${1:-}" = --sha ]; then tree_sha; exit 0; fi
if [ "${1:-}" = --stamp ]; then tree_sha > "$LOG/stamp-$2"; echo "[win] stamped $2 $(cat "$LOG/stamp-$2")"; exit 0; fi
want=$(tree_sha)
# Self-heal (radiance still UP): a stale or missing stamp means the tree changed after its last build (a queue-order accident, not a failure).
# Rebuild that stamp here, in the job's own slot, instead of wasting the window; a build that fails still leaves the stamp stale and we refuse below.
heal() {
  case $1 in
    tests) echo "[win] healing stamp tests"; bash "$HOME/wg-prep.sh" tests > "$LOG/heal-tests.log" 2>&1 ;;
    bench-intarsiaml-wg-var) echo "[win] healing stamp $1"; BUILD=$HOME/wg-build-v2 REPO=$HOME/intarsiaml-wg-var bash "$HOME/wg-bench.sh" build > "$LOG/heal-bench.log" 2>&1 ;;
    probe) [ -f "$HOME/wg-probe.sh" ] && { echo "[win] healing stamp probe"; bash "$HOME/wg-probe.sh" build > "$LOG/heal-probe.log" 2>&1; } ;;
  esac
}
for n in ${WG_NEED:-tests bench-intarsiaml-wg-var probe}; do
  [ "$(cat "$LOG/stamp-$n" 2>/dev/null || echo none)" = "$want" ] || heal "$n"
done
for n in ${WG_NEED:-tests bench-intarsiaml-wg-var}; do   # (the probe is optional: its tail skips itself)
  have=$(cat "$LOG/stamp-$n" 2>/dev/null || echo none)
  if [ "$have" != "$want" ]; then
    echo "[win] REFUSING to stop radiance: stamp '$n' is $have, tree is $want (the tree did not build, or changed after the build). Radiance untouched."
    exit 3
  fi
done
restore() {
  trap - EXIT TERM INT
  echo "[win] starting radiance $(now)"
  systemctl --user start radiance
  for i in $(seq 1 36); do
    if curl -s -m 3 http://localhost:8080/v1/models | grep -q '"data":\['; then
      echo "[win] radiance answers $(now) (after $((i * 5)) s); $(systemctl --user is-active radiance)"; return 0
    fi
    sleep 5
  done
  echo "[win] RADIANCE DID NOT ANSWER within 180 s $(now); $(systemctl --user is-active radiance)"; return 1
}
trap restore EXIT
CHILD=
# the command runs as a background job and is `wait`ed for, so a TERM/INT trap fires at once (bash defers a trap until a foreground child ends)
trap 'echo "[win] caught TERM/INT $(now): restoring"; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null; exit 143' TERM INT
echo "[win] stopping radiance $(now) (tree $want)"
systemctl --user stop radiance
"$@" <&0 &
CHILD=$!
wait "$CHILD"; rc=$?; CHILD=
exit $rc
