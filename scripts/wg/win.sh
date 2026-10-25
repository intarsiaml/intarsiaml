#!/usr/bin/env bash
# Lane wg, ON tatooine inside a fleetq job that takes radiance down (--takes-down radiance). Runs one command with radiance stopped and
# ALWAYS restores radiance afterwards (trap, also on TERM/INT: a fleetq --expect deadline sends SIGTERM), waiting until it answers.
#   win.sh <command...>            e.g.  win.sh bash ~/wg-bench.sh run
#   win.sh --stamp NAME            (called by prep.sh / bench.sh build after a SUCCESSFUL build) record the sha of the synced tree for NAME
# Radiance is stopped only if every stamp in $WG_NEED (default "tests bench-intarsiaml-wg-var") equals the sha of ~/intarsiaml-wg now:
# a window never runs on a tree that did not build (the kernels and the crates are compiled only on tatooine).
set -uo pipefail
now() { date '+%F %T'; }
LOG=$HOME/wg-logs; mkdir -p "$LOG"
tree_sha() { (cd "$HOME/intarsiaml-wg" && find kernels crates scripts/wg Cargo.toml Cargo.lock -type f -not -path '*/target/*' 2>/dev/null | LC_ALL=C sort | xargs sha1sum | sha1sum | cut -c1-16); }
if [ "${1:-}" = --stamp ]; then tree_sha > "$LOG/stamp-$2"; echo "[win] stamped $2 $(cat "$LOG/stamp-$2")"; exit 0; fi
want=$(tree_sha)
for n in ${WG_NEED:-tests bench-intarsiaml-wg-var}; do
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
trap 'echo "[win] caught TERM/INT $(now): restoring"; exit 143' TERM INT
echo "[win] stopping radiance $(now) (tree $want)"
systemctl --user stop radiance
"$@"
