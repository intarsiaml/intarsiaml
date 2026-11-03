#!/bin/bash
# on endor inside a tatooine slot that takes radiance down: stop radiance, run the matmul_mxfp4 sweeps for head and main in the dev
# container, restore radiance (trap, always). Keep the outage as short as possible: everything is built before.
set -uo pipefail
O=/opt/code/intarsiaml/.worktrees/m27-gemv/docs/results/2026-11-02-m27-gemv/raw/hip; mkdir -p $O
restore() {
  ssh -o BatchMode=yes tatooine 'systemctl --user start radiance; for i in $(seq 1 36); do curl -s -m 3 http://localhost:8080/v1/models | grep -q "\"data\":\[" && break; sleep 5; done; echo "[win] radiance $(systemctl --user is-active radiance) after $((i * 5)) s"'
  echo "[win] restored $(date +%T)"
}
trap restore EXIT
echo "[win] stop $(date +%T)"
ssh -o BatchMode=yes tatooine 'systemctl --user stop radiance; sleep 3; echo "[win] radiance $(systemctl --user is-active radiance)"'
run() { # run <name> <args...>: the tree's binary inside the dev container, GPU 0 only
  local n=$1; shift
  ssh -n -o BatchMode=yes tatooine "R=\$HOME/intarsia-m27-gemv/$n; HIP_VISIBLE_DEVICES=0 INTARSIA_REPO=\$R INTARSIA_TARGET_VOL=intarsia-cargo-m27gemv-$n INTARSIA_REGISTRY_VOL=intarsia-cargo-m27gemv-registry \$R/deploy/tatooine/dev.sh /target/release/intarsia $*"
}
for n in head main; do
  run $n kernels test --backend hip --op matmul_mxfp4 --quick -v > $O/$n-mxfp4-quick.log 2>&1; echo "$n mxfp4 quick rc=$?"
  run $n kernels test --backend hip --op matmul_mxfp4 -v > $O/$n-mxfp4-full.log 2>&1; echo "$n mxfp4 full rc=$?"
  run $n kernels test --backend hip --quick > $O/$n-all-quick.log 2>&1; echo "$n all quick rc=$?"
  run $n plan show -m /models/Qwen3.6-35B-A3B-NVFP4 --backend hip > $O/$n-plan-35b.txt 2>&1; echo "$n plan 35b rc=$?"
  run $n plan show -m /models/Qwen3-8B --backend hip > $O/$n-plan-8b.txt 2>&1; echo "$n plan 8b rc=$?"
done
for m in 35b 8b; do
  if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' $O/main-plan-$m.txt) <(grep -v -iE 'built|stamp|took|loaded in|ms\b' $O/head-plan-$m.txt) >/dev/null; then echo "plan show $m: identical ($(wc -l < $O/head-plan-$m.txt) lines)"; else echo "plan show $m: DIFFERS"; fi
done
