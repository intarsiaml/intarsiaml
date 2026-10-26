#!/usr/bin/env bash
# Lane wadm (from the Mac): ONE radiance-down window on tatooine for the hip admission tests, after build.sh. Stops radiance, runs the clean
# admission binary (all of tests/admission.rs: pass 1 knob unset, pass 2 knob on for the kernels pass 1 gave no case), then each mutant binary in
# its knob-on pass only (INTARSIA_TEST_CHILD_KERNELS=<the mutated kernels>, which is exactly what pass 2 runs), and m1 through the whole test when
# the clean run was short. Radiance is restored in an EXIT trap on tatooine and polled until it answers. Logs: docs/results/2026-10-26-wadm-wmma-admission/raw/.
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --quiet gpu,cpu --expect 35 --mem 30 --label "wadm: admission" -- scripts/wadm/win.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$(cd "$S/../.." && pwd)/docs/results/2026-10-26-wadm-wmma-admission/raw"
mkdir -p "$RAW"
ssh -o BatchMode=yes tatooine "FLEETQ_JOB=$FLEETQ_JOB bash -s" <<'EOS' 2>&1 | tee "$RAW/window.log"
set -uo pipefail
now() { date '+%F %T'; }
O=$HOME/wadm-out; rm -rf "$O"; mkdir -p "$O"
restore() {
  echo "[window] starting radiance $(now)"
  systemctl --user start radiance < /dev/null
  for i in $(seq 1 48); do
    if curl -s -m 3 http://localhost:8080/v1/models < /dev/null | grep -q '"data":\['; then echo "[window] radiance answers $(now) (after $((i * 5)) s)"; return 0; fi
    sleep 5
  done
  echo "[window] RADIANCE DID NOT ANSWER within 240 s $(now)"; return 1
}
trap restore EXIT
echo "[window] job $FLEETQ_JOB; stopping radiance $(now)"
systemctl --user stop radiance < /dev/null
sleep 2
echo "[window] radiance stopped $(now)"
export INTARSIA_REPO=$HOME/intarsiaml-wadm INTARSIA_TARGET_VOL=intarsia-cargo-wadm HIP_VISIBLE_DEVICES=0
D=$INTARSIA_REPO/deploy/tatooine/dev.sh
KNOB="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
run() { # <log> <timeout s> <env words> <binary> <test args...>
  local log=$1 to=$2 envs=$3 bin=$4; shift 4
  local t0=$SECONDS
  # shellcheck disable=SC2086
  timeout "$to" $D env $envs /target/bin/$bin "$@" --nocapture > "$O/$log.out" 2> "$O/$log.err" < /dev/null
  RC=$?; DT=$((SECONDS - t0))
  echo "[window] $log exit $RC in ${DT}s $(now): $(grep -aE '^test |test result|child graded' "$O/$log.out" | tr '\n' '|')"
}
run clean 1500 "" clean
clean_rc=$RC; clean_dt=$DT
# a mutant must FAIL the knob-on pass and the failure must name the mutated kernel
mut() { # <name> <kernels comma list>
  run "$1" 300 "$KNOB INTARSIA_TEST_CHILD_KERNELS=$2" "$1" --exact every_hip_kernel_matches_the_reference
  local first=${2%%,*}
  if [ "$RC" -ne 0 ] && grep -aq "kernels not admitted" "$O/$1.out" "$O/$1.err" && grep -aq "FAIL.*${first}\|${first}.*\(FAIL\|NO CASES\|REJECTED\)" "$O/$1.out" "$O/$1.err"; then
    echo "[window] MUTANT $1 CAUGHT (names $first)"; else echo "[window] MUTANT $1 NOT CAUGHT (exit $RC)"; fi
}
mut m1 matmul_pair_bf16_wmma
mut m2 matmul_fp8_pair_wmma
mut m3 rmsnorm_add_matmul_fp8_pair_wmma
mut k1 matmul_fp8_wmma,matmul_fp8_wmma_rows
# the same mutants must not trip the clean binary in the same mode (control for the mode itself)
run control 300 "$KNOB INTARSIA_TEST_CHILD_KERNELS=matmul_pair_bf16_wmma,matmul_fp8_pair_wmma,rmsnorm_add_matmul_fp8_pair_wmma,matmul_fp8_wmma,matmul_fp8_wmma_rows" clean --exact every_hip_kernel_matches_the_reference
if [ "$clean_dt" -lt 420 ]; then
  run m1-full 900 "" m1 --exact every_hip_kernel_matches_the_reference
  if [ "$RC" -ne 0 ]; then echo "[window] MUTANT m1 CAUGHT by the full test"; else echo "[window] MUTANT m1 NOT CAUGHT by the full test"; fi
else echo "[window] clean run took ${clean_dt}s: m1 full run skipped"; fi
echo "[window] done $(now) (clean exit $clean_rc)"
EOS
rc=${PIPESTATUS[0]}
rsync -a "tatooine:wadm-out/" "$RAW/" 2>/dev/null
exit $rc
