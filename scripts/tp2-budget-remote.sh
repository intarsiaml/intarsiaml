#!/usr/bin/env bash
# TP=2 self-golden budget: measurements and the validation of the shipped gate. Runs ON tatooine, inside a fleetq lease that
# has radiance down and BOTH R9700s (scripts/tp2-budget-burst.sh takes care of the stop / restart / poll).
#   scripts/tp2-budget-remote.sh m1        # S0, tp=1 equivalent variants
#   scripts/tp2-budget-remote.sh m2        # (same build, after m1) tp=2, tp=2 variants, tp=2 with a deliberate bug, short golden floor
#   scripts/tp2-budget-remote.sh validate  # the shipped gate, no hand-passed budget flags
# Logs: $INTARSIA_REPO/bench-out/tp2b/<TAG>/ (pull them back before the next rsync --delete).
# Needs the release build from `deploy/tatooine/dev.sh cargo build --release -p intarsia-cli --features hip`.
# No `-e`: a gate that FAILS exits non-zero, which is a result here, not an error. Every run's exit code is logged.
set -uo pipefail
export INTARSIA_REPO="${INTARSIA_REPO:-$HOME/intarsiaml-tp2b}" INTARSIA_TARGET_VOL="${INTARSIA_TARGET_VOL:-intarsia-cargo-target-tp2b}"
cd "$INTARSIA_REPO"
D=deploy/tatooine/dev.sh
B=/target/release/intarsia
M=/models/Qwen3-8B
G=/models/golden
GL=$G/qwen3-8b-long-fp32.safetensors
GS=$G/qwen3-8b-fp32.safetensors
TAG=${TAG:-measure}
L=$INTARSIA_REPO/bench-out/tp2b/$TAG   # host path; the container sees the repo at /work
W=/work/bench-out/tp2b/$TAG
mkdir -p "$L"
ts() { date +%T; }
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
GEMVOFF=matmul_bf16_gemv,logits_bf16_gemv

# acc <name> <tp> <"ENV=.. ENV=.."> <accuracy args...>: one `intarsia accuracy` run, full log in $L/<name>.log. The env goes
# through `env` INSIDE the container (dev.sh forwards only INTARSIA_HIP_* / INTARSIA_TP_*).
acc() {
  local name=$1 tp=$2 envs=$3
  shift 3
  local t0=$SECONDS
  # shellcheck disable=SC2086
  $D env $envs $B accuracy --backend hip --model $M --tp "$tp" --mode both "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "[$(ts)] $name rc=$rc ($((SECONDS - t0)) s) env='$envs' args='$*'" | tee -a "$L/runs.txt"
  grep -E "SELF-STATS|^golden gate|^self-golden|OVERALL" "$L/$name.log" | cut -c1-230 | sed "s/^/    /"
  return 0
}

# A run (floor on the fp32 golden + a dump of this configuration's own distribution) and B run (this configuration against S0).
# variant <name> <tp> <env> <extra args for both runs> -- <extra args for B only>
variant() {
  local name=$1 tp=$2 envs=$3
  shift 3
  local both=() bonly=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; continue; fi
    if [ $seen = 1 ]; then bonly+=("$a"); else both+=("$a"); fi
  done
  echo "=== [$(ts)] variant $name (tp=$tp)"
  acc "A-$name" "$tp" "$envs" --golden "$GL" --gate-class bf16 --write-self-golden "$W/sg-$name.safetensors" "${both[@]}"
  acc "B-$name" "$tp" "$envs" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt "${both[@]}" "${bonly[@]}"
}

stage_m1() {
  echo "=== [$(ts)] binary check"
  $D bash -c "grep -c 'built without the .hip. feature' $B; $B --version; ls -la $B" 2>&1 | tail -3 | tee -a "$L/runs.txt"
  # S0: tp=1, device 0, defaults. Everything else is judged against it.
  echo "=== [$(ts)] S0 (tp=1 defaults)"
  acc A-S0 1 "HIP_VISIBLE_DEVICES=0" --golden "$GL" --gate-class bf16 --write-self-golden "$W/S0.safetensors"
  acc B-repeat 1 "HIP_VISIBLE_DEVICES=0" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  acc B-repeat2 1 "HIP_VISIBLE_DEVICES=0" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  # tp=1 equivalent variants (the budget comes from these)
  variant t1-chunk64   1 "HIP_VISIBLE_DEVICES=0" --prefill-chunk 64 -- --allow-config-mismatch
  variant t1-chunk1024 1 "HIP_VISIBLE_DEVICES=0" --prefill-chunk 1024 -- --allow-config-mismatch
  variant t1-simt      1 "HIP_VISIBLE_DEVICES=0 INTARSIA_HIP_DISABLE=$SIMT"
  variant t1-nofusion  1 "HIP_VISIBLE_DEVICES=0 INTARSIA_NO_FUSION=1"
  variant t1-gemvoff   1 "HIP_VISIBLE_DEVICES=0 INTARSIA_HIP_DISABLE=$GEMVOFF"
  variant t1-dev1      1 "HIP_VISIBLE_DEVICES=1"
  variant t1-combo     1 "HIP_VISIBLE_DEVICES=0 INTARSIA_NO_FUSION=1 INTARSIA_HIP_DISABLE=$SIMT" --prefill-chunk 64 -- --allow-config-mismatch
}

stage_m2() {
  # tp=2 (measured, NOT part of the budget); S0 comes from stage m1 of the same build
  variant t2           2 ""
  acc B-t2-repeat 2 "" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  variant t2-bf16part  2 "INTARSIA_TP_PARTIAL=bf16"
  variant t2-host      2 "" --comm host
  variant t2-chunk64   2 "" --prefill-chunk 64 -- --allow-config-mismatch
  variant t2-simt      2 "INTARSIA_HIP_DISABLE=$SIMT"
  # tp=2 spread within tp=2 (against the tp=2 default dump)
  for v in t2-bf16part t2-chunk64 t2-simt t2-host; do
    case $v in
      t2-bf16part) e="INTARSIA_TP_PARTIAL=bf16"; x=() ;;
      t2-chunk64) e=""; x=(--prefill-chunk 64 --allow-config-mismatch) ;;
      t2-simt) e="INTARSIA_HIP_DISABLE=$SIMT"; x=() ;;
      t2-host) e=""; x=(--comm host) ;;
    esac
    acc "C-$v" 2 "$e" --self-golden "$W/sg-t2.safetensors" --allow-same-code --revlist /work/revlist.txt "${x[@]}"
  done
  # deliberate TP bug: rank 1's partial scaled before every all-reduce
  for s in 1.01 1.003 1.001; do
    variant "bug-$s" 2 "INTARSIA_TP_TEST_BUG=scale:$s"
  done
  # the short fp32 golden: floor only
  echo "=== [$(ts)] short fp32 golden (floor)"
  acc F-S0 1 "HIP_VISIBLE_DEVICES=0" --golden "$GS" --gate-class bf16
  acc F-t1-chunk64 1 "HIP_VISIBLE_DEVICES=0" --golden "$GS" --gate-class bf16 --prefill-chunk 64
  acc F-t1-simt 1 "HIP_VISIBLE_DEVICES=0 INTARSIA_HIP_DISABLE=$SIMT" --golden "$GS" --gate-class bf16
  acc F-t2 2 "" --golden "$GS" --gate-class bf16
  acc F-bug-1.01 2 "INTARSIA_TP_TEST_BUG=scale:1.01" --golden "$GS" --gate-class bf16
}

stage_validate() {
  # The shipped gate, no budget flags: a fresh S0 from this very build, then tp=2 clean must PASS, bugs must FAIL.
  echo "=== [$(ts)] binary check"
  $D bash -c "grep -c 'built without the .hip. feature' $B; ls -la $B" 2>&1 | tail -2 | tee -a "$L/runs.txt"
  acc A-S0 1 "HIP_VISIBLE_DEVICES=0" --golden "$GL" --gate-class bf16 --write-self-golden "$W/S0.safetensors"
  acc V-t2 2 "" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  acc V-t2-floor 2 "" --golden "$GL" --gate-class bf16
  acc V-t2-bf16part 2 "INTARSIA_TP_PARTIAL=bf16" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  acc V-t2-host 2 "" --comm host --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  for s in 1.01 1.003; do
    acc "V-bug-$s" 2 "INTARSIA_TP_TEST_BUG=scale:$s" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  done
  # tp=1 variants against S0 must still judge by the DEFAULT budget (the TP budget is for world != writer's world)
  acc V-t1-chunk64 1 "HIP_VISIBLE_DEVICES=0" --prefill-chunk 64 --allow-config-mismatch --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
  acc V-t1-repeat 1 "HIP_VISIBLE_DEVICES=0" --self-golden "$W/S0.safetensors" --allow-same-code --revlist /work/revlist.txt
}

for s in "$@"; do
  echo "################ stage $s start $(ts)"
  "stage_$s"
  echo "################ stage $s end $(ts)"
done
