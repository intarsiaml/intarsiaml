#!/usr/bin/env bash
# Lane tp2 window W=w25 (ON tatooine, radiance down, window.sh with SNAP=batch STAGES=w17): repro rate of the intermittent first-step mismatch of
# the 27B batched step at tp=2 (w15: N=4, step 0, every logit ~8 % off; w16: the same full test passed), and the candidate fix.
#  - the test real_35b_tp2_batch_first_steps_repro loads the 27B afresh per rep and runs INTARSIA_REPRO_STEPS batched steps per size 2..16 against the
#    solo steps, counting mismatches (REPRO failures|CTRS f/reps). Each size's first step records its graphs, which is where the w15 failure was.
#  V0 baseline; V1 INTARSIA_TP_BATCH_PREPARE=1 (the peers' graphs are recorded one after the other on the main thread, before rank 0 replays and before
#  the peer threads run: no capture overlaps another rank's launches); V2 batched graphs off (INTARSIA_BATCH_GRAPHS=0); V3 HIP_LAUNCH_BLOCKING=1.
#  First the 27B accuracy of w16 that failed on a missing golden: tp=1, tp=2 twice with dumps, tp=2 against the golden alone (hard gate), run to run.
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; PC=${PC:-none}  # the build head (the queued window was submitted with an older PC)
CAP=${CAP:-1750}; T0=$(date +%s)
left() { echo $(( CAP - ($(date +%s) - T0) )); }
need() { if (( $(left) > $1 )); then return 0; fi; log "SKIPPED (time left $(left)s < $1): $2"; return 1; }
vram() { local m=0 v c; for c in /sys/class/drm/card[0-9]/device/mem_info_vram_used; do v=$(cat "$c" 2>/dev/null || echo 0); (( v > m )) && m=$v; done; echo $m; }
clear() { local i; for i in 1 2 3 4 5 6; do (( $(vram) < 1073741824 )) && return 0; sleep 10; done
  log "CONTENDED before '$1': $(( $(vram) / 1048576 )) MiB in use on a card: NO VERDICT"; return 1; }
M27=/models/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/5233554c5fa56afda40150556b95573c2d7d29c0
G27=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
export INTARSIA_MODELS=/data/models/radiance/hf-cache/hub
log "window start, CAP ${CAP}s, pre-change $PC, VRAM in use $(( $(vram) / 1048576 )) MiB, golden $(ls "$(T $D)/p1-golden" 2>&1 | head -1)"
mkdir -p "$(T $D)/tp2-out"
FLOOR="--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8 --pre-change $PC"
# every container started by a step is removed when the step ends (a timeout kills the docker client, not the container)
sweep() { local keep=$1 c; for c in $(docker ps -q); do grep -q "$c" <<< "$keep" || { docker rm -f "$c" > /dev/null 2>&1 && log "removed leftover container $c"; }; done; }
repro() { # repro <tag> <timeout> <test> <env...>
  local tag=$1 to=$2 test=$3; shift 3
  local keep; keep=$(docker ps -q)
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=0,1 timeout -k 10 "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR="$M27" INTARSIA_BATCH_TEST_TP=2 "$@" \
    cargo test --release -p intarsia-models --test batch_cuda_real -- --test-threads=1 --nocapture --ignored --exact $test \
    > "$O/repro-$tag.log" 2>&1 < /dev/null
  local rc=$?
  sweep "$keep"
  log "repro $tag exit $rc [$*]: $(grep -aE '^REPRO failures|CTRS' "$O/repro-$tag.log" | tail -1) | $(grep -aE '^REPRO rep .*FAIL' "$O/repro-$tag.log" | head -3 | cut -c1-200 | tr '\n' '|')"
}
FS=real_35b_tp2_batch_first_steps_repro; SO=real_35b_tp2_solo_fresh_states_repro
variant() { need 215 "$1" && clear "$1" && repro "$@"; }
# w25: a clean rerun of the w24 skew stages (w24 started with l5's stray 35B server on a card; its s0 control still hung at rep 10, 70 s after that server died).
# A 1 s sampler of both cards (busy %, current sclk/mclk level, power, temperature) runs for the whole window, timestamped, into $O/sampler.log, so a hang
# has a clock/DPM reading of both cards at that second. The ledger + counters + a 12 s watchdog are on in every variant: at a hang they print which rank waits
# (its launch counter is ahead of its flag) before the kernel's own trap fires. HIP does not expose the HSA queue address of a stream, so the queue in an
# abort dump cannot be mapped to a rank; the counters tell who waits.
( while :; do
    for c in /sys/class/drm/card[0-9]; do d=$c/device; [ -r "$d/pp_dpm_sclk" ] || continue
      echo "$(date +%T) ${c##*/} busy=$(cat $d/gpu_busy_percent 2>/dev/null) sclk=$(grep '\*' $d/pp_dpm_sclk | tr -s ' ' | tr '\n' ';') mclk=$(grep '\*' $d/pp_dpm_mclk | tr -s ' ' | tr '\n' ';') pwr_uW=$(cat $d/hwmon/hwmon*/power1_average 2>/dev/null) temp_mC=$(cat $d/hwmon/hwmon*/temp1_input 2>/dev/null)"
    done; sleep 1; done ) > "$O/sampler.log" 2>&1 &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT
log "env: $(uptime | sed 's/.*load/load/') | top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | tr -s ' ' | tr '\n' ';') | containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
B="INTARSIA_REPRO_REPS=20 INTARSIA_REPRO_STATES=2 INTARSIA_REPRO_BATCH_BETWEEN=1 INTARSIA_TP_COLL_TRACE=1 INTARSIA_REPRO_CTRS=1 INTARSIA_REPRO_WATCHDOG=12"
variant c0-control-a 200 $SO $B
variant c1-control-b 200 $SO $B
variant c2-skew-rank1-2ms 200 $SO $B INTARSIA_TP_SKEW_US=0,2000
variant c3-skew-rank0-2ms 200 $SO $B INTARSIA_TP_SKEW_US=2000,0
variant c4-skew-rank1-300us 200 $SO $B INTARSIA_TP_SKEW_US=0,300
variant c5-skew-rank1-20ms 200 $SO $B INTARSIA_TP_SKEW_US=0,20000
variant c6-skew-rank0-20ms 200 $SO $B INTARSIA_TP_SKEW_US=20000,0
variant c7-size-loop-control 200 $FS INTARSIA_REPRO_REPS=8 INTARSIA_REPRO_STEPS=2 INTARSIA_TP_COLL_TRACE=1 INTARSIA_REPRO_WATCHDOG=12
variant c8-control-after-size-loop 200 $SO $B
log "done, time left $(left)s"
