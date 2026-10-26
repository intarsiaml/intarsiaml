#!/usr/bin/env bash
# Lane tp2 window W=w29 (ON tatooine, radiance down, window.sh with SNAP=batch STAGES=w17): repro rate of the intermittent first-step mismatch of
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
( while :; do
    for c in /sys/class/drm/card[0-9]; do d=$c/device; [ -r "$d/pp_dpm_sclk" ] || continue
      echo "$(date +%T) ${c##*/} busy=$(cat $d/gpu_busy_percent 2>/dev/null) sclk=$(grep '\*' $d/pp_dpm_sclk | tr -s ' ' | tr '\n' ';') mclk=$(grep '\*' $d/pp_dpm_mclk | tr -s ' ' | tr '\n' ';') pwr_uW=$(cat $d/hwmon/hwmon*/power1_average 2>/dev/null) temp_mC=$(cat $d/hwmon/hwmon*/temp1_input 2>/dev/null)"
    done; sleep 1; done ) > "$O/sampler.log" 2>&1 &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null' EXIT
log "env: $(uptime | sed 's/.*load/load/') | top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,5p | tr -s ' ' | tr '\n' ';') | containers: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
variant() { need 215 "$1" && clear "$1" && repro "$@"; }
# w28 h1 (6b17aff, counters in host memory): at the hang rank 0 = 6619, rank 1 = 6620, unchanged 12 s later and already so right after the batched step (and before it: 6490/6491).
# The ledger's cumulative `issued` equals the device counter minus a constant 721 on rank 0 (6619-5898) and 722 on rank 1: rank 1 launched ONE collective kernel that no ledger
# line accounts for, somewhere in the solo rows of the failing batch (the previous probe, 6150/6150, was equal). Every ledger line now carries `dev=<rank0>/<rank1>` (the host
# counters read when the line is written), 400 lines are dumped: the first line where (dev1 - issued1) exceeds (dev0 - issued0) names the pass that launched the extra kernel.
E="INTARSIA_REPRO_REPS=8 INTARSIA_REPRO_STEPS=2 INTARSIA_TP_COLL_TRACE=1 INTARSIA_REPRO_CTRS=1 INTARSIA_REPRO_WATCHDOG=12 INTARSIA_REPRO_WD_SOLO=1 INTARSIA_HIP_COMM_LOG=1 INTARSIA_HIP_CTRS_HOST=1 INTARSIA_TP_BATCH_SYNC=0"
E20="${E/REPS=8/REPS=20}"
# the minimal repro (3 fresh states, one batched step of 3 rows, ~4 s a rep) also fails with host counters (w28 m1: rep 3 of 4): its failing rep has the SAME +1 (1402/1403 after the
# solo rows against 1412/1412 in every passing rep, host call log 1412), and both ranks 9-10 launches short of the host accounting. Its ledger is short: start with it.
M="INTARSIA_REPRO_REPS=20 INTARSIA_REPRO_STATES=2 INTARSIA_REPRO_BATCH_BETWEEN=1 INTARSIA_TP_COLL_TRACE=1 INTARSIA_REPRO_CTRS=1 INTARSIA_REPRO_WATCHDOG=12 INTARSIA_REPRO_WD_SOLO=1 INTARSIA_HIP_CTRS_HOST=1 INTARSIA_TP_BATCH_SYNC=0"
# the mitigation (branch lane-tp2/batch-sync, default ON: every rank's stream idle before a batched tp step) is exercised with INTARSIA_TP_BATCH_SYNC unset; diagnostics set it to 0
Y="INTARSIA_REPRO_REPS=20 INTARSIA_REPRO_STEPS=2 INTARSIA_TP_COLL_TRACE=1 INTARSIA_REPRO_WATCHDOG=12 INTARSIA_REPRO_WD_SOLO=1"
variant m1-minimal-dev 200 $SO $M
variant m2-minimal-dev 200 $SO $M
variant y1-size-loop-sync-on 330 $FS $Y
variant y2-size-loop-sync-on 330 $FS $Y
variant y3-size-loop-sync-on-40 500 $FS ${Y/REPS=20/REPS=40}
variant h1-host-ctrs-dev 330 $FS $E20
log "done, time left $(left)s"
