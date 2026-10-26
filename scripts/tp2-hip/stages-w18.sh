#!/usr/bin/env bash
# Lane tp2 window W=w18 (ON tatooine, radiance down, window.sh with SNAP=batch STAGES=w17): repro rate of the intermittent first-step mismatch of
# the 27B batched step at tp=2 (w15: N=4, step 0, every logit ~8 % off; w16: the same full test passed), and the candidate fix.
#  - the test real_35b_tp2_batch_first_steps_repro loads the 27B afresh per rep and runs INTARSIA_REPRO_STEPS batched steps per size 2..16 against the
#    solo steps, counting mismatches (REPRO failures f/reps). Each size's first step records its graphs, which is where the w15 failure was.
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
  log "repro $tag exit $rc [$*]: $(grep -aE '^REPRO failures' "$O/repro-$tag.log" | tail -1) | $(grep -aE '^REPRO rep .*FAIL' "$O/repro-$tag.log" | head -3 | cut -c1-200 | tr '\n' '|')"
}
FS=real_35b_tp2_batch_first_steps_repro
# w17 reproduced it: V0 (graphs on, a fresh model per rep, sizes 2..16) failed at rep 1 on "batch 4 step 0 row 0 (sequence 7, context 221 -> 222)" -- the
# very row of w15 -- with the solo logit identical to w15 (2.473164) and the batched one different every time (2.27, 4.58): garbage, not a rounding.
# All rows of the step were wrong (rows 2 and 3 by ~200) and the process then hung ~50 s in a collective until the bounded wait trapped (0x1016): a
# rank desync. INTARSIA_HIP_AR_DRAIN=1 did not help (v1: batch 3 step 0, rep 0). The test now leaves at the first mismatch and prints both ranks' graph
# counters. Each variant runs up to 12 fresh-model reps x sizes 2..16 x 2 steps; "stopped at the first, after N reps" is the rate evidence.
variant() { need 300 "$1" && clear "$1" && repro "$@"; }
variant v0-graphs-on 290 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2
variant v1-prepare 290 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2 INTARSIA_TP_BATCH_PREPARE=1
variant v2-graphs-off 290 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2 INTARSIA_BATCH_GRAPHS=0
variant v3-blocking 290 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2 HIP_LAUNCH_BLOCKING=1
variant v4-graphs-on-again 290 $FS INTARSIA_REPRO_REPS=12 INTARSIA_REPRO_STEPS=2
log "done, time left $(left)s"
