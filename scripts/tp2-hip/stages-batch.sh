#!/usr/bin/env bash
# Lane tp2/batch, window stage W=batch (ON tatooine, radiance down, called by window.sh): continuous batching under in-process
# tensor parallelism (`serve --max-batch N --tp 2 --backend hip`, branch lane-tp2/batch). Trees: D = the batch build
# (scripts/tp2-hip/sync.sh batch <rev>; build.sh with HIPTEST=1 HIPTEST_MODELS=1 TREES=batch=<stamp>, so the GPU tests of
# intarsia-backend-hip AND intarsia-models' batch_recurrent / batch_cuda_real are prebuilt in intarsia-cargo-target-tp2-batch-hiptest),
# B = the lane-tp2/stack build this branch starts from (511d1416, tree `stack`).
#   fleetq run --host tatooine-r9700,tatooine-r9700-2 --takes-down radiance --expect 40 --mem 60 --label "tp2 batch" -- \
#     env W=batch CAP=2100 XENV="D=batch B=stack" scripts/tp2-hip/window.sh
# Budget ~35 min (CAP=2100), model loads dominate (~1-1.5 min each); stages in priority order, so a CAP cut loses the least
# important last. Every check logs a gate line, SAME / DIFF (lib.sh `same`) or an exit code in steps.log.
#  1. tp=2 batched == solo, bitwise (~8 min): `batch_cuda_real` real_35b_batched_decode_is_bitwise_the_solo_step with
#     INTARSIA_BATCH_TEST_TP=2 (vocab-parallel head, fusion on, graphs and row tables on: the serve configuration): every
#     N = 2..16, 130 consecutive batched steps on growing states at mixed context lengths (5..400 tokens), each row's f32
#     LOGITS bit for bit and its on-device argmax token for token against that sequence's solo tp=2 step, sampled rows too,
#     plus decode rows beside prefill chunks. This is the logits comparison of the brief at a stricter budget than the accuracy
#     floor (`--self-max-kl 1e-9 --self-max-pos-kl 1e-6 --self-max-flip-pct 0 --self-max-tail 5e-8`): those budgets are met
#     by any bitwise-equal pair, and `intarsia accuracy` itself cannot drive batched decode (`--max-batch` is serve-only), so
#     the floor-budget dump comparison would only re-test the solo path; stage 6 checks that path is unchanged instead.
#  2. sensitivity (~3 min): real_35b_tp2_batch_mutants_are_caught: the same gate must FAIL for a peer whose rows run in
#     reverse order (its row table), whose row 0 is one position behind (its batch_pos), or which drops the row-exact mark
#     (ordinary multi-row kernels). The row-count and dropped-collective mutants are cpu-only (tests/tp_batch.rs): the P2P
#     communicator would wait on them.
#  3. serve-level identity (~7 min): `intarsia serve --max-batch 8 --tp 2 --backend hip`, then tools/cbidentity.py gate: a solo
#     reference run of 40 cases (stratified sample) on the same server, then the cases concurrently at B = 4 and 8 (mixed
#     streaming), every request's token ids compared with its solo run. The serve log must show the peer's batched graph
#     replays (`tensor-parallel rank 1: batched decode graph ... replays=`, printed when the model drops: may be missing after
#     `docker stop`, so VACUOUS-GRAPHS there is a weak signal; stage 1 ASSERTS both ranks replayed in lockstep).
#  4. tp=1 unchanged (~6 min): GPU 0 batch_recurrent on HIP (tiny hybrid, N 2..16 x 130 steps, engine, prefix
#     reuse), GPU 1 the HIP crate's rows_exact (with the new --tp 2 shard shapes: K/2 projections, k/v per rank, the shared
#     expert per rank, the V/2 head, MoE at half the routed intermediate).
#  5. speed (~6 min, not localeval): real_35b_batch_step_timing at tp=1 (GPU 0) and tp=2, B = 1 (solo), 4, 8, 16: per-step
#     wall time (TIMING lines; aggregate tok/s = B * 1000 / ms).
#  6. solo tp=2 unchanged (~3 min): 256 greedy ids D vs B at tp=2 (the path every batched row must equal).
#  7. tp=2 batched engine (~6 min): real_35b_batched_engine_equals_the_local_engine at tp=2 (one stream = LocalEngine, 20
#     requests on 4 and 16 slots = their solo runs).
#  8. if time is left: stage 1 with INTARSIA_TP_VOCAB=0 (the replicated head): batched == solo there too, so with stage 1 and
#     PR #135's split == replicated (solo), every batched split row is the replicated head's bits.
#
# Aggregate decode tok/s at 1 / 4 / 16 concurrent streams, --gen 512, tp=1 vs tp=2 (localeval takes its own fleetq slot, so NOT
# inline here: it would queue behind this window's job). With a server up on tatooine in a window of its own:
#   tp=2: deploy/tatooine/dev.sh /work/tp2-intarsia serve --backend hip --model /models/Qwen3.6-35B-A3B-NVFP4 --tp 2 --devices 0,1 \
#           --max-batch 16 --profile-every 0 --mem-budget 29 --max-seq 8192 --no-supervisor --host 0.0.0.0 --port 8090 \
#           --served-model-name qwen36-35b-tp2b
#   tp=1: the same with HIP_VISIBLE_DEVICES=0 and no --tp / --devices (port 8091, --served-model-name qwen36-35b-tp1b)
#   cd /opt/code/deepeval && uv run localeval speed --url http://tatooine:8090/v1 --model qwen36-35b-tp2b --label tp2-batch16 \
#     --sizes 128,1024 --gen 512 --reps 3 --concurrency 4,16 --recipe "<the serve command above>"
#   cd /opt/code/deepeval && uv run localeval speed --url http://tatooine:8091/v1 --model qwen36-35b-tp1b --label tp1-batch16 \
#     --sizes 128,1024 --gen 512 --reps 3 --concurrency 4,16 --recipe "<the serve command above>"
#   (the single-stream row of each run is the 1-stream number; target: tp=2 clearly above tp=1 at 4 and 16 streams)
set -uo pipefail
export O=${1:?out dir}
source "$HOME/tp2/lib.sh"
D=${D:-batch}; B=${B:-stack}
IMG=${INTARSIA_IMAGE:-intarsia-hip-dev:rocm7.2.4-rust1.90.0}
for t in $D $B; do echo "$t: $(cat "$(T $t)/tp2-intarsia.stamp" 2>/dev/null || ls -la "$(T $t)/tp2-intarsia")"; done > "$O/versions.txt"

# mtest <tag> <gpus> <timeout s> <cargo test args...>: intarsia-models GPU tests, prebuilt in the -hiptest volume (no rebuild)
mtest() {
  local tag=$1 gpus=$2 to=$3; shift 3
  # shellcheck disable=SC2086
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=$gpus timeout "$to" \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 ${E:-} \
    cargo test --release -p intarsia-models "$@" > "$O/mtest-$tag.log" 2>&1 < /dev/null
  local rc=$?
  log "mtest $tag exit $rc${E:+ [$E]}: $(grep -aE '^test result|FAILED|panicked|SKIPPED|pairs bitwise|mutant .* caught|TIMING|tensor-parallel rank 1: batched' "$O/mtest-$tag.log" | head -12 | cut -c1-160 | tr '\n' '|')"
  return $rc
}
# hiptest <tag> <gpus> <cargo test args...>: the HIP crate's GPU tests (same volume)
hiptest() {
  local tag=$1 gpus=$2; shift 2
  INTARSIA_REPO=$(T $D) INTARSIA_TARGET_VOL=intarsia-cargo-target-tp2-$D-hiptest HIP_VISIBLE_DEVICES=$gpus timeout 600 \
    "$(T $D)/deploy/tatooine/dev.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip "$@" -- --test-threads=1 --nocapture \
    > "$O/hiptest-$tag.log" 2>&1 < /dev/null
  log "hiptest $tag exit $?: $(grep -aE '^test result|FAILED|panicked|SKIPPED' "$O/hiptest-$tag.log" | head -6 | tr '\n' '|')"
}
TP2ENV="INTARSIA_BATCH_TEST_TP=2"

# 1. tp=2 batched == solo (logits bit for bit, ids), N = 2..16
E="$TP2ENV" mtest tp2-decode 0,1 900 --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step

# 2. the gate's sensitivity to the step protocol's mutants
E="$TP2ENV" mtest tp2-mutants 0,1 420 --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_tp2_batch_mutants_are_caught

# 3. serve --max-batch 8 --tp 2 + cbidentity at B = 4 and 8
CID=tp2-batch-serve
docker rm -f $CID > /dev/null 2>&1
docker run -d --name $CID --network host \
  --device /dev/kfd --device /dev/dri --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --group-add video \
  --ipc host --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e GPU_MAX_HW_QUEUES=1 \
  -e INTARSIA_HOSTNAME="$(hostname)" -v "$(T $D)":/work -v /data/models/intarsia:/models:ro -w /work "$IMG" \
  /work/tp2-intarsia serve --backend hip --model $M35 $TP2 --max-batch 8 --profile-every 0 --mem-budget 29 --no-supervisor \
  --host 0.0.0.0 --port 8090 --served-model-name qwen36-35b-tp2b --max-seq 8192 > "$O/serve.cid" 2> "$O/serve-start.err" < /dev/null
up=0
for i in $(seq 1 60); do
  if curl -s -m 3 http://127.0.0.1:8090/v1/models < /dev/null | grep -q '"data":\['; then up=1; log "serve tp2 --max-batch 8 up after $((i * 5)) s"; break; fi
  sleep 5
done
if [[ $up == 1 ]]; then
  timeout 900 python3 "$(T $D)/tools/cbidentity.py" gate --url http://127.0.0.1:8090 --b 4,8 --streams mixed --sample 40 \
    --outdir "$O/cbid" > "$O/cbid.log" 2>&1 < /dev/null
  log "cbidentity tp2 B=4,8 exit $?: $(grep -aiE 'pass|fail|mismatch|identical' "$O/cbid.log" | tail -4 | cut -c1-200 | tr '\n' '|')"
else
  log "serve tp2 --max-batch 8 DID NOT come up (see serve.log)"
fi
docker stop -t 30 $CID > /dev/null 2>&1
docker logs $CID > "$O/serve.log" 2>&1
docker rm -f $CID > /dev/null 2>&1
if grep -aqE 'tensor-parallel rank 1: batched decode graph captures=[1-9][0-9]* replays=[1-9]' "$O/serve.log"; then
  log "serve: peer batched graphs replayed: $(grep -aE 'tensor-parallel rank 1: batched decode graph' "$O/serve.log" | tail -1)"
else
  log "VACUOUS-GRAPHS: no peer batched graph replays in serve.log (eager batched steps, or the line was not printed at shutdown)"
fi
grep -aiE '\bnan\b|non-finite|panic|IK_ERR|hip error|illegal|fatal|diverged' "$O/serve.log" > "$O/serve-suspect.txt"
log "serve log suspect lines: $(wc -l < "$O/serve-suspect.txt")"

# 4. tp=1 unchanged: the tiny hybrid batching suite on GPU 0, the row-exact kernels incl. the shard shapes on GPU 1
# (one after the other: two cargo processes on one target volume would serialise on its lock anyway)
mtest tp1-batch-recurrent 0 900 --test batch_recurrent -- --test-threads=1
hiptest rows-exact 1 --test rows_exact

# 5. speed: per-step wall time at tp=1 and tp=2 (B = 1 is the solo step line)
E="INTARSIA_TIMING_ROWS=4,8,16" mtest timing-tp1 0 420 --test batch_cuda_real -- --test-threads=1 --nocapture --ignored --exact real_35b_batch_step_timing
E="INTARSIA_TIMING_ROWS=4,8,16 $TP2ENV" mtest timing-tp2 0,1 420 --test batch_cuda_real -- --test-threads=1 --nocapture --ignored --exact real_35b_batch_step_timing
grep -ah '^TIMING' "$O/mtest-timing-tp1.log" | sed 's/^/tp1 /' > "$O/timing.txt"
grep -ah '^TIMING' "$O/mtest-timing-tp2.log" | sed 's/^/tp2 /' >> "$O/timing.txt"

# 6. the solo tp=2 path is unchanged: 256 greedy ids, this branch vs its base
gen $D 0,1 tp2-d-35b-short $M35 short $TP2
gen $B 0,1 tp2-b-35b-short $M35 short $TP2
same gen-tp2-d-35b-short.ids gen-tp2-b-35b-short.ids "35B tp=2 solo 256 greedy ids, batch branch vs stack base"

# 7. the batched engine at tp=2 against the local engine
E="$TP2ENV" mtest tp2-engine 0,1 900 --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_engine_equals_the_local_engine

# 8. the replicated head (if the CAP leaves time)
E="$TP2ENV INTARSIA_TP_VOCAB=0" mtest tp2-decode-vocab0 0,1 900 --test batch_cuda_real -- --test-threads=1 --nocapture --exact real_35b_batched_decode_is_bitwise_the_solo_step
log "stages-batch done"
