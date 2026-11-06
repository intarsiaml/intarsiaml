#!/bin/bash
# Lane dsv41/replay: tests of the bounded-replay prefill (layers 20-39 run only on the last 128 prompt rows, layers 0-19 and
# layer 20's kv-source update on all rows; on by default where supported, INTARSIA_DSV41_REPLAY=0 off). It changes numerics, so eight steps check
# the path and then measure speed and quality. Each stops the run at the first failure; each step's PASS criterion is stated
# here and enforced by an exit status (replay-test-lib.py, whose criteria replay-test-lib-selftest.py tests on synthetic data):
#   1  off vs on, prompts of at most 128 tokens (no cut): greedy ids identical (also: every prompt really is <= 128 ids)
#   2  golden ids (tools/prompts/golden*.jsonl, 10 x 64): this branch with replay off vs the binary of the merged base MAIN_REF (default c39dcaeb3 = origin/main before the default turned on; the branch's default path must equal it):
#      640 of 640 identical
#   3  GATE, encoder half bit-identical: off and on serve the same request sequence (replay-prompts-gate.jsonl: an encoder-only
#      chunk and a crossing chunk, a crossing chunk then a whole chunk under the floor, a resumed prefill) with the position
#      dump (INTARSIA_DSV41_DUMP) on layer 0 / 9 / 19 outputs and layer 20's kv-source steps: all dumps byte-identical on both
#      ranks; and layer 39 absent below the floors, present above, different at the first replayed rows (replay really on)
#   4  ABAB prefill timing (off, on, off, on) at 4K and 32K tokens, chunk 512 and 2048, localeval speed: no errors, on/off prefill
#      time at most 1.05 in every cell (a replay that is slower than off is broken)
#   5  agreement: (a) on vs TensorFold's ids on the golden prompts, where the replay is active (prompts of more than 128 ids): not
#      worse than off by more than 8 of 64 tokens per prompt, prompts of <= 128 ids identical off vs on; (b) on vs off over 160
#      decode tokens of three long prompts (the decode window ring): same lengths, at least 30 % equal positions (a tripwire
#      against garbage, nothing finer)
#   6  paired localeval quality off vs on on the frozen chunk32k-2 set (150 items at 31K + 30 at 64K): at least 90 % of the items
#      scored in both arms; on not worse than off by 0.10 or more, nor by more than 0.05 with McNemar p < 0.05. No oracle exists
#      (off and on differ by design), the items depend on compressed KV rather than the recent window, so this step detects a
#      large loss only; see docs/results/2026-11-03-dsv41-replay.md
#   7  paired localeval quality off vs on on the recent-window set (scripts/dsv41-tp2/replay-recent-gen.py: one planted code per
#      prompt at a measured token distance from the prompt end, 6 distance buckets x 4K and 30K prompts x 12 items, exact scoring,
#      no judge). The reading rule (which off-vs-on loss in which bucket is a DEFECT) is fixed in the doc, "Step 7", before any
#      data. The step FAILS when the measurement is invalid (a dataset sha256 that differs, an empty reply or call error, a server
#      prompt_tokens that differs from the manifest, a prefix-cache hit, an item missing in an arm); a DEFECT per the rule is its
#      reported status (the later steps still run, the job exits 1).
#   8  the default: both ranks with the variable unset vs INTARSIA_DSV41_REPLAY=1, golden ids at chunk 2048: 640 of 640 identical,
#      and both ranks log that the replay is on
# Run it by name; it queues itself as ONE fleetq job over both Sparks (rank 0 spark1 `serve --tp-remote`, rank 1 spark2
# `tp-worker`, the script starts both over ssh, as run-full.sh does; so no `--each`), takes the two vLLM containers down and
# ALWAYS brings them back (EXIT trap). The Sparks are shared: queue it, do not run its inner part by hand.
#   scripts/dsv41-tp2/replay-test.sh [--dry-run] [--steps 1,3]      (steps 1 to 8, run in the order given; default 1,2,3,4,5,6,7,8)
# Knobs (environment): SPARK_DIR (default ~/intarsia-dsv41-replay on the Sparks), NO_BUILD=1 (binaries already in place),
#   CAP (66000), Q_SHARDS (frozen shards to run, default all 9 = 180 items; e.g. "s32-00 s65-00" for a pilot), Q_CHUNK (512),
#   Q_TOKENS (400, max_tokens of the quality replies), S3_CHUNKS ("512 2048"), S3_SIZES (4096,32768), S3_REPS (2), KILL_GIB (5),
#   GATE_CHUNKS ("512 2048"; step 3), ALIVE_TRIES (4 ssh tries before a host counts as unreachable; an unreachable host never stops the ranks),
#   EARLYOOM_OK ("3,2 2,1"), XENV (extra engine env, both ranks),
#   RAW (results folder), TF_IDS (TensorFold token file for step 5), MAIN_REF (the baseline commit for step 2), MAIN_WT (checkout of it, built for step 2),
#   Q2RAW / QDIR (manifest + sha256 and the shards of the frozen chunk32k-2 set, step 6; the shards are gitignored: regenerate with
#   the command step 6 prints when they are missing),
#   STOP_EPOCH (unix time: no step starts after it, step 6 starts no shard within S6_MARGIN_MIN (120) minutes of it, step 7 starts no
#   unit whose on arm would start after it; RESTORE_BY_EPOCH (default STOP_EPOCH + 90 min) is when vLLM must be back: every step's
#   deadline is capped at RESTORE_BY_EPOCH - RESTORE_MIN),
#   step 7: S7_PLAN (units CHUNK:LENGTHS, default "2048:4k,32k"; e.g. "2048:4k,32k 512:4k"), S7_K4 / S7_K32 (items per bucket used of
#   each length, at most 12; the files are ordered round robin over the buckets), S7_TOKENS (40, max_tokens of the replies), RDIR
#   (the generated item files, default .scratch/recent-window/data), R_MAN / R_SHA (the committed manifest and sha256s),
#   CONT_AFTER (steps whose failure does not end the run, e.g. "7"; the job still exits 1).
# Inner mode (REPLAY_TEST_INNER=1 plus FLEETQ_JOB) is what the fleetq job runs; set both only when you hold a slot yourself
# (`fleetq acquire --host spark1,spark2 --takes-down spark1-vllm,spark2-vllm ...`).
set -u -o pipefail
S="$(cd "$(dirname "$0")" && pwd)"; SELF="$S/replay-test.sh"; REPO="$(cd "$S/../.." && pwd)"

# ---- the vLLM restore, the way run-full.sh does it: `docker start` the guidance container on both Sparks, wait for a
# "Application startup complete" line newer than StartedAt, then `fleetq mark-up`. Container name: one place.
VLLM_C=missioncntrl-guidance
VLLM_ROLES=spark1-vllm,spark2-vllm

# ---- configuration
SPARK_DIR_IN=${SPARK_DIR:-intarsia-dsv41-replay}; SPARK_DIR=$SPARK_DIR_IN
source "$S/spark-dir.sh"
[[ $SPARK_REL == i* ]] || { echo "SPARK_DIR '$SPARK_DIR_IN' must start with the letter i (pkill patterns use [i]...)"; exit 2; }
BIN_BR="$SPARK_DIR/bin/intarsia"; BIN_MAIN="$SPARK_DIR/bin/intarsia.main"; LOGD="$SPARK_DIR/logs"
C0='$HOME/intarsia-dsv41-tp2/cache/r0of2'; C1='$HOME/intarsia-dsv41-tp2/cache/r1of2'
WIP=10.100.152.3; WPORT=29701; SPORT=8899; SURL=http://100.120.88.104:$SPORT
# 8 channels (was 2): docs/results/2026-11-05-dsv41-nccl.md, A (+3 % prefill at 1K/4K/16K, ids identical); NCCL_MAX_NCHANNELS=2 restores the old setting
NCCLENV="INTARSIA_NCCL_LIB=\$HOME/intarsia-nccl/libnccl.so.2 NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 NCCL_IB_GID_INDEX=3 NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8} NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-1048576}"
CAP=${CAP:-66000}                      # 32K prompts at step 3 and 64K items at step 5 (chunk32k-2 used 66000)
TOK=${TOKENIZER:-/opt/code/intarsiaml/.worktrees/dsv41-official/mia/tokenizer.json}
MAIN_REF=${MAIN_REF:-c39dcaeb3}; MAIN_WT=${MAIN_WT:-$(dirname "$REPO")/dsv41-replay-main}
RESDIR=${RESDIR:-$REPO/docs/results/2026-11-03-dsv41-replay}; RAW=${RAW:-$RESDIR/raw}
# TensorFold's greedy ids for the golden prompt set (step 5). The record docs/results/2026-11-01-dsv41-tensorfold-token-agreement.txt
# holds only the summary table and snippets, not the ids; the ids it was computed from are this file (docs/results/2026-11-01-dsv41-tp2.md).
# They come from TensorFold's served lane (run-golden.sh: start.sh, 2048-row chunks, replay = prompt length - window always on) with
# TF_DS_REPLAY_FLOOR unset (nothing sets it; model.zig:151 defaults it off). That flag only matters for chunks of at most 16 rows
# (prompt.zig ring_extra) whose window reaches below the floor; the prompts here are single chunks of 300 and 700 rows (gather path,
# floor honoured either way), the others are at most 99 ids (no cut), and decode rows never reach below the floor. So the ids are
# the masked-floor semantics ours implements, and the comparison is meaningful for long_prose_300 / long_prose_700 only.
TF_IDS=${TF_IDS:-/opt/code/intarsiaml/.worktrees/dsv41-golden-tf/tok-tensorfold-mia.st}
# frozen question set of dsv41/chunk32k-2 (shards are gitignored there: referenced by path, checked by sha256)
C32=${C32:-/opt/code/intarsiaml/.worktrees/chunk32k-2}
Q2RAW=${Q2RAW:-$C32/docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw}; QDIR=${QDIR:-$C32/.scratch}
Q_SHARDS=${Q_SHARDS:-s32-00 s65-00 s32-01 s65-01 s32-02 s65-02 s32-03 s32-04 s32-05}
Q_CHUNK=${Q_CHUNK:-512}; Q_TOKENS=${Q_TOKENS:-400}
S3_CHUNKS=${S3_CHUNKS:-512 2048}; S3_SIZES=${S3_SIZES:-4096,32768}; S3_REPS=${S3_REPS:-2}
GATE_CHUNKS=${GATE_CHUNKS:-512 2048}
# step 7: the recent-window item set (generated by replay-recent-gen.py, gitignored; manifest and sha256s are committed)
RDIR=${RDIR:-$REPO/.scratch/recent-window/data}; R_MAN=${R_MAN:-$S/replay-recent.manifest.json}; R_SHA=${R_SHA:-$S/replay-recent.sha256.json}
S7_PLAN=${S7_PLAN:-2048:4k,32k}; S7_TOKENS=${S7_TOKENS:-40}; S7_K4=${S7_K4:-12}; S7_K32=${S7_K32:-12}; S7_START_S=${S7_START_S:-90}     # a rank restart took 58-60 s from kill to kill (23-30 s of it to /health) in fleetq #8415 / #8423
S6_MARGIN_MIN=${S6_MARGIN_MIN:-120}
# step 3 dump selection: positions that fall in the encoder-only, crossing and whole chunks of the gate prompts (floors 1072 for
# the 1200-token prompt, 1372 for the 1500-token one, 484 for the 612-token one), and the steps compared. A name filter is a
# substring match: layers.0. / layers.9. / layers.19. do not match each other; layer 20's rope / act_quant of the kv source are
# the encoder part's last steps (ratio 1: one compressed row per input row, so the hook selects them); layer 39 shows the decoder half.
GATE_POS=5,300,483,484,511,512,600,700,1000,1071,1072,1100,1199,1200,1300,1371,1372,1373,1450,1499
GATE_WHAT='layers.0.hc_post(ffn),layers.9.hc_post(ffn),layers.19.hc_post(ffn),layers.20.attn.rope(index_k),layers.20.attn.rope(latent),layers.20.attn.act_quant(index_k),layers.20.attn.act_quant(latent),layers.39.hc_post(ffn)'
GATE_ARGS=(--min-files 100 --absent 300,700,1000 --present 1100,1199,1450,1499 --differ 1372,1373)

# ---- the plan: minutes per piece (guesses, high on purpose; measured: load 31 s, 32K item ~54 s, shard of 25 items 22 min)
BUILD_MAIN_MIN=30; BUILD_BR_MIN=25; DOWN_MIN=3; RESTORE_MIN=12
[[ -z ${STOP_EPOCH:-} ]] || RESTORE_BY=${RESTORE_BY_EPOCH:-$((STOP_EPOCH + 90 * 60))}   # when vLLM must be back (default: 90 min after STOP_EPOCH)
# step 7 cost, deciseconds per item (a reply of at most S7_TOKENS tokens): the prefill time (TTFT) of the chunk32k-2 quality run at
# 31K prompts (median 40.2 s at chunk 2048, 52.8 s at 512; scaled to the 30.4K mean here) for the off arm, 0.50 of it for the on arm
# (step 4: 16.7 s / 33.4 s at 2048, 19.9 / 40.4 at 512), the 4K figures from step 4 (4.0 / 2.2 s at 2048, 4.8 / 2.6 at 512); +1 s of decode
s7_ds() { # CHUNK LENGTH ARM
  case $1:$2:$3 in
    2048:4k:off) echo 50 ;; 2048:4k:on) echo 32 ;; 2048:32k:off) echo 410 ;; 2048:32k:on) echo 210 ;;
    512:4k:off) echo 58 ;; 512:4k:on) echo 36 ;; 512:32k:off) echo 530 ;; 512:32k:on) echo 270 ;;
    *) echo 600 ;;    # a chunk size nobody measured: slower than every figure above
  esac
}
s7_items() { local k=$S7_K4; [[ $1 == 32k ]] && k=$S7_K32; echo $((6 * k)); }
s7_arm_s() { # CHUNK LENGTHS ARM: seconds of one arm = rank start + the items of each length
  local t=$S7_START_S ln
  for ln in ${2//,/ }; do t=$((t + $(s7_items "$ln") * $(s7_ds "$1" "$ln" "$3") / 10)); done
  echo $t
}
s7_unit_s() { echo $(( $(s7_arm_s "${1%%:*}" "${1#*:}" off) + $(s7_arm_s "${1%%:*}" "${1#*:}" on) )); }
est_shard() { case $1 in s65-*) echo 52 ;; *) echo 48 ;; esac; }   # off + on arm, two rank starts
est() {
  local n=$1 t=0 sh c
  case $n in
    1) echo 8 ;;
    2) echo 10 ;;
    3) c=0; for _ in $GATE_CHUNKS; do c=$((c + 1)); done; echo $((c * 2 * 8 + 2)) ;;   # a chunk: 2 starts x (start 5 + 3 prompts 1 + copy 1 + slack 1)
    4) c=0; for _ in $S3_CHUNKS; do c=$((c + 1)); done; echo $((c * 4 * 6)) ;;     # 4 configs a chunk x (start 2 + 4K 0.5 + 32K 3 + slack)
    5) echo 30 ;;                                                                  # 2 starts x (start 5 + 10 golden x 64 tokens 5 + 3 long decodes 4 + slack)
    6) for sh in $Q_SHARDS; do t=$((t + $(est_shard "$sh"))); done; echo $((t + 3)) ;;
    7) for sh in $S7_PLAN; do t=$((t + $(s7_unit_s "$sh"))); done; echo $(( (t + 59) / 60 + 1 )) ;;       # + the sha check and the analysis
    8) echo 12 ;;                                                                  # 2 starts x (start 2 + 10 golden x 64 tokens 3 + slack 1)
  esac
}
name_of() {
  case $1 in
    1) echo "off vs on, prompts <= 128 tokens: identical ids" ;;
    2) echo "golden ids, off vs main binary $MAIN_REF: 640/640" ;;
    3) echo "GATE: encoder half (layers 0-19, layer 20 kv source) bit-identical off vs on; decoder half skipped below the floor" ;;
    4) echo "ABAB prefill timing 4K/32K, chunk 512/2048 (localeval speed), on/off <= 1.05" ;;
    5) echo "agreement: on vs TensorFold ids (golden prompts) and on vs off over 160 decode tokens" ;;
    6) echo "paired localeval quality off vs on (frozen chunk32k-2 set): no large loss" ;;
    7) echo "recent-window items (planted code at a measured distance from the prompt end), off vs on, per distance bucket: reading rule" ;;
    8) echo "the default: variable unset vs =1, golden ids 640/640 at chunk 2048, both ranks log replay on" ;;
  esac
}
fmt() { printf '%dh%02dm' $(($1 / 60)) $(($1 % 60)); }

describe() {
  case $1 in
    1) cat <<EOF
     for mode in off on: restart both ranks (branch binary, chunk 512, INTARSIA_DSV41_REPLAY=1 only for on, on BOTH ranks)
       python tools/dsv41/golden_tf.py tokens --base-url $SURL/v1 --model dsv41-intarsia --no-draft-field --tokenizer \$TOK \\
         --prompts tools/prompts/golden.jsonl --long-prompts 127,128 --long-continue 64 --out raw/step1/tok-\$mode.st
     replay-test-lib.py tokcmp tok-off.st tok-on.st --require-identical --max-prompt 128      (every prompt <= 128 ids, ids equal)
EOF
    ;;
    2) cat <<EOF
     build origin/main $MAIN_REF in \$MAIN_WT (git worktree, $MAIN_WT) with its scripts/dsv41-tp2/build.sh, BIN_NAME=intarsia.main
     restart ranks with $BIN_MAIN, then with $BIN_BR (replay unset); each:
       python tools/dsv41/golden_tf.py tokens ... --tokenizer \$TOK --out raw/step2/tok-{main,off}.st      (default prompts: 10 x 64)
     replay-test-lib.py tokcmp tok-main.st tok-off.st --require-total 640                       (640 of 640 tokens identical)
EOF
    ;;
    3) cat <<EOF
     PASS when: every encoder-half dump (layers 0 / 9 / 19 outputs, layer 20 rope + act_quant of the kv source) is byte-identical off vs on
       on both ranks (>= 100 dumps), the on run has no layer-39 dump at positions below every floor (300 700 1000), both runs have it
       above (1100 1199 1450 1499), and layer 39 differs off vs on at the first replayed rows (1372 1373).
     for chunk in $GATE_CHUNKS; for mode in off on: restart both ranks (INTARSIA_DSV41_REPLAY=1 for on) with
       INTARSIA_DSV41_DUMP=\$SPARK_DIR/dump/g\$chunk-\$mode INTARSIA_DSV41_DUMP_POS=$GATE_POS INTARSIA_DSV41_DUMP_WHAT=<8 step names>
       python tools/dsv41/golden_tf.py tokens ... --prompts scripts/dsv41-tp2/replay-prompts-gate.jsonl     (in this order, one sequence:
         g_A_1200 cold: chunk 512 = encoder, encoder, crossing; g_C_1500 extends it: resumed prefill, crossing; g_B_612 (code corpus):
         chunk 512 = crossing then a whole chunk under the floor)
       copy the dump folders of both ranks back; replay-test-lib.py gatecmp off on ${GATE_ARGS[*]}
EOF
    ;;
    4) cat <<EOF
     PASS when: no localeval error, and the mean on/off prefill time (TTFT median of the two passes) is at most 1.05 for every chunk and size.
     for chunk in $S3_CHUNKS; for pass in 1 2; for mode in off on   (= off, on, off, on):  restart ranks, INTARSIA_DSV41_CHUNK=\$chunk
       cd /opt/code/deepeval && uv run localeval speed --sizes $S3_SIZES --gen 8 --reps $S3_REPS --label dsv41replay-speed-c\$chunk-\$mode\$pass \\
         --url $SURL/v1 --model dsv41-intarsia --thinking off --recipe "<stamp, chunk, mode>"     (inside our fleetq slot: FLEETQ_JOB is set)
     replay-test-lib.py speedtab raw/step4/speed-table.md --max-ratio 1.05 --expect-cells <4 x chunks> <cells>
EOF
    ;;
    5) cat <<EOF
     PASS when: (a) on vs TensorFold (on the prompts of more than 128 ids, i.e. long_prose_300 / _700): on's equal positions are not below
       off's minus 8 of 64, prompts of <= 128 ids identical off vs on (TF ids: $TF_IDS; sha256 logged; if missing (a) prints SKIPPED);
       (b) on vs off over 160 decode tokens of d_1200, d_4106 (extends d_1200: resumed prefill), d_612: same lengths, >= 30 % equal positions.
     for mode in off on: restart ranks (chunk 512); golden_tf.py tokens --prompts golden.jsonl golden_long.jsonl replay-prompts-decode.jsonl
     replay-test-lib.py tfagree TF.st tok-off.st tok-on.st ; replay-test-lib.py tokcmp tok-off.st tok-on.st --prefix d_ --min-equal-pct 30 --same-lengths
EOF
    ;;
    6) cat <<EOF
     PASS when: >= 90 % of the items scored in both arms, on not worse than off by 0.10 or more, nor by > 0.05 with McNemar exact p < 0.05.
     shards ($Q_SHARDS) of $QDIR/final.<shard>.jsonl, sha256-checked against $Q2RAW/items.sha256.json
     for shard: for arm in off on: restart ranks (chunk $Q_CHUNK), then
       cd /opt/code/deepeval && uv run localeval quality --dataset final.<shard>.jsonl --judge none --max-tokens $Q_TOKENS \\
         --label dsv41replay-<arm>-<shard> --url $SURL/v1 --model dsv41-intarsia --thinking off --recipe "..."
     replay-test-lib.py quality $Q2RAW/items.manifest.json raw/step6/pairs.txt raw/step6/quality.md --expect-items <90 % of the shard items>
       (accuracy overall and by earliest / latest fact depth and fact count, discordant counts, McNemar exact p)
EOF
    ;;
    7) cat <<EOF
     PASS when the measurement is valid and the reading rule (docs/results/2026-11-03-dsv41-replay.md, "Step 7") finds no DEFECT; FAIL when invalid.
     items: $RDIR/recent.{4k,32k}.jsonl (12 per distance bucket per length; ${S7_K4} / ${S7_K32} used), manifest $R_MAN, both sha256-checked
       against $R_SHA before any arm starts; localeval rows are {name, input, expected_output}, scored by the last code in the reply
     units (S7_PLAN='$S7_PLAN'), each: for arm in off on: restart ranks (chunk of the unit), then per length
       cd /opt/code/deepeval && uv run localeval quality --dataset <file> --judge none --max-tokens $S7_TOKENS --label dsv41replay-s7-c<chunk>-<arm>-<len> \\
         --url $SURL/v1 --model dsv41-intarsia --thinking off --recipe "..."
       replay-test-lib.py recentcheck after every call: an error or empty reply, a prompt_tokens differing from the manifest or a cached_tokens > 0 stops the step
     replay-test-lib.py recent $R_MAN raw/step7/pairs.txt raw/step7/recent.md
       (accuracy off vs on by distance bucket and length, b = off right on wrong, c = on right off wrong, one-sided p; the rule per chunk size)
     a unit starts only if its on arm would start before STOP_EPOCH and the unit plus the vLLM restore ends before RESTORE_BY_EPOCH
EOF
    ;;
    8) cat <<EOF
     for mode in default on: restart both ranks (branch binary, chunk 2048; the variable unset for default, =1 for on)
       the golden ids as in step 2, and grep 'bounded-replay prefill on' in the serve and the worker log
     replay-test-lib.py tokcmp tok-default.st tok-on.st --require-total 640
EOF
    ;;
  esac
}

# ---- arguments
DRY=0; STEPS_ARG="1,2,3,4,5,6,7,8"; PASS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1; shift ;;
    --steps) STEPS_ARG=${2:?--steps needs a list like 1,3}; PASS+=(--steps "$2"); shift 2 ;;
    -h|--help) sed -n 2,49p "$0"; exit 0 ;;
    *) echo "unknown argument '$1' (--dry-run, --steps 1,3)"; exit 2 ;;
  esac
done
SELECTED=${STEPS_ARG//,/ }
for n in $SELECTED; do case $n in [1-8]) ;; *) echo "--steps: '$n' is not a step (1..8)"; exit 2 ;; esac; done
for u in $S7_PLAN; do [[ $u =~ ^[0-9]+:(4k|32k)(,(4k|32k))*$ ]] || { echo "S7_PLAN unit '$u' is not CHUNK:4k[,32k]"; exit 2; }; done
(( S7_K4 >= 1 && S7_K4 <= 12 && S7_K32 >= 1 && S7_K32 <= 12 )) || { echo "S7_K4 / S7_K32 must be 1..12"; exit 2; }
has() { [[ " $SELECTED " == *" $1 "* ]]; }

PLAN_MIN=0; BUILD_MIN=0
if [[ ${NO_BUILD:-0} != 1 ]]; then BUILD_MIN=$BUILD_BR_MIN; has 2 && BUILD_MIN=$((BUILD_MIN + BUILD_MAIN_MIN)); fi
PLAN_MIN=$((BUILD_MIN + DOWN_MIN + RESTORE_MIN))
for n in $SELECTED; do PLAN_MIN=$((PLAN_MIN + $(est "$n"))); done
EXPECT=${EXPECT_MIN:-$(( (PLAN_MIN * 11 + 9) / 10 ))}   # planned + 10 %, whole minutes: the honest high guess for fleetq --expect (EXPECT_MIN overrides)
LABEL="dsv41 replay test, steps ${STEPS_ARG}: bounded-replay prefill off vs on (vLLM restored after)"
# Same fleetq shape as run-full.sh: ONE EXCLUSIVE job over both Sparks, the script starts both ranks itself. Exclusive (not
# `--quiet gpu,cpu`) because the 5 GiB MemAvailable watcher SIGKILLs the ranks: a chunk32k-2 serve log showed 8.5 GiB available
# at chunk 2048, so even a co-running `--load none` job of about 3.5 GiB would trip it. --mem 116 as in run-full.sh.
# the knobs the inner run reads travel IN the command (resolved values), so the printed command is the whole truth and no queue
# runner has to inherit this shell's environment
FWD=()
for v in NO_BUILD CAP Q_SHARDS Q_CHUNK Q_TOKENS S3_CHUNKS S3_SIZES S3_REPS GATE_CHUNKS KILL_GIB EARLYOOM_OK ALIVE_TRIES XENV HOT_MB TP_SPIN_MS \
         STOP_EPOCH RESTORE_BY_EPOCH CONT_AFTER MAIN_REF TF_IDS C32 QDIR Q2RAW RDIR R_MAN R_SHA S7_PLAN S7_TOKENS S7_K4 S7_K32 S7_START_S S6_MARGIN_MIN; do
  [[ -n ${!v+x} ]] && FWD+=("$v=${!v}")
done
FQ=(fleetq run --host spark1,spark2 --exclusive --takes-down "$VLLM_ROLES" --mem spark1=116,spark2=116 --expect "$EXPECT"
    --label "$LABEL" -- env REPLAY_TEST_INNER=1 ${FWD[@]+"${FWD[@]}"} "$SELF" ${PASS[@]+"${PASS[@]}"})

if [[ $DRY == 1 ]]; then
  echo "DRY RUN: nothing is run, no host is contacted."
  echo "binary under test: $BIN_BR (this branch $(git -C "$REPO" rev-parse --short=9 HEAD), replay unset = off; INTARSIA_DSV41_REPLAY=1 = on)"
  echo "results folder:    $RAW"
  echo "SPARK_DIR:         $SPARK_DIR_IN (build.sh syncs the source and builds there)"
  echo
  [[ ${NO_BUILD:-0} == 1 ]] && echo "PREP  build skipped (NO_BUILD=1): binaries must already be in $SPARK_DIR/bin" || {
    echo "PREP  ~${BUILD_MIN} min  build"
    has 2 && echo "     git -C $REPO worktree add --detach $MAIN_WT $MAIN_REF   (if absent), then"
    has 2 && echo "     BIN_NAME=intarsia.main SPARK_DIR=$SPARK_REL $MAIN_WT/scripts/dsv41-tp2/build.sh     (CLI baseline only, beside the real binary)"
    echo "     SPARK_DIR=$SPARK_REL $S/build.sh     (this branch; build.sh rsyncs to spark2, cargo build --release --features cuda, copies to both)"; }
  echo "PREP  ~${DOWN_MIN} min  preflight (earlyoom -m in '${EARLYOOM_OK:-3,2 2,1}', binaries, caches, equal STAMPs, REPLAY string in the binary), docker stop -t 60 $VLLM_C on both"
  for n in $SELECTED; do
    echo "STEP $n  ~$(est "$n") min  $(name_of "$n")"; describe "$n"
  done
  echo "END   ~${RESTORE_MIN} min  docker start $VLLM_C on both, wait for 'Application startup complete', fleetq mark-up $VLLM_ROLES"
  echo
  echo "PLANNED DURATION: ${PLAN_MIN} min ($(fmt "$PLAN_MIN")); fleetq --expect ${EXPECT} (planned + 10 %)"
  has 6 && for sh in $Q_SHARDS; do ls "$QDIR/final.$sh.jsonl" >/dev/null 2>&1 || echo "NOTE: frozen shard $QDIR/final.$sh.jsonl is missing: step 6 would fail"; done
  has 6 && for f in items.sha256.json items.manifest.json; do [[ -f $Q2RAW/$f ]] || echo "NOTE: $Q2RAW/$f is missing: step 6 would fail"; done
  if has 6 && [[ -f $Q2RAW/items.sha256.json ]]; then
    # shellcheck disable=SC2086
    python3 "$S/replay-test-lib.py" checksha "$Q2RAW/items.sha256.json" "$QDIR" final. $Q_SHARDS | sed 's/^/step 6 shard sha256: /'
  fi
  if has 7; then
    for f in "$R_MAN" "$R_SHA"; do [[ -f $f ]] || echo "NOTE: $f is missing: step 7 would fail"; done
    if [[ -f $R_SHA ]]; then python3 "$S/replay-test-lib.py" checksha "$R_SHA" "$RDIR" recent. 4k 32k | sed 's/^/step 7 item files: /'; fi
  fi
  if [[ -n ${STOP_EPOCH:-} ]]; then
    echo
    echo "SCHEDULE CHECK (STOP_EPOCH $STOP_EPOCH = $(TZ=America/New_York date -r "$STOP_EPOCH" '+%a %Y-%m-%d %H:%M %Z'), vLLM back by RESTORE_BY $(TZ=America/New_York date -r "$RESTORE_BY" '+%H:%M %Z'), restore takes up to ${RESTORE_MIN} min):"
    echo "  every step's deadline is capped at $(TZ=America/New_York date -r "$((RESTORE_BY - RESTORE_MIN * 60))" '+%H:%M %Z'); a step that would start at or after STOP_EPOCH is skipped"
    has 5 && echo "  step 5: starts only before STOP_EPOCH (planned $(est 5) min, deadline 2x that, capped as above)"
    has 6 && echo "  step 6: a shard starts only if STOP_EPOCH is at least ${S6_MARGIN_MIN} min away: latest shard start $(TZ=America/New_York date -r "$((STOP_EPOCH - S6_MARGIN_MIN * 60))" '+%H:%M %Z'); a shard is planned $(est_shard s32-00) to $(est_shard s65-00) min, so the last one ends by about $(TZ=America/New_York date -r "$((STOP_EPOCH - S6_MARGIN_MIN * 60 + 52 * 60))" '+%H:%M %Z') (hard stop: the cap above)"
    if has 7; then
      for u in $S7_PLAN; do
        echo "  step 7 unit $u: planned $(( $(s7_unit_s "$u") / 60 )) min (off arm $(( $(s7_arm_s "${u%%:*}" "${u#*:}" off) / 60 )), on arm $(( $(s7_arm_s "${u%%:*}" "${u#*:}" on) / 60 ))); latest start = the earlier of $(TZ=America/New_York date -r "$((STOP_EPOCH - $(s7_arm_s "${u%%:*}" "${u#*:}" off)))" '+%H:%M %Z') (on arm must start before STOP_EPOCH) and $(TZ=America/New_York date -r "$((RESTORE_BY - RESTORE_MIN * 60 - $(s7_unit_s "$u")))" '+%H:%M %Z') (unit + restore done by RESTORE_BY)"
      done
    fi
    echo "  plan, if the job starts at its earliest: ${PLAN_MIN} min; to finish all selected steps before STOP_EPOCH it must start by $(TZ=America/New_York date -r "$((STOP_EPOCH - (PLAN_MIN - RESTORE_MIN) * 60))" '+%a %H:%M %Z'); later starts run the units that still fit"
  fi
  [[ -f $TF_IDS ]] || echo "NOTE: $TF_IDS missing: step 5 (a) would print SKIPPED"
  for f in replay-prompts-gate.jsonl replay-prompts-decode.jsonl; do [[ -f $S/$f ]] || echo "NOTE: $S/$f is missing: steps 3 and 5 would fail"; done
  echo
  echo "fleetq command:"
  printf '  '
  for a in "${FQ[@]}"; do case $a in *[\ \(\)\;\&\|\<\>\$\"\']*) printf "'%s' " "$a" ;; *) printf '%s ' "$a" ;; esac; done
  echo
  exit 0
fi

if [[ ${REPLAY_TEST_INNER:-0} != 1 ]]; then
  echo "planned ${PLAN_MIN} min ($(fmt "$PLAN_MIN")), steps ${STEPS_ARG}; queueing as one fleetq job over spark1,spark2 (--expect ${EXPECT})"
  exec "${FQ[@]}"
fi
: "${FLEETQ_JOB:?inner mode runs inside a fleetq job (fleetq run ... or FLEETQ_JOB=<id> of a slot you hold)}"

# ============================================================ inner part: runs inside the fleetq job
mkdir -p "$RAW"
LOG=$RAW/steps.log
log() { echo "$(date +%T) $*" | tee -a "$LOG"; }
r() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 "$1" "export FLEETQ_JOB=$FLEETQ_JOB; $2"; }
lib() { uv run --no-project --with numpy --with safetensors python "$S/replay-test-lib.py" "$@"; }
FAIL_WHY=""; STEP_NOTE=""
fail() { FAIL_WHY="$*"; return 1; }
RUN0=$(date +%s)
log "replay-test start: steps $SELECTED, planned ${PLAN_MIN} min, fleetq job $FLEETQ_JOB, out $RAW"

# ---- memory watchers and the restore (as in run-full.sh)
STOPPED=0; WATCH=""; BGPID=""
RES_N=(); RES_ST=(); RES_EL=(); RES_PL=()
summary() {
  local i
  log "================ SUMMARY (total elapsed $(( ($(date +%s) - RUN0) / 60 )) min, planned ${PLAN_MIN} min) ================"
  for ((i = 0; i < ${#RES_N[@]}; i++)); do
    log "STEP ${RES_N[$i]}: ${RES_ST[$i]}  elapsed $(( RES_EL[i] / 60 ))m$(( RES_EL[i] % 60 ))s (planned ${RES_PL[$i]} min)  $(name_of "${RES_N[$i]}")"
  done
}
kill_ours() { for h in spark1 spark2; do r $h "pkill -TERM -f '[i]${SPARK_REL:1}/bin/intarsia'; for i in \$(seq 1 20); do pgrep -f '[i]${SPARK_REL:1}/bin/intarsia' >/dev/null || break; sleep 1; done; pkill -KILL -f '[i]${SPARK_REL:1}/bin/intarsia'; true"; done; }
restore() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [[ -z $BGPID ]] || killtree "$BGPID"
  summary
  log "restore (exit $rc)"
  [[ -z $WATCH ]] || kill "$WATCH" 2>/dev/null
  if [[ $STOPPED == 1 ]]; then
    kill_ours
    for h in spark1 spark2; do r $h "pkill -f '[b]in/memfine.sh'; true"; done
    for h in spark1 spark2; do scp -q "$h:$SPARK_REL/logs/$h-${MF:-none}" "$RAW/" 2>/dev/null; done
    # a `docker logs --since` prints nothing on these hosts; wait for a "startup complete" line stamped (UTC) after StartedAt
    for h in spark1 spark2; do r $h "docker start $VLLM_C >/dev/null" || log "WARNING: docker start on $h failed"; done
    for h in spark1 spark2; do
      local ok=0 i
      for i in $(seq 1 120); do
        r $h "s=\$(docker inspect -f '{{.State.StartedAt}}' $VLLM_C); t=\$(docker logs -t --tail 2000 $VLLM_C 2>&1 | grep 'Application startup complete' | tail -1 | cut -d' ' -f1); [[ -n \$t && \$t > \$s ]]" && { ok=1; break; }
        sleep 10
      done
      [[ $ok == 1 ]] && log "vLLM on $h up ($(r $h "docker ps --filter name=$VLLM_C --format {{.Status}}"))" || log "WARNING: vLLM on $h not up after 20 min"
    done
    fleetq mark-up spark1-vllm; fleetq mark-up spark2-vllm
  fi
  log "restored; the main-binary checkout $MAIN_WT stays for reruns (remove: git -C $REPO worktree remove $MAIN_WT)"
  exit $rc
}
trap restore EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

# kill a process and its children (macOS has no setsid)
killtree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; return 0; }
# alive HOST PATTERN: 0 = process there, 1 = the host answered and there is none (a dead rank), 2 = the host could not be asked
# (ssh exit 255 on every one of ALIVE_TRIES tries, 5 s apart: step 5 polls for hours, one network blip must not kill an arm)
ALIVE_TRIES=${ALIVE_TRIES:-4}
alive() {
  local rc i
  for ((i = 1; i <= ALIVE_TRIES; i++)); do
    r "$1" "pgrep -f '$2' >/dev/null"; rc=$?
    [[ $rc != 255 ]] && return $((rc != 0))
    log "ssh to $1 failed (try $i of $ALIVE_TRIES)"; sleep 5
  done
  return 2
}
RANK0_PAT="[i]${SPARK_REL:1}/bin/intarsia[^ ]* serve"; RANK1_PAT="[i]${SPARK_REL:1}/bin/intarsia[^ ]* tp-worker"
# ranks_alive: both ranks run (0), a host answered that its rank is gone (1), or a host could not be asked and no rank is known gone (2).
# 2 says nothing about the ranks: an ssh drop from this Mac once stopped two healthy ranks (#8476, 2026-11-05)
ranks_alive() {
  local a=0 b=0
  alive spark1 "$RANK0_PAT" || a=$?
  alive spark2 "$RANK1_PAT" || b=$?
  (( a == 1 || b == 1 )) && return 1
  (( a == 2 || b == 2 )) && return 2
  return 0
}
# bgrun MAX_SECONDS LOGFILE cmd...: run in the background and poll every 15 s (no single wait is long): stops at MAX_SECONDS, at the
# step deadline, or (WATCH_RANKS=1) when a host says its rank has died; an unreachable host only logs. Output goes to LOGFILE; its tail to the steps log. Returns the command's status.
bgrun() {
  local max=$1 lf=$2 t0 rc ra; shift 2; t0=$(date +%s)
  ( "$@" ) > "$lf" 2>&1 < /dev/null &
  BGPID=$!
  while kill -0 "$BGPID" 2>/dev/null; do
    sleep 15
    kill -0 "$BGPID" 2>/dev/null || break
    if (( $(date +%s) - t0 > max )); then killtree "$BGPID"; log "timeout after ${max}s: $*"; tail -n 15 "$lf" | tee -a "$LOG"; BGPID=""; return 124; fi
    if (( $(date +%s) > STEP_DEADLINE )); then killtree "$BGPID"; log "step deadline passed: $*"; BGPID=""; return 124; fi
    if [[ ${WATCH_RANKS:-0} == 1 ]]; then
      ra=0; ranks_alive || ra=$?
      if (( ra == 1 )); then killtree "$BGPID"; log "a rank died during: $*"; tail -n 15 "$lf" | tee -a "$LOG"; BGPID=""; return 125; fi
      (( ra == 2 )) && log "a host is unreachable, ranks left running, still watching: $*"
    fi
  done
  wait "$BGPID"; rc=$?; BGPID=""
  tail -n "${BG_TAIL:-40}" "$lf" | tee -a "$LOG"
  return $rc
}
STEP_DEADLINE=$(( $(date +%s) + 86400 ))

# ---- build (the repo's own build.sh; it runs inside our slot, vLLM still up)
prep_build() {
  if [[ ${NO_BUILD:-0} == 1 ]]; then log "NO_BUILD=1: using the binaries in $SPARK_DIR/bin"; return 0; fi
  STEP_DEADLINE=$(( $(date +%s) + 2 * 60 * BUILD_MIN ))
  if has 2; then
    if [[ ! -d $MAIN_WT ]]; then git -C "$REPO" worktree add --detach "$MAIN_WT" "$MAIN_REF" 2>&1 | tee -a "$LOG" || return 1; fi
    [[ "$(git -C "$MAIN_WT" rev-parse --short=9 HEAD)" == "$MAIN_REF" ]] || { log "$MAIN_WT is not at $MAIN_REF"; return 1; }
    log "build main $MAIN_REF -> $BIN_MAIN"
    BG_TAIL=8 bgrun $((BUILD_MAIN_MIN * 120)) "$RAW/build-main.log" env BIN_NAME=intarsia.main SPARK_DIR="$SPARK_REL" "$MAIN_WT/scripts/dsv41-tp2/build.sh" || return 1
  fi
  log "build this branch -> $BIN_BR"
  BG_TAIL=8 bgrun $((BUILD_BR_MIN * 120)) "$RAW/build-branch.log" env SPARK_DIR="$SPARK_REL" "$S/build.sh" || return 1
  grep -a "stamp:" "$RAW/build-branch.log" | tail -1 | grep -q "dirty=yes" && log "WARNING: the branch binary was built from a dirty tree (uncommitted changes in crates/ or kernels/)"
  return 0
}

preflight() {
  local h eo
  for h in spark1 spark2; do
    # EARLYOOM_OK: the earlyoom -m settings the user has set on purpose; never changed here, our watcher stops us at KILL_GIB above either kill line
    eo=$(r $h 'cat /proc/$(pgrep -x earlyoom)/cmdline | tr "\0" " "' | grep -oE -- '-m [0-9]+,[0-9]+' | cut -c4-)
    [[ " ${EARLYOOM_OK:-3,2 2,1} " == *" $eo "* ]] || { log "earlyoom on $h is at -m '$eo', not one of '${EARLYOOM_OK:-3,2 2,1}'; not starting"; return 1; }
    log "earlyoom on $h: -m $eo"
    r $h "test -x $BIN_BR && test -f ~/intarsia-nccl/libnccl.so.2" || { log "binary $BIN_BR or the NCCL library missing on $h"; return 1; }
    has 2 && { r $h "test -x $BIN_MAIN" || { log "baseline $BIN_MAIN missing on $h"; return 1; }; }
    r $h "grep -qa INTARSIA_DSV41_REPLAY $BIN_BR" || { log "$BIN_BR on $h does not contain INTARSIA_DSV41_REPLAY: built from a tree without the replay feature"; return 1; }
  done
  r spark1 "test -f $C0/manifest.json" && r spark2 "test -f $C1/manifest.json" || { log "rank cache missing"; return 1; }
  # the replay (cut at layer 20, decoder half = layers 20-39) is refused on a truncated or layer-range model: the caches must be whole
  local mchk="import json,os,sys; m=json.load(open(os.path.expandvars(sys.argv[1]))); sys.exit(0 if m.get('layers') is None and not m.get('first_layer') and m.get('world') == 2 else 1)"
  r spark1 "python3 -c \"$mchk\" $C0/manifest.json" && r spark2 "python3 -c \"$mchk\" $C1/manifest.json" || { log "a rank cache manifest has layers / first_layer set or world != 2: the replay would refuse it"; return 1; }
  [[ -f $S/replay-prompts-gate.jsonl && -f $S/replay-prompts-decode.jsonl ]] || { log "replay-prompts-{gate,decode}.jsonl missing in $S"; return 1; }
  [[ "$(r spark1 "cat $SPARK_DIR/bin/STAMP")" == "$(r spark2 "cat $SPARK_DIR/bin/STAMP")" ]] || { log "binary stamps differ"; return 1; }
  STAMP=$(r spark1 "cat $SPARK_DIR/bin/STAMP"); log "binary $STAMP; out $RAW"
  LE_RECIPE="intarsia $STAMP scripts/dsv41-tp2/replay-test.sh: rank 0 spark1 serve --tp 2 --tp-remote, rank 1 spark2 tp-worker, rank caches r0of2/r1of2, NCCL CX7, cap $CAP, engram hot ${HOT_MB:-1024} MiB"
}

vllm_down() {
  for h in spark1 spark2; do r $h "docker stop -t 60 $VLLM_C >/dev/null"; STOPPED=1; done
  log "vLLM stopped on both"
  # our own watcher: kill our ranks on a box under KILL_GIB (default 5) GiB available; memfine.sh logs every second on each box
  ( while true; do
      line="$(date +%T)"
      for h in spark1 spark2; do
        a=$(ssh -n -o ConnectTimeout=5 $h "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA)
        line="$line $h=$a"
        if [[ "$a" != NA ]] && awk "BEGIN{exit !($a < ${KILL_GIB:-5})}"; then ssh -n $h "pkill -KILL -f '[i]${SPARK_REL:1}/bin/intarsia'"; line="$line KILLED-$h"; fi
      done
      echo "$line"; sleep 5
    done ) > "$RAW/mem.log" 2>&1 & WATCH=$!; disown "$WATCH"
  MF=memfine-$(date +%H%M%S).log
  for h in spark1 spark2; do
    scp -q "$S/memfine.sh" "$h:$SPARK_REL/bin/memfine.sh"
    r $h "mkdir -p $LOGD; nohup bash $SPARK_DIR/bin/memfine.sh ${KILL_GIB:-5} '[i]${SPARK_REL:1}/bin/intarsia' > $LOGD/$h-$MF 2>&1 < /dev/null &"
  done
}

# ---- engine: start_ranks BIN CHUNK MODE (off|on|main|default). The SAME env string goes to both ranks; INTARSIA_DSV41_REPLAY is
# removed first (env -u) so an inherited value never leaks, then set to 1 for on, 0 for off and main, and left unset for default.
start_ranks() {
  local bin=$1 chunk=$2 mode=$3 renv t0 up=0 ra
  renv="INTARSIA_TP_SPIN_MS=${TP_SPIN_MS:-100} INTARSIA_DSV41_CAP=$CAP INTARSIA_DSV41_CHUNK=$chunk INTARSIA_DSV41_ENGRAM=\$HOME/dsv41-engram INTARSIA_DSV41_ENGRAM_HOT_MB=${HOT_MB:-1024} RUST_BACKTRACE=1 ${XENV:-} ${SRV_ENV:-}"
  case $mode in on) renv="$renv INTARSIA_DSV41_REPLAY=1" ;; off|main) renv="$renv INTARSIA_DSV41_REPLAY=0" ;; esac
  TAG=$(date +%H%M%S)
  r spark2 "mkdir -p $LOGD; nohup env -u INTARSIA_DSV41_REPLAY $NCCLENV $renv $bin tp-worker --listen $WIP:$WPORT -m $C1 --backend cuda --tp 2 > $LOGD/worker-$TAG.log 2>&1 < /dev/null &"
  sleep 2
  r spark1 "mkdir -p $LOGD; nohup env -u INTARSIA_DSV41_REPLAY $NCCLENV $renv $bin serve -m $C0 --backend cuda --tp 2 --tp-remote $WIP:$WPORT --port $SPORT --host 0.0.0.0 --served-model-name dsv41-intarsia --no-supervisor --no-ui > $LOGD/serve-$TAG.log 2>&1 < /dev/null &"
  log "ranks started (tag $TAG, $(basename "$bin"), chunk $chunk, replay $mode)"
  t0=$(date +%s)
  while (( $(date +%s) - t0 < 900 )); do
    curl -sf -m 5 "$SURL/health" >/dev/null && { up=1; break; }
    ra=0; alive spark1 "$RANK0_PAT" || ra=$?
    (( ra == 1 )) && { log "rank 0 exited"; break; }
    ra=0; alive spark2 "$RANK1_PAT" || ra=$?
    (( ra == 1 )) && { log "rank 1 exited"; break; }
    sleep 5
  done
  log "serve up=$up after $(( $(date +%s) - t0 )) s"
  r spark1 "grep -aEi 'dsv41|error|panick|replay' $LOGD/serve-$TAG.log | tail -25" | tee -a "$LOG"
  r spark2 "grep -aEi 'dsv41|error|panick|replay' $LOGD/worker-$TAG.log | tail -15" | tee -a "$LOG"
  [[ $up == 1 ]] || { fail "ranks did not come up (see $LOG)"; return 1; }
}
# serve_cfg NAME BIN CHUNK MODE cmd...: fresh ranks, run cmd, keep the logs, stop the ranks. Returns cmd's status.
serve_cfg() {
  local name=$1 bin=$2 chunk=$3 mode=$4 rc; shift 4
  kill_ours
  start_ranks "$bin" "$chunk" "$mode" || { scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$RAW/serve-$name.log" 2>/dev/null; kill_ours; return 1; }
  WATCH_RANKS=1 "$@"; rc=$?
  scp -q "spark1:$SPARK_REL/logs/serve-$TAG.log" "$RAW/serve-$name.log" 2>/dev/null; scp -q "spark2:$SPARK_REL/logs/worker-$TAG.log" "$RAW/worker-$name.log" 2>/dev/null
  kill_ours
  return $rc
}

# ---- clients
# gold OUT NOTE promptargs...: greedy token ids (tools/dsv41/golden_tf.py tokens), temperature 0, ignore_eos
gold() {
  local out=$1 note=$2; shift 2
  bgrun 1200 "$out.log" bash -c 'cd "$1" && shift && exec uv run --with numpy --with safetensors --with tokenizers python tools/dsv41/golden_tf.py tokens "$@"' _ "$REPO" \
    --base-url "$SURL/v1" --model dsv41-intarsia --no-draft-field --tokenizer "$TOK" --out "$out" --note "$note $STAMP cap $CAP" "$@"
}
# le ...: localeval inside our slot (FLEETQ_JOB is set, so it does not queue again)
le() { ( cd /opt/code/deepeval && uv run localeval "$@" --url "$SURL/v1" --model dsv41-intarsia --thinking off --recipe "$LE_RECIPE" ); }
result_of() { grep -ao '/[^ ]*/results/[^ ]*result.json' "$1" | tail -1; }

# ============================================================ the steps
step1() {
  local d=$RAW/step1 mode; mkdir -p "$d"
  for mode in off on; do
    serve_cfg "s1-$mode" "$BIN_BR" 512 $mode gold "$d/tok-$mode.st" "step1 replay=$mode" \
      --prompts "$REPO/tools/prompts/golden.jsonl" --long-prompts 127,128 --long-continue 64 || { fail "ids for replay=$mode could not be generated (${FAIL_WHY:-see $d/tok-$mode.st.log})"; return 1; }
  done
  lib tokcmp "$d/tok-off.st" "$d/tok-on.st" --name-a off --name-b on --require-identical --max-prompt 128 > "$d/compare.txt" 2>&1; local rc=$?
  tee -a "$LOG" < "$d/compare.txt"
  [[ $rc == 0 ]] || { fail "off and on differ on a prompt of at most 128 tokens (or a prompt is longer than 128): $d/compare.txt"; return 1; }
}

step2() {
  local d=$RAW/step2; mkdir -p "$d"
  serve_cfg s2-main "$BIN_MAIN" 512 main gold "$d/tok-main.st" "step2 main $MAIN_REF" || { fail "golden ids of the main binary failed (${FAIL_WHY:-see $d/tok-main.st.log})"; return 1; }
  serve_cfg s2-off "$BIN_BR" 512 off gold "$d/tok-off.st" "step2 branch replay=off" || { fail "golden ids of this branch (replay off) failed (${FAIL_WHY:-see $d/tok-off.st.log})"; return 1; }
  lib tokcmp "$d/tok-main.st" "$d/tok-off.st" --name-a main --name-b off --require-total 640 > "$d/compare.txt" 2>&1; local rc=$?
  tee -a "$LOG" < "$d/compare.txt"
  [[ $rc == 0 ]] || { fail "golden ids of this branch (replay off) are not 640/640 identical to the main binary: $d/compare.txt"; return 1; }
}

step8() {
  local d=$RAW/step8 mode f; mkdir -p "$d"
  for mode in default on; do
    serve_cfg "s8-$mode" "$BIN_BR" 2048 $mode gold "$d/tok-$mode.st" "step8 replay=$mode" || { fail "golden ids for replay=$mode failed (${FAIL_WHY:-see $d/tok-$mode.st.log})"; return 1; }
    for f in serve worker; do
      grep -q 'bounded-replay prefill on' "$RAW/$f-s8-$mode.log" || { fail "$f of replay=$mode does not log the replay on: $RAW/$f-s8-$mode.log"; return 1; }
    done
  done
  lib tokcmp "$d/tok-default.st" "$d/tok-on.st" --name-a default --name-b on --require-total 640 > "$d/compare.txt" 2>&1; local rc=$?
  tee -a "$LOG" < "$d/compare.txt"
  [[ $rc == 0 ]] || { fail "golden ids with the variable unset are not 640/640 identical to =1: $d/compare.txt"; return 1; }
}

# ---- step 3: the gate. serve_cfg runs its command with the ranks up and SRV_ENV (dump settings) in the engine's environment.
gate_dump_dir() { echo "$SPARK_DIR/dump/g$1-$2"; }   # CHUNK MODE; the same relative path on both Sparks
step3() {
  local d=$RAW/step3 chunk mode h rc=0; mkdir -p "$d"
  for chunk in $GATE_CHUNKS; do
    for mode in off on; do
      for h in spark1 spark2; do r $h "rm -rf $(gate_dump_dir "$chunk" "$mode")"; done
      SRV_ENV="INTARSIA_DSV41_DUMP=$(gate_dump_dir "$chunk" "$mode") INTARSIA_DSV41_DUMP_POS=$GATE_POS INTARSIA_DSV41_DUMP_WHAT='$GATE_WHAT'"
      serve_cfg "s3-c$chunk-$mode" "$BIN_BR" "$chunk" $mode gold "$d/tok-c$chunk-$mode.st" "step3 gate chunk $chunk replay=$mode" \
        --prompts "$S/replay-prompts-gate.jsonl" || { SRV_ENV=""; fail "gate requests for chunk $chunk replay=$mode failed (${FAIL_WHY:-see $d})"; return 1; }
      SRV_ENV=""
      for h in spark1 spark2; do
        scp -rq "$h:$SPARK_REL/dump/g$chunk-$mode" "$d/$h-c$chunk-$mode" 2>/dev/null || { fail "could not copy the dumps of $h (chunk $chunk, $mode)"; return 1; }
      done
    done
    for h in spark1 spark2; do
      log "gate, chunk $chunk, $h: encoder half off vs on"
      lib gatecmp "$d/$h-c$chunk-off" "$d/$h-c$chunk-on" "${GATE_ARGS[@]}" > "$d/gate-c$chunk-$h.txt" 2>&1; rc=$?
      tee -a "$LOG" < "$d/gate-c$chunk-$h.txt"
      [[ $rc == 0 ]] || { fail "gate failed on $h at chunk $chunk: $d/gate-c$chunk-$h.txt"; return 1; }
    done
  done
}

speed_cell() { # CHUNK MODE PASS
  local d=$RAW/step4 label="dsv41replay-speed-c$1-$2$3" res
  BG_TAIL=25 bgrun 1500 "$d/le-$label.log" le speed --sizes "$S3_SIZES" --gen 8 --reps "$S3_REPS" --label "$label" || return 1
  res=$(result_of "$d/le-$label.log"); [[ -f $res ]] || { fail "no result.json in $d/le-$label.log"; return 1; }
  cp "$res" "$d/result-c$1-$2$3.json"; echo "$1:$2:$3:$d/result-c$1-$2$3.json" >> "$d/cells.txt"
}
step4() {
  local d=$RAW/step4 chunk pass mode n=0 rc; mkdir -p "$d"; : > "$d/cells.txt"
  for chunk in $S3_CHUNKS; do
    for pass in 1 2; do for mode in off on; do   # ABAB: off, on, off, on
      serve_cfg "s4-c$chunk-$mode$pass" "$BIN_BR" "$chunk" $mode speed_cell "$chunk" $mode $pass || { fail "speed cell chunk $chunk $mode pass $pass failed (${FAIL_WHY:-see $d})"; return 1; }
      n=$((n + 1))
    done; done
  done
  lib speedtab "$d/speed-table.md" --max-ratio 1.05 --expect-cells "$n" $(cat "$d/cells.txt") | tee -a "$LOG"; rc=${PIPESTATUS[0]}
  [[ $rc == 0 ]] || fail "timing criterion failed (on/off prefill time above 1.05 or a cell with errors): $d/speed-table.md"
}

step5() {
  local d=$RAW/step5 mode rc=0; mkdir -p "$d"
  for mode in off on; do
    serve_cfg "s5-$mode" "$BIN_BR" 512 $mode gold "$d/tok-$mode.st" "step5 replay=$mode" \
      --prompts "$REPO/tools/prompts/golden.jsonl" "$REPO/tools/prompts/golden_long.jsonl" "$S/replay-prompts-decode.jsonl" || { fail "ids for replay=$mode failed (${FAIL_WHY:-see $d})"; return 1; }
  done
  # (b) the decode window ring: 160 decode tokens after prompts above the floor, on vs off (a tripwire against garbage: 30 % equal positions)
  lib tokcmp "$d/tok-off.st" "$d/tok-on.st" --name-a off --name-b on --prefix d_ --min-equal-pct 30 --same-lengths > "$d/compare-decode.txt" 2>&1; rc=$?
  tee -a "$LOG" < "$d/compare-decode.txt"
  [[ $rc == 0 ]] || { fail "on vs off over 160 decode tokens: $d/compare-decode.txt"; return 1; }
  # (a) TensorFold's ids (served lane, replay always on, TF_DS_REPLAY_FLOOR unset: see the comment at TF_IDS)
  if [[ ! -f $TF_IDS ]]; then
    log "(a) SKIPPED: TensorFold's ids are not at $TF_IDS (docs/results/2026-11-01-dsv41-tensorfold-token-agreement.txt has the summary only)"
    STEP_NOTE="(a) SKIPPED, (b) passed"; return 0
  fi
  lib sha256 "$TF_IDS" | tee -a "$LOG"
  lib tokcmp "$TF_IDS" "$d/tok-off.st" --name-a TF --name-b off --common > "$d/compare-tf-off.txt" 2>&1
  lib tokcmp "$TF_IDS" "$d/tok-on.st" --name-a TF --name-b on --common > "$d/compare-tf-on.txt" 2>&1
  tail -n 14 "$d/compare-tf-off.txt" "$d/compare-tf-on.txt" | tee -a "$LOG"
  lib tfagree "$TF_IDS" "$d/tok-off.st" "$d/tok-on.st" > "$d/tfagree.txt" 2>&1; rc=$?
  tee -a "$LOG" < "$d/tfagree.txt"
  log "reference: docs/results/2026-11-01-dsv41-tensorfold-token-agreement.txt, TF vs an earlier intarsia golden: leading 335/640, positionwise 398/640 (the 8 prompts of <= 99 ids already diverge there, with no cut)"
  [[ $rc == 0 ]] || { fail "on is worse than off against TensorFold's ids, or off != on on a short prompt: $d/tfagree.txt"; return 1; }
}

q_arm() { # SHARD ARM
  local d=$RAW/step6 label="dsv41replay-$2-$1" res
  BG_TAIL=15 bgrun $((90 * 60)) "$d/le-$label.log" le quality --dataset "$QDIR/final.$1.jsonl" --judge none --max-tokens "$Q_TOKENS" --label "$label" || return 1
  res=$(result_of "$d/le-$label.log"); [[ -f $res ]] || { fail "no result.json in $d/le-$label.log"; return 1; }
  cp "$res" "$d/result-$2-$1.json"
}
step6() {
  local d=$RAW/step6 sh arm st=0 rc items=0 n; mkdir -p "$d"; : > "$d/pairs.txt"
  [[ -f $Q2RAW/items.sha256.json && -f $Q2RAW/items.manifest.json ]] || { fail "frozen item files missing: $Q2RAW/items.{sha256,manifest}.json (branch dsv41/chunk32k-2)"; return 1; }
  cp "$Q2RAW/items.sha256.json" "$d/"; lib sha256 "$Q2RAW/items.manifest.json" | tee -a "$LOG"
  # shellcheck disable=SC2086
  lib checksha "$Q2RAW/items.sha256.json" "$QDIR" final. $Q_SHARDS | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { fail "a frozen shard is missing or its sha256 differs; regenerate in $C32: uv run --no-project --with tokenizers python docs/results/2026-11-04-dsv41-chunk32k-quality-2/raw/gen_items2.py --spec <spec-final.json> --out .scratch/final"; return 1; }
  for sh in $Q_SHARDS; do
    if [[ -n ${STOP_EPOCH:-} ]] && (( $(date +%s) + S6_MARGIN_MIN * 60 > STOP_EPOCH )); then log "STOP_EPOCH close: not starting shard $sh or the later ones"; STEP_NOTE="PARTIAL"; break; fi
    for arm in off on; do
      serve_cfg "s6-$sh-$arm" "$BIN_BR" "$Q_CHUNK" $arm q_arm "$sh" $arm || { fail "shard $sh arm $arm failed (${FAIL_WHY:-see $d})"; st=1; break; }
    done
    [[ $st == 0 ]] || break
    echo "$sh $d/result-off-$sh.json $d/result-on-$sh.json" >> "$d/pairs.txt"
    n=$(wc -l < "$QDIR/final.$sh.jsonl"); items=$((items + n))
  done
  if [[ -s $d/pairs.txt ]]; then
    lib quality "$Q2RAW/items.manifest.json" "$d/pairs.txt" "$d/quality.md" --expect-items $((items * 9 / 10)) | tee -a "$LOG"; rc=${PIPESTATUS[0]}
    [[ $rc == 0 ]] || { fail "quality criterion failed (too few items scored, or on lost accuracy): $d/quality.md"; st=1; }
  else
    fail "no shard completed"; st=1
  fi
  return $st
}

# ---- step 7: the recent-window items. One unit = a chunk size and the lengths run at it; per unit an off arm then an on arm (fresh
# ranks each), the same dataset files in both. s7_file LEN: the file to run (the first 6*K rows of the frozen file when K < 12: the
# files are ordered round robin over the 6 buckets, so a head is balanced).
s7_file() {
  local ln=$1 k=$S7_K4 f=$RDIR/recent.$1.jsonl
  [[ $ln == 32k ]] && k=$S7_K32
  [[ $k == 12 ]] && { echo "$f"; return 0; }
  head -n $((6 * k)) "$f" > "$RAW/step7/recent.$ln.k$k.jsonl" && echo "$RAW/step7/recent.$ln.k$k.jsonl"
}
s7_arm() { # CHUNK ARM LENGTHS
  local chunk=$1 arm=$2 d=$RAW/step7 ln f label res to
  for ln in ${3//,/ }; do
    f=$(s7_file "$ln") || { fail "could not cut the $ln file"; return 1; }
    label="dsv41replay-s7-c$chunk-$arm-$ln"
    to=$(( 2 * $(s7_items "$ln") * $(s7_ds "$chunk" "$ln" off) / 10 + 900 ))
    BG_TAIL=12 bgrun "$to" "$d/le-$label.log" le quality --dataset "$f" --judge none --max-tokens "$S7_TOKENS" --label "$label" || { fail "localeval $label failed or timed out (see $d/le-$label.log)"; return 1; }
    res=$(result_of "$d/le-$label.log"); [[ -f $res ]] || { fail "no result.json in $d/le-$label.log"; return 1; }
    cp "$res" "$d/result-c$chunk-$arm-$ln.json"
    lib recentcheck "$R_MAN" "$d/result-c$chunk-$arm-$ln.json" | tee -a "$LOG"
    [[ ${PIPESTATUS[0]} == 0 ]] || { fail "an invalid reply set from the server ($label): an error or EMPTY reply, a prompt_tokens that differs from the manifest, or a prefix-cache hit; $d/result-c$chunk-$arm-$ln.json"; return 1; }
  done
}
step7() {
  local d=$RAW/step7 u chunk lens ln rc ran=0 now off_s pair_s files=""
  mkdir -p "$d"; : > "$d/pairs.txt"
  # the frozen item files: both sha256-checked before any arm starts (a regenerated or edited file would move every distance)
  lib sha256 "$R_MAN" | tee -a "$LOG"
  lib checksha "$R_SHA" "$RDIR" recent. 4k 32k | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { fail "the item files in $RDIR are missing or their sha256 differs from $R_SHA; regenerate: uv run --no-project --with tokenizers python $S/replay-recent-gen.py --out $RDIR --n4 12 --n32 12 (it must give the committed sha256s)"; return 1; }
  for u in $S7_PLAN; do
    chunk=${u%%:*}; lens=${u#*:}
    now=$(date +%s); off_s=$(s7_arm_s "$chunk" "$lens" off); pair_s=$(s7_unit_s "$u")
    if [[ -n ${STOP_EPOCH:-} ]]; then
      if (( now + off_s > STOP_EPOCH )); then log "STOP_EPOCH: unit $u not started (its on arm would start after it)"; STEP_NOTE="PARTIAL"; continue; fi
      if (( now + pair_s + RESTORE_MIN * 60 > RESTORE_BY )); then log "RESTORE_BY: unit $u not started (it and the vLLM restore would end after $(TZ=America/New_York date -r "$RESTORE_BY" '+%a %H:%M %Z'))"; STEP_NOTE="PARTIAL"; continue; fi
    fi
    log "step 7 unit $u: chunk $chunk, lengths $lens, planned $((pair_s / 60)) min"
    for arm in off on; do
      serve_cfg "s7-c$chunk-$arm" "$BIN_BR" "$chunk" $arm s7_arm "$chunk" $arm "$lens" || { fail "unit $u, arm $arm failed (${FAIL_WHY:-see $d})"; return 1; }
    done
    files=""; for ln in ${lens//,/ }; do files="$files $d/result-c$chunk-off-$ln.json $d/result-c$chunk-on-$ln.json"; done
    echo "$chunk$files" >> "$d/pairs.txt"; ran=$((ran + 1))
  done
  if (( ran == 0 )); then log "step 7: no unit started (STOP_EPOCH / RESTORE_BY)"; STEP_NOTE="SKIPPED"; return 0; fi
  lib recent "$R_MAN" "$d/pairs.txt" "$d/recent.md" | tee -a "$LOG"; rc=${PIPESTATUS[0]}
  case $rc in
    0) return 0 ;;
    2) STEP_NOTE="DEFECT per the reading rule, see $d/recent.md"; return 0 ;;
    *) fail "the step 7 measurement is invalid (INVALID MEASUREMENT at the top of $d/recent.md)"; return 1 ;;
  esac
}

# ============================================================ main
prep_build || { fail "build failed (see $RAW/build-*.log)"; log "PREP FAILED: $FAIL_WHY"; exit 1; }
preflight || { log "PREP FAILED: preflight"; exit 1; }
vllm_down
RC=0
for n in $SELECTED; do
  t0=$(date +%s); FAIL_WHY=""; STEP_NOTE=""; STEP_DEADLINE=$(( t0 + 2 * 60 * $(est "$n") ))
  if [[ -n ${STOP_EPOCH:-} ]]; then
    # no new step after STOP_EPOCH, and no step may run past the time vLLM must be back (less the restore itself)
    if (( t0 >= STOP_EPOCH )); then
      log "STOP_EPOCH passed: step $n not started"
      RES_N+=("$n"); RES_ST+=("SKIPPED (STOP_EPOCH)"); RES_EL+=(0); RES_PL+=("$(est "$n")"); continue
    fi
    (( STEP_DEADLINE > RESTORE_BY - RESTORE_MIN * 60 )) && STEP_DEADLINE=$(( RESTORE_BY - RESTORE_MIN * 60 ))
  fi
  log "=== STEP $n: $(name_of "$n") (planned $(est "$n") min) ==="
  if "step$n"; then
    st="PASS${STEP_NOTE:+ ($STEP_NOTE)}"; [[ $STEP_NOTE == SKIPPED ]] && st=SKIPPED
    [[ $STEP_NOTE == DEFECT* ]] && { st="$STEP_NOTE"; RC=1; }
  else st=FAIL; fi
  RES_N+=("$n"); RES_ST+=("$st"); RES_EL+=($(( $(date +%s) - t0 ))); RES_PL+=("$(est "$n")")
  if [[ $st == FAIL ]]; then
    log "STEP $n FAILED: ${FAIL_WHY:-no reason recorded; see $LOG}"; RC=1
    [[ " ${CONT_AFTER:-} " == *" $n "* ]] && { log "CONT_AFTER lists step $n: the later steps still run"; continue; }
    break
  fi
  log "STEP $n $st in $(( ($(date +%s) - t0) / 60 )) min"
done
exit $RC
