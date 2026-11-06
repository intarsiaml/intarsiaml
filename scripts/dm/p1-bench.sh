#!/bin/zsh
# Lane dm phase 1 on endor (ONE exclusive fleetq slot): a fresh server per visit, localeval suite cell speed-spec (natural-v1
# 1k/4k/16k, 256 forced tokens) every visit; the 7 categories (scripts/dfh/cats, 512 forced tokens, `localeval quality --judge none`)
# on the visits named in CATV. Arms: mtp2 (default MTP k2), dflK (--spec dflash --spec-k K), tf (TensorFold DFlash2, the endor service's
# binary on its own port). Ours run with ENVS (default none; MR is default on).
#   VISITS="mtp2 dfl7 mtp2 dfl7 dfl3 tf" CATV="1 2 5 6" zsh scripts/dm/p1-bench.sh <binary> <out dir>
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O; O=${O:A}
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
R=${0:A:h:h:h}; HERE=$R/scripts
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=8298
DRAFT=/Users/josef/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4
TF=/Users/josef/.local/share/tensorfold-mac-all/tensorfold
TOK=(--tokenizer-repo Vontra/Qwen3.8-27B-MLX-4bit --tokenizer-revision 70ae7fac63274ff2eac54152031433374cb80f2f)
GEN=512
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT
up() { local i; for i in {1..300}; do curl -sf -m 3 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && return 0; kill -0 $SPID 2>/dev/null || return 1; sleep 1; done; return 1; }
down() { kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done; }
metric() { curl -s -m 5 http://127.0.0.1:$PORT/metrics | grep -E '^intarsia_spec_(steps|proposed_tokens|accepted_tokens)_total' | sed 's/{[^}]*}//' | tr '\n' ' '; }
v=0
for arm in ${=VISITS:-mtp2 dfl7 mtp2 dfl7 tf}; do
  v=$((v + 1)); say "visit $v arm $arm: serve"
  case $arm in
    tf)
      HF_HUB_OFFLINE=1 $TF serve Vontra/Qwen3.8-27B-MLX-4bit --host 127.0.0.1 --port $PORT --context 20480 --name bench > $O/serve-$v-$arm.out 2> $O/serve-$v-$arm.err < /dev/null &
      SPID=$!; lane=spec:dflash; dr=dflash2
      rec="tensorfold Vontra/Qwen3.8-27B-MLX-4bit --context 20480 (endor service binary, $(basename $(readlink -f $TF 2>/dev/null || echo $TF)); DFlash2 tree drafter default; lane dm)" ;;
    *)
      case $arm in mtp2) spec=(--spec mtp --spec-k 2); lane=spec:mtp; dr=mtp ;; dfl*) spec=(--spec dflash --spec-k ${arm#dfl} --spec-draft $DRAFT); lane=spec:dflash; dr=dflash2 ;; esac
      env ${=ENVS:-} $BIN serve --backend metal --model $M --served-model-name qwen38-27b-$arm --max-seq 20480 --max-batch 1 --profile-every 0 --no-supervisor $spec \
        --host 127.0.0.1 --port $PORT > $O/serve-$v-$arm.out 2> $O/serve-$v-$arm.err < /dev/null &
      SPID=$!
      rec="intarsia serve --backend metal --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 20480 --max-batch 1 ${${spec[*]}//$DRAFT/z-lab-DFlash2} env ${ENVS:-} (lane dm, $(git -C $R rev-parse --short HEAD))" ;;
  esac
  if ! up; then say "visit $v arm $arm: server failed"; tail -20 $O/serve-$v-$arm.err >> $O/steps.log; down; continue; fi
  mid=$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
  ( cd /opt/code/deepeval && uv run localeval suite --url http://127.0.0.1:$PORT/v1 --model $mid --thinking off $TOK --lane $lane --drafter $dr --max-num-seqs 1 \
      --max-context 20480 --power-mode default --tier adhoc --cells speed-spec --label "dm-p1-v$v-$arm-speed-spec" --recipe "$rec" --wait-idle 60 ) > $O/suite-$v-$arm.log 2>&1 < /dev/null
  say "visit $v arm $arm suite speed-spec exit $? $(grep -aoE 'results/[^`]+' $O/suite-$v-$arm.log | tail -1)"
  if [[ " ${CATV:-} " == *" $v "* ]]; then
    printf '[targets.dm-%s]\nurl = "http://127.0.0.1:%s/v1"\nmodel = "%s"\nextra_body = { ignore_eos = true, min_tokens = %s }\n' "$arm" "$PORT" "$mid" "$GEN" > $O/targets-$arm.toml
    for f in $HERE/dfh/cats/*.jsonl; do
      c=$(basename $f .jsonl); m0=$(metric)
      ( cd /opt/code/deepeval && LOCALEVAL_TARGETS=$O/targets-$arm.toml uv run localeval quality "dm-$arm" --thinking off --label "dm-p1-v$v-$arm-$c" \
          --recipe "$rec" --dataset $f --judge none --max-tokens $GEN --wait-idle 60 ) > $O/q-$v-$arm-$c.log 2>&1 < /dev/null
      rc=$?; m1=$(metric)
      echo "$arm $c r$v before: $m0" >> $O/spec-metrics.txt; echo "$arm $c r$v after:  $m1" >> $O/spec-metrics.txt
      say "round $v arm $arm $c: localeval exit $rc $(grep -aoE 'results/[^ ]+' $O/q-$v-$arm-$c.log | tail -1)"
    done
  fi
  down
done
say done
