#!/bin/zsh
# Lane df, Metal fix 1 speed: ABAB of INTARSIA_METAL_MXFP4_MR (A unset, B = $BV, default 1) on the 27B, one exclusive slot, fresh serve a visit.
#   fleetq run --host endor --exclusive --mem 30 --expect 75 --label "..." -- zsh scripts/df/metal-abab.sh <binary> <out dir>
# VISITS (default "A B A B") per spec arm (SPECS, default "mtp2 off"); mtp2 runs speed-spec + speed-plain, off runs speed-plain.
set -u
BIN=${1:?binary}; O=${2:?out dir}; mkdir -p $O
[[ -n "${FLEETQ_JOB:-}" ]] || { echo "not inside a fleetq slot" >&2; exit 2; }
M=${M:-/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4}; PORT=8296; BV=${BV:-1}
TOK=(--tokenizer-repo Vontra/Qwen3.8-27B-MLX-4bit --tokenizer-revision 70ae7fac63274ff2eac54152031433374cb80f2f)
say() { echo "$(date +%T) $*" | tee -a $O/steps.log; }
SPID=""; trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null' EXIT
n=0
for sp in ${=SPECS:-mtp2 off}; do
  for v in ${=VISITS:-A B A B}; do
    n=$((n+1)); envs=(); [[ $v == B ]] && envs=(INTARSIA_METAL_MXFP4_MR=$BV)
    extra=(); [[ $sp == off ]] && extra=(--spec off); [[ $sp == mtp3 ]] && extra=(--spec mtp --spec-k 3); [[ $sp == mtp4 ]] && extra=(--spec mtp --spec-k 4)
    say "visit $n $sp $v: starting server"
    env $envs $BIN serve --backend metal --model $M --served-model-name qwen38-27b --max-seq 20480 --profile-every 0 --no-supervisor $extra --host 127.0.0.1 --port $PORT \
      > $O/serve-$n-$sp-$v.out 2> $O/serve-$n-$sp-$v.err < /dev/null &
    SPID=$!
    up=0; for i in {1..240}; do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
    [[ $up == 1 ]] || { say "visit $n: server failed"; SPID=""; continue; }
    lane=spec:mtp; cells="speed-spec speed-plain"; [[ $sp == off ]] && { lane=plain; cells=speed-plain; }
    for c in ${=cells}; do
      ( cd /opt/code/deepeval && uv run localeval suite --url http://127.0.0.1:$PORT/v1 --model qwen38-27b --thinking off $TOK --lane $lane --drafter mtp --max-num-seqs 1 \
          --max-context 20480 --power-mode default --tier adhoc --cells $c --label "df-mr-$n-$sp-$v-$c" \
          --recipe "intarsia serve --backend metal --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 20480 ${extra[*]} ${envs[*]} (lane df fix 1)" --wait-idle 60 ) > $O/suite-$n-$sp-$v-$c.log 2>&1 < /dev/null
      say "visit $n $sp $v $c: exit $? $(grep -aoE 'results/[^`]+' $O/suite-$n-$sp-$v-$c.log | tail -1)"
    done
    kill $SPID 2>/dev/null; for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""
    for i in {1..30}; do lsof -iTCP:$PORT -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  done
done
say done
