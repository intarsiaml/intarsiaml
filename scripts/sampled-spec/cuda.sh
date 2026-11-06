#!/usr/bin/env bash
# Sampled speculation on CUDA (spark1), Mac side, inside one fleetq slot (FLEETQ_JOB set):
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 120 --label "ss: CUDA" -- scripts/sampled-spec/cuda.sh
# (1) build base (602a1fcfe) and branch CLIs on spark1 (~/intarsiaml-ss); (2) plan show 35B json, base vs branch; (3) kernels test
# --quick; (4) greedy ids 35B tp=1 --spec mtp --spec-k 2, base vs branch with INTARSIA_SPEC_SAMPLED=1; (5) localeval speed-spec
# (natural-v1 1k / 4k, suite copy as on Metal) for T0 off / mtp, T0.7 and T1.0 off / match / reject.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
W="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
O=$W/.scratch/cuda; mkdir -p $O
HN=spark-4e77; PORT=18431
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='~/intarsiaml-ss'; R="$L/src"; B=$L/bin
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes spark1 "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
build() { # build <ref> <name>
  rsh "rm -rf $L/stage && mkdir -p $L/stage $R $B"
  git -C "$W" archive "$1" | ssh -o BatchMode=yes spark1 "tar -x -C $L/stage"
  rsh "rsync -a --delete --exclude target $L/stage/ $R/ && rm -rf $L/stage && cd $R && find . -path ./target -prune -o -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.toml' \) -print0 | xargs -0 touch"
  rsh "rm -f $R/target/release/intarsia; cd $R && CARGO_BUILD_JOBS=4 MAX_JOBS=4 cargo build --release --features cuda -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3; cp $R/target/release/intarsia $B/$2"
}
if [[ -z "${SKIP_BUILD:-}" ]]; then
  echo "== build $(date +%T)"; build 602a1fcfe base; build lane-ss/sampled-spec br
fi
echo "== plan show $(date +%T)"
for b in base br; do rsh "$B/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --json" > $O/plan-$b.json 2> $O/plan-$b.err; done
cmp $O/plan-base.json $O/plan-br.json && echo "plan show json: identical" | tee $O/summary.txt || echo "plan show json: DIFFERS" | tee $O/summary.txt
for b in base br; do rsh "$B/$b plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 40" > $O/plan-$b.txt 2>/dev/null; done
cmp $O/plan-base.txt $O/plan-br.txt && echo "plan show text: identical" | tee -a $O/summary.txt || echo "plan show text: DIFFERS" | tee -a $O/summary.txt
echo "== kernels $(date +%T)"
rsh "$B/br kernels test --backend cuda --quick" > $O/kernels.log 2>&1; echo "kernels exit $? $(tail -1 $O/kernels.log)" | tee -a $O/summary.txt
echo "== greedy ids $(date +%T)"
P4K="$(cat /opt/code/deepeval/tests/data/prompts/natural-v1/n4096.txt)"
printf '%s' "$P4K" | ssh -o BatchMode=yes spark1 "cat > $L/p4k.txt"
for b in base br; do
  for p in short 4k; do
    pr="Write a short paragraph about the history of the printing press."
    rsh "INTARSIA_SPEC_SAMPLED=\$([ $b = br ] && echo 1 || echo 0) $B/$b generate --backend cuda --model $M35 --max-seq 8192 --mem-budget 40 --spec mtp --spec-k 2 --temperature 0 --max-tokens 96 --ignore-eos --show-ids --prompt \"\$([ $p = 4k ] && cat $L/p4k.txt || echo '$pr')\"" > $O/ids-$b-$p.out 2> $O/ids-$b-$p.err
  done
done
for p in short 4k; do cmp -s <(grep '^ids:' $O/ids-base-$p.err) <(grep '^ids:' $O/ids-br-$p.err) && [ -n "$(grep '^ids:' $O/ids-br-$p.err)" ] && echo "ids $p identical" || echo "ids $p DIFFER"; done | tee -a $O/summary.txt
echo "== speed $(date +%T)"
T=$W/.scratch/cuda/targets.toml
sed "s#http://127.0.0.1:18431#http://$HN:$PORT#; s#host = \"endor\"#host = \"spark-4e77\"#" $W/.scratch/suite/targets.toml > $T
for arm in T00-off T00-mtp T07-off T07-match T07-reject T10-off T10-match T10-reject; do
  t=${arm%%-*}; mode=${arm#*-}; spec="--spec mtp --spec-k 2"; env_s=0; lane=spec:mtp; suite=$W/.scratch/suite/suite-spec.json
  case $mode in off) spec="--spec off"; lane=plain; suite=$W/.scratch/suite/suite-plain.json;; match) env_s=match;; reject) env_s=reject;; esac
  cmd="INTARSIA_SPEC_SAMPLED=$env_s intarsia serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --model-id ss --max-seq 8192 --mem-budget 40 --max-batch 1 --profile-every 0 --no-supervisor --no-ui $spec"
  echo "== $arm $(date +%T)"
  SPID=$(rsh "nohup env INTARSIA_SPEC_SAMPLED=$env_s $B/br serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --model-id ss --max-seq 8192 --mem-budget 40 --max-batch 1 --profile-every 0 --no-supervisor --no-ui $spec > $L/serve-$arm.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 150); do curl -sf http://$HN:$PORT/health >/dev/null && break; sleep 2; done
  ( cd /opt/code/deepeval && LOCALEVAL_TARGETS=$T uv run localeval suite ss-t${t#T} --tier adhoc --cells speed-spec --lane $lane \
      --suite "$suite" --tokenizer-file ~/models/Qwen3.6-35B-A3B-NVFP4/tokenizer.json --label "ss-c35-$arm" --recipe "$cmd" ) > $O/c35-$arm.localeval.log 2>&1
  echo "localeval exit $?"
  curl -s http://$HN:$PORT/metrics | grep -E '^intarsia_spec_(steps|proposed_tokens|accepted_tokens)_total' > $O/c35-$arm.metrics1
  rsh "kill $SPID; for i in \$(seq 1 20); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"
  rsh "cat $L/serve-$arm.log" > $O/c35-$arm.serve.log
done
echo "== done $(date +%T)"
