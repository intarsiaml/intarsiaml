#!/bin/zsh
# Metal P3 gates, one fleetq slot. base = spike/metal 2e1fdb1 build (chunk 256, no [tuning.metal]); p3 = this branch
# (recipes/qwen3.toml [tuning.metal] prefill_chunk = 2048, stamped build, no --prefill-chunk flag anywhere unless stated).
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad
W=/opt/code/intarsiaml/.claude/worktrees/agent-af7ec7dd89422c6a9
R=$W/docs/results/raw/2026-10-14-metal-prefill-p3/gates
RAW=$W/docs/results/raw
PB=$RAW/2026-10-14-metal-phase-b
FD=$RAW/2026-10-14-metal-prefill-p1/full
BASE=$S/p3/intarsia-base
P3=$S/p3/intarsia-p3
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
export INTARSIA_CPU_FREE_MEM_GB=60
mkdir -p $R/decode $R/identity $R/chunk $R/accuracy $R/plan
J=$(fleetq acquire --host endor --exclusive --mem 24 --expect 90 --label "metal P3: gates (tests, accuracy, ids, chunk invariance, decode, qualify)" --wait --json | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "fleetq job $J" | tee $R/fleetq-job.txt
trap "fleetq release $J" EXIT
export FLEETQ_JOB=$J
cd $W
{ echo "date: $(date -Iseconds)"; echo "head: $(git rev-parse HEAD)"; echo "base binary: $(ls -l $BASE)"; echo "p3 binary: $(ls -l $P3)"; $P3 --version; sw_vers | tr '\n' ' '; echo; } > $R/versions.txt 2>&1
pgrep -fl zoom.us > $R/zoom.txt 2>&1
for i in 1 2 3; do ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*'; sleep 1; done > $R/idle-start.txt

echo "== 1. unit tests and admission"
cargo test -p intarsia-backend-metal --release -- --test-threads=1 > $R/cargo-test-metal.txt 2>&1; echo "cargo test metal exit $?"
$P3 kernels test --backend metal --device 0 --model $M -v > $R/kernels-test.txt 2>&1; echo "kernels test exit $?"
grep -cE "ADMITTED" $R/kernels-test.txt; grep -E "FAIL|REJECT" $R/kernels-test.txt | head; grep -E "kernels," $R/kernels-test.txt

echo "== 2. the default is live (plan show, scratch per chunk)"
for C in default 256 1024 2048; do
  if [ $C = default ]; then X=(); else X=(--prefill-chunk $C); fi
  $P3 plan show --model $M --backend metal --device 0 --json $X > $R/plan/metal-$C.json 2> $R/plan/metal-$C.err
  echo "plan metal $C: $(grep -o '"prefill_chunk[a-z_]*": *[^,]*' $R/plan/metal-$C.json | tr '\n' ' ') $(grep -o '"scratch_bytes": *[0-9]*' $R/plan/metal-$C.json | tail -1)"
done
$P3 plan show --model $M --backend cpu --json > $R/plan/cpu-default.json 2> $R/plan/cpu-default.err
echo "plan cpu default: $(grep -o '"prefill_chunk[a-z_]*": *[^,]*' $R/plan/cpu-default.json | tr '\n' ' ')"

echo "== 3. decode dispatches per step (364 expected) and tok/s, base vs p3, ABBA"
for P in short 4k; do
  for B in base p3 p3 base; do
    BIN=$P3; [ $B = base ] && BIN=$BASE
    N=$(ls $R/decode | grep -c "^$B-$P-.*err$"); F=$R/decode/$B-$P-$((N+1))
    INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 \
      --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $PB/prompt-$P.txt)" > $F.out 2> $F.err
    echo "$B $P: $(python3 -I $PB/b2b/period.py $F.jsonl 364 128 2>&1 | tail -1) $(grep -o '([0-9.]* tok/s)' $F.err | tail -1) $(grep -o 'prefill chunk [0-9]* ([^)]*)' $F.err)"
  done
done

echo "== 4. accuracy against the fp32 goldens (bf16 gate class)"
for B in p3 base; do
  BIN=$P3; [ $B = base ] && BIN=$BASE
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/accuracy/$B-short.out 2> $R/accuracy/$B-short.err; echo "accuracy $B short exit $?"
  $BIN accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/accuracy/$B-long.out 2> $R/accuracy/$B-long.err; echo "accuracy $B long exit $?"
  for F in short long; do
    echo "$B $F: $(grep -E 'floor \[bf16\]' $R/accuracy/$B-$F.out | grep -oE '(prefill|decode[^ ]*) floor|top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|=> (PASS|FAIL)' | tr '\n' ' ')"
  done
  grep -h "prefill chunk" $R/accuracy/$B-short.err $R/accuracy/$B-long.err | sort -u
done
# the long golden at other chunk sizes (rounding effect of the chunk on KL, 8192-token prompts)
for C in 256 1024; do
  $P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 8192 --prefill-chunk $C > $R/accuracy/p3-long-c$C.out 2> $R/accuracy/p3-long-c$C.err
  echo "p3 long prefill chunk $C: $(grep -E 'floor \[bf16\]' $R/accuracy/p3-long-c$C.out | grep -oE 'top-1 [0-9.]+%|full-vocab KL [0-9.e-]+|=> (PASS|FAIL)' | tr '\n' ' ')"
done

echo "== 5. greedy ids, base vs p3"
for N in 48 256; do
  for P in short 4k; do
    for B in base p3; do
      BIN=$P3; [ $B = base ] && BIN=$BASE
      $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens $N --ignore-eos --show-ids \
        --prompt "$(cat $PB/prompt-$P.txt)" > $R/identity/$B-$P-$N.out 2> $R/identity/$B-$P-$N.err
    done
    python3 -I $FD/first_divergence.py $R/identity/base-$P-$N.err $R/identity/p3-$P-$N.err "$P $N tokens base vs p3" | tee -a $R/identity/summary.txt
  done
done

echo "== 6. chunk invariance (p3 binary): 4k prompt ids at several --prefill-chunk vs the default (2048), short golden prefill KL"
$P3 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
  --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/4k-cdefault.out 2> $R/chunk/4k-cdefault.err
grep -h "prefill chunk" $R/chunk/4k-cdefault.err
for C in 64 128 256 512 1024 2048 4050; do
  $P3 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 48 --ignore-eos --show-ids \
    --prefill-chunk $C --prompt "$(cat $PB/prompt-4k.txt)" > $R/chunk/4k-c$C.out 2> $R/chunk/4k-c$C.err
  python3 -I $FD/first_divergence.py $R/chunk/4k-cdefault.err $R/chunk/4k-c$C.err "p3 4k: default(2048) vs chunk $C" | tee -a $R/chunk/summary.txt
done
for C in 16 64 128 2048; do
  $P3 accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --mode prefill --max-seq 4096 --prefill-chunk $C \
    > $R/chunk/acc-short-c$C.out 2> $R/chunk/acc-short-c$C.err
  echo "accuracy prefill short chunk $C: $(grep -oE 'full-vocab KL [0-9.e-]+' $R/chunk/acc-short-c$C.out | head -1)"
done

echo "== 7. qualify (full, fresh, goldens)"
$P3 qualify --model $M --backend metal --bandwidth-gbs 614 --golden-dir $G --out $R/qualify > $R/qualify-console.log 2>&1; echo "qualify exit $?"
grep -E "Verdict|verdict|Accuracy|Prefill|Decode" $R/qualify/full-report.md | cut -c1-260
echo done
