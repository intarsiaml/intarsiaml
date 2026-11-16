#!/bin/zsh
# Lane mstore: the prefix-reuse gate (`accuracy --mode reuse`) on Metal with the MTP default, store off and store on.
# usage (inside a fleetq slot): scripts/mstore/reuse.sh 35b|27b
# Store on: INTARSIA_PREFIX_STORE=on + INTARSIA_MTP_STORE=1. Caveat: the gate's "cold" run follows a cache flush that replaces the
# held sequence, not the store, so with the store on the cold run can itself restore from entries the warm run left; the
# fresh-engine comparisons of tests/mtp_store_metal.rs are the store's warm == cold gate. This shows the drafting + store
# path passes the reuse gate's tolerance checks and its aligned-restore bit check.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
# the CLI with the Metal backend (a no-op once built; a plain `-p intarsia-cli` build has no Metal backend)
# pinned: the first job builds it at the branch head of that moment, every later job measures that same binary
PIN=/opt/code/intarsiaml/.worktrees/mstore/target/mstore-run/intarsia-measure
if [ ! -x $PIN ]; then (cd /opt/code/intarsiaml/.worktrees/mstore && cargo build --release --locked -q -p intarsia-cli --features metal && cp target/release/intarsia $PIN.tmp && mv $PIN.tmp $PIN && git rev-parse --short HEAD > $PIN.rev && git status --short crates > $PIN.dirty) || { echo "build failed" >&2; exit 3; }; fi
W=/opt/code/intarsiaml/.worktrees/mstore; BIN=/opt/code/intarsiaml/.worktrees/mstore/target/mstore-run/intarsia-measure; R=$W/docs/results/2026-11-15-mstore-metal/raw/reuse; mkdir -p $R
case $1 in
  35b) M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=$W/golden/qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors; C=1024 ;;
  27b) M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; G=$W/docs/results/2026-11-02-m27-metal/raw/golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors; C=512 ;;
  *) echo "model?"; exit 2 ;;
esac
for arm in storeoff storeon; do
  E=(); [ $arm = storeon ] && E=(INTARSIA_PREFIX_STORE=on INTARSIA_MTP_STORE=1)
  echo "## $1 $arm $(date +%T) chunk $C"
  env X=1 $E $BIN accuracy --model $M --backend metal --golden $G --mode reuse --prefill-chunk $C --max-seq 8192 > $R/$1-$arm.out 2> $R/$1-$arm.err
  echo "exit $?"
  grep -a "spec\|prefix store\|MTP" $R/$1-$arm.err | head -5
  grep -a "PASS\|FAIL\|verdict\|judged\|SKIPPED" $R/$1-$arm.out | tail -8
done
