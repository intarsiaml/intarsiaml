#!/bin/zsh
# T1 builds + CPU checks, one non-exclusive endor slot (--mem 8): base = origin/main (base.tar), dev = this branch.
# usage (inside slot, from anywhere): zsh build.sh <out-dir>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
O=$1; mkdir -p $O
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/t1
W=/opt/code/intarsiaml/.claude/worktrees/agent-a06622c5d21b6b82d
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
echo "== base build $(date -Iseconds)"
( cd $S/base-src && tar -xf $S/base.tar && cargo build --release -p intarsia-cli --features metal > $O/build-base.txt 2>&1 ) \
  && cp $S/base-src/target/release/intarsia $S/intarsia-base; echo "base build exit $?"
echo "== dev build $(date -Iseconds)"
( cd $W && cargo build --release -p intarsia-cli --features metal > $O/build-dev.txt 2>&1 ) \
  && cp $W/target/release/intarsia $S/intarsia-dev; echo "dev build exit $?"
md5 $S/intarsia-base $S/intarsia-dev | tee $O/md5.txt
echo "== cargo check (cpu, metal) $(date -Iseconds)"
( cd $W && cargo check --workspace --tests > $O/check-cpu.txt 2>&1 ); echo "check workspace exit $?"
( cd $W && cargo check --workspace --tests --features intarsia-cli/metal > $O/check-metal.txt 2>&1 ); echo "check metal exit $?"
echo "== cpu tests (no GPU crates) $(date -Iseconds)"
( cd $W && cargo test --workspace --exclude intarsia-backend-metal > $O/test-cpu.txt 2>&1 ); echo "cpu tests exit $?"
grep -h "test result" $O/test-cpu.txt | awk '{p+=$4; f+=$6} END {print "total passed", p, "failed", f}'
echo BUILD_DONE $(date -Iseconds)
