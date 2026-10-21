#!/bin/zsh
# G3 mutant gate for the first-commit change (sync / order logic), NOT YET RUN (endor hold 2026-10-21 11:45).
# Each mutant edits crates/intarsia-backend-metal/src/lib.rs, runs the fill tests on the GPU, and restores the file;
# every mutant must make at least one test fail. Run inside one exclusive endor slot from the worktree root:
#   fleetq run --host endor --exclusive --mem 24 --label "G3 mutants" -- zsh docs/results/raw/2026-10-21-g3-metal/mutants.sh
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
W=/opt/code/intarsiaml/.claude/worktrees/agent-a34836167f72ea687
F=$W/crates/intarsia-backend-metal/src/lib.rs
O=$W/docs/results/raw/2026-10-21-g3-metal/mutants; mkdir -p $O
cp $F $O/lib.rs.orig
trap 'cp $O/lib.rs.orig $F' EXIT
typeset -A MUT
MUT[no-fresh-on-sync]='s/^        st\.fresh = true;$/        \/\/ mutant/'
MUT[no-fresh-clear-on-flush]='s/^            st\.fresh = false;$/            \/\/ mutant/'
MUT[gt-instead-of-ge]='s/st\.dispatches >= st\.first_commit;/st.dispatches > st.first_commit;/'
MUT[first-commit-zero]='s/^pub const FIRST_COMMIT: usize = 4;/pub const FIRST_COMMIT: usize = 0;/'
cd $W
for m in ${(k)MUT}; do
  cp $O/lib.rs.orig $F
  sed -i '' -e "${MUT[$m]}" $F
  if cmp -s $O/lib.rs.orig $F; then echo "$m: MUTATION DID NOT APPLY"; continue; fi
  cargo test -p intarsia-backend-metal --release --test fill > $O/$m.txt 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "$m: killed (tests failed, rc $rc)" || echo "$m: SURVIVED"
done
cp $O/lib.rs.orig $F
echo MUTANTS_DONE
