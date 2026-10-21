#!/usr/bin/env bash
# Lane G1b: mutants for the tp=2 order / sharding logic (docs/design/35b-tp2.md), on the Mac against
# crates/intarsia-models/tests/tp_hybrid.rs. Works on an exported copy of HEAD (never the working tree):
#   scripts/35b-tp2/mutants.sh <scratch dir>
# Each mutant is one perl substitution; a mutant must make the test binary fail (or hang: alarm 300 s kills it).
set -uo pipefail
W=${1:?scratch dir}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
M=$W/tree
rm -rf "$M" && mkdir -p "$M"
git -C "$REPO" archive HEAD | tar -x -C "$M"
cd "$M" || exit 1
run() { # name file perl-expr
  local name=$1 f=$2 expr=$3
  cp "$f" "$f.orig"
  perl -0pi -e "$expr" "$f"
  if [[ $name != clean ]] && cmp -s "$f" "$f.orig"; then echo "MUTANT $name: NOT APPLIED"; mv "$f.orig" "$f"; return; fi
  CARGO_TARGET_DIR=$W/target perl -e 'alarm 300; exec @ARGV' cargo test --release -q -p intarsia-models --test tp_hybrid > "$W/mut-$name.log" 2>&1
  local rc=$?
  local res; res=$(grep -E "test result" "$W/mut-$name.log" | tail -1)
  echo "MUTANT $name: rc=$rc ${res:-no result (killed by the alarm, or a compile error)}"
  mv "$f.orig" "$f"
}
S=crates/intarsia-core/src/state.rs
MD=crates/intarsia-core/src/model.rs
T=crates/intarsia-core/src/tp_remote.rs
G=crates/intarsia-core/src/model/gdn_block.rs
run clean "$S" 's/zzzz_never//'
run no-remote-snapshot "$S" 's/r\.snapshot\(keep\)\?;/let _ = r;/'
run no-state-truncate "$S" 's/r\.truncate\(len\)\?;/let _ = (r, len);/'
run no-resync-truncate "$MD" 's/rs\.truncate\(st\.len\)\?;/let _ = 0;/'
run rank0-before-send "$MD" 's/rs\.start_forward\(ids\)\?;\n(\s*)let r0 = self\.forward_inner\(st, ids, mode, on_row\);/let r0 = self.forward_inner(st, ids, mode, on_row);\n$1rs.start_forward(ids)?;/'
run gdn-no-reduce "$G" 's/self\.proj_reduce\(&core\.reshaped\(&\[t, vdim\]\), &g\.out, out, t, profile\)/self.proj(&core.reshaped(&[t, vdim]), &g.out, out, profile)/'
run moe-no-reduce "$MD" 's/self\.all_reduce\(&blkv, &blkv, profile\)\?;/let _ = 0;/'
run serve-ignores-truncate "$T" 's/if let Err\(e\) = st\.truncate\(len, &\*model\.backend\)/if let Err(e) = Ok::<(), crate::error::Error>(())/'
run gdn-groups-as-rows "$MD" 's/Some\(qkv_groups\)\)\?,/Some(ShardAxis::Rows))?,/'
