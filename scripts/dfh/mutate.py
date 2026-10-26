#!/usr/bin/env python3
"""Lane dfh mutants (deliberate bugs) of the DFlash2 path. Applies one mutant to a tree:
    mutate.py <tree dir> <name>
Each must be caught by crates/intarsia-models/tests/dflash2_cpu.rs (mutants.sh runs it and requires a failure)."""
import sys

MUTANTS = {
    # the draft taps the layer after the one it names (a tap off by one layer: drafts only, the ids gates cannot see it)
    "tap-off-by-one": ("crates/intarsia-core/src/model.rs",
        "if let Some(ti) = head.taps.iter().position(|&x| x == li) {",
        "if let Some(ti) = head.taps.iter().position(|&x| x + 1 == li) {"),
    # the recurrent rollback restores the state saved one verify row too late (a wrong rollback after a partial accept)
    "rollback-next-row": ("crates/intarsia-core/src/state.rs",
        "            let src = &sr.bufs[m - 1];",
        "            let src = &sr.bufs[if m < sr.saved { m } else { m - 1 }];"),
    # the sliding window keeps one context key too many in every block row
    "window-off-by-one": ("crates/intarsia-core/src/model/dflash2.rs",
        "let win = if d2.window == 0 { 0 } else { d2.window + t - 1 - r };",
        "let win = if d2.window == 0 { 0 } else { d2.window + t - r };"),
    # the draft's norms use the target recipe's zero-centred scale (1 + w) instead of standard RMSNorm
    "draft-norm-offset": ("crates/intarsia-core/src/model/dflash2.rs",
        "OpAttrs { eps: d2.eps, w_offset: 0.0, ..Default::default() }, profile)\n    }",
        "OpAttrs { eps: d2.eps, w_offset: 1.0, ..Default::default() }, profile)\n    }"),
    # the selector scores candidates with the predecessor code of the candidate instead of the previous draft
    "selector-pred": ("crates/intarsia-core/src/model/dflash2.rs",
        "            let pc = &self.pred[prev * r..(prev + 1) * r];",
        "            let pc = &self.pred[top[0].1 * r..(top[0].1 + 1) * r];"),
    # tensor parallelism: the peers keep the rejected rows' recurrent state (only rank 0 rolls back)
    "tp-peer-no-rollback": ("crates/intarsia-core/src/state.rs",
        "            p.st.spec_rollback_rank(keep, &*p.backend)?;",
        "            let _ = &p.st;"),
}

tree, name = sys.argv[1], sys.argv[2]
path, old, new = MUTANTS[name]
p = f"{tree}/{path}"
s = open(p).read()
if s.count(old) != 1 or old == new:
    sys.exit(f"mutant {name}: anchor found {s.count(old)} times in {path} (must be exactly 1): NOT APPLIED")
open(p, "w").write(s.replace(old, new))
print(f"mutant {name} applied to {path}")
