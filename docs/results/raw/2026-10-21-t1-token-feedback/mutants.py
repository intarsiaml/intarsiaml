#!/usr/bin/env python3
"""T1 mutants on the token-feedback rollback logic: each must make `cargo test -p intarsia-models --test token_feedback`
fail. Usage (inside a fleetq slot, from the worktree root): python3 mutants.py <out-dir>
`MUTANT_CARGO` (default `cargo`) prefixes the command, e.g. `deploy/tatooine/dev.sh cargo` and `--release`."""
import os, shlex, subprocess, sys

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
E = "crates/intarsia-core/src/engine.rs"
F = "crates/intarsia-core/src/model/feedback.rs"
ROLL = "let want = self.plen + self.generated - 1;\n        if st.len > want {\n            let t"
MUTANTS = [
    ("no-rollback", E, ROLL, ROLL.replace("if st.len > want", "if false && st.len > want")),
    ("rollback-keeps-extra-row", E, ROLL, ROLL.replace("self.generated - 1", "self.generated")),
    ("rollback-one-too-many", E, ROLL, ROLL.replace("self.generated - 1", "self.generated - 2")),
    ("no-token-patch", E, "                st.tokens[i] = tok;", "                let _ = (i, tok);"),
    ("speculate-at-max-tokens", E, "let next = if self.generated + 1 < self.max_new {", "let next = if self.generated < self.max_new {"),
    ("read-wrong-ring-slot", F, "p.slot * 4, &mut b, 0)", "(3 - p.slot) * 4, &mut b, 0)"),
    ("feed-stale-host-token", E, "Some(model.feedback_step(st, Feed::Slot(cur.slot), out, true)?)", "Some(model.feedback_step(st, Feed::Host(self.tok), out, true)?)"),
]
res = []
for name, path, old, new in MUTANTS:
    src = open(path).read()
    assert src.count(old) == 1, f"{name}: anchor absent or not unique"
    open(path, "w").write(src.replace(old, new))
    try:
        r = subprocess.run(shlex.split(os.environ.get("MUTANT_CARGO", "cargo")) + ["test", "-p", "intarsia-models", "--test", "token_feedback"], capture_output=True, text=True)
    finally:
        open(path, "w").write(src)
    log = r.stdout + r.stderr
    open(f"{OUT}/mutant-{name}.txt", "w").write(log)
    built = "test result" in log
    killed = built and r.returncode != 0
    failed = [l.split()[1] for l in log.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
    res.append((name, "KILLED" if killed else ("BUILD-FAIL" if not built else "SURVIVED"), ",".join(failed)))
    print(*res[-1], flush=True)
open(f"{OUT}/mutants.txt", "w").write("\n".join(" ".join(r) for r in res) + "\n")
print("killed", sum(r[1] == "KILLED" for r in res), "of", len(res))
