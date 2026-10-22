#!/usr/bin/env python3
"""Lane H1 mutants of the HIP tp=2 decode-graph capture / replay logic and of fusion under TP (after G1c's
scripts/tp2-graphs/mutate.py; HIP tp=2 is in-process, one thread per rank). Applies one mutant to a COPY of the tree:
    mutate.py <tree dir> <name>
Each mutant must be caught by the window-3 gates (stages-w3.sh): 256 greedy ids against base tp=2 eager, rank 1's graph
counters (printed by rank 0 when the group goes away), or a GPU trap / timeout."""
import sys

MUTANTS = {
    # rank 1 replays without rewriting its device token id and position (the graph reads stale inputs)
    "no-peer-upload": ("crates/intarsia-core/src/model/graph_step.rs",
        """        self.backend.upload(&self.ws.ids, 0, &(id as i32).to_le_bytes(), 0)?;
        self.backend.upload(&self.ws.pos, 0, &(st.len as i32).to_le_bytes(), 0)?;""",
        """        if head {
            self.backend.upload(&self.ws.ids, 0, &(id as i32).to_le_bytes(), 0)?;
            self.backend.upload(&self.ws.pos, 0, &(st.len as i32).to_le_bytes(), 0)?;
        }"""),
    # rank 1's graph records one all-reduce fewer than rank 0's (the first collective of every capture is dropped)
    "drop-captured-allreduce": ("crates/intarsia-core/src/model.rs",
        "        comm.all_reduce_sum_into(partial, out, 0)?;",
        """        if self.tp.rank == 1
            && self.capture_ops.lock().unwrap().is_some()
            && self.capture_comms.load(std::sync::atomic::Ordering::Relaxed) == 0
        {
            self.capture_comms.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }
        comm.all_reduce_sum_into(partial, out, 0)?;"""),
    # the in-process peer ignores rank 0's graph decision and always runs the step eagerly
    "peer-ignores-flag": ("crates/intarsia-core/src/model.rs",
        """    fn run_peer(&self, ps: &mut SequenceState, ids: &[u32], graph: bool) -> Result<()> {
        if graph {""",
        """    fn run_peer(&self, ps: &mut SequenceState, ids: &[u32], graph: bool) -> Result<()> {
        if graph && false {"""),
    # fusion under TP also folds the GatedDeltaNet out-projection (it would write the block output and skip the reduce)
    "tp-fold-outproj": ("crates/intarsia-core/src/model.rs",
        "        self.tp_fusion && !matches!(op, Op::GatedRmsnormMatmulFp8)",
        "        self.tp_fusion"),
}

tree, name = sys.argv[1], sys.argv[2]
path, old, new = MUTANTS[name]
p = f"{tree}/{path}"
s = open(p).read()
assert s.count(old) == 1, f"mutant {name}: anchor found {s.count(old)} times in {path}"
open(p, "w").write(s.replace(old, new))
print(f"mutant {name} applied to {path}")
