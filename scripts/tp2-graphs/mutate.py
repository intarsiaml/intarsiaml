#!/usr/bin/env python3
"""Lane G1c mutants of the tp=2 decode-graph capture and replay logic. Applies one mutant to a COPY of the tree:
    mutate.py <tree dir> <name>
Each mutant must be caught by the cross-host gates (scripts/tp2-graphs/xg.sh MODE=mut): greedy ids against tp=2 eager,
a second prompt in the same process, or a hang caught by the timeout."""
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
    # a recording is kept across sequence states (replays another state's KV / recurrent buffers)
    "no-rekey": ("crates/intarsia-core/src/model/graph_step.rs",
        "if cache.as_ref().map(|c| c.state_id) != Some(st.id) {",
        "if cache.is_none() {"),
    # the NCCL f32 landing buffer is not reserved at load: it grows (moves) after a graph captured its address
    "no-reserve": ("crates/intarsia-core/src/model.rs",
        "            c.reserve(mt * dims.hidden, 0)?;",
        "            let _ = c;"),
    # the remote rank ignores rank 0's graph decision and always runs eagerly
    "peer-ignores-flag": ("crates/intarsia-core/src/tp_remote.rs",
        "match model.forward_rank(st, &ids, flags & FWD_GRAPH != 0) {",
        "match model.forward_rank(st, &ids, flags & FWD_GRAPH != 0 && false) {"),
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
}

tree, name = sys.argv[1], sys.argv[2]
path, old, new = MUTANTS[name]
p = f"{tree}/{path}"
s = open(p).read()
assert s.count(old) == 1, f"mutant {name}: anchor found {s.count(old)} times in {path}"
open(p, "w").write(s.replace(old, new))
print(f"mutant {name} applied to {path}")
