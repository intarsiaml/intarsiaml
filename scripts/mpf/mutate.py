#!/usr/bin/env python3
"""Lane mpf mutants (deliberate bugs) of the MTP path. Applies one mutant to a tree:
    mutate.py <tree dir> <name>
Shared-logic mutants (the code CUDA runs too) must fail crates/intarsia-models/tests/mtp.rs (TEST=mtp scripts/mpf/mutants.sh);
default-mechanism mutants must fail tests/tp_mtp.rs (TEST=tp_mtp); cuda-* mutants are built for CUDA and must fail the ids gate."""
import sys

MUTANTS = {
    # the recurrent (GatedDeltaNet) state is not restored after a partial accept: the next pass continues from a rejected row's state
    "rollback-skip": ("crates/intarsia-core/src/state.rs",
        "                    backend.copy(&r.buf, 0, src, off, r.bytes, 0)?;",
        "                    let _ = (&r.buf, src, off);"),
    # the acceptance count includes the first rejected draft
    "accept-off-by-one": ("crates/intarsia-core/src/spec.rs",
        "    drafts.iter().zip(out).take_while(|(d, o)| d == o).count()\n",
        "    (drafts.iter().zip(out).take_while(|(d, o)| d == o).count() + 1).min(drafts.len())\n"),
    # a chained draft takes its hidden state from the first row of the head's output instead of the last (a wrong hidden-state tap)
    "mtp-chain-row": ("crates/intarsia-core/src/model/mtp.rs",
        "                            out.offset + (c - 1) * m.row_bytes,\n                            m.row_bytes,",
        "                            out.offset,\n                            m.row_bytes,"),
    # the head's own KV cache keeps the rejected rows after a rollback
    "mtp-kv-no-truncate": ("crates/intarsia-core/src/state.rs",
        "                m.kv.truncate(keep, backend)?;",
        "                let _ = backend;"),
    # queued hidden rows pair with the token of their own position instead of the next one
    "mtp-token-pairing": ("crates/intarsia-core/src/model/mtp.rs",
        "st.tokens.get(p0 + 1..(p0 + n + 1).min(st.len))",
        "st.tokens.get(p0..(p0 + n).min(st.len))"),
    # the recipe default ignores INTARSIA_SPEC_DEFAULT=0
    "default-optout-ignored": ("crates/intarsia-models/src/load.rs",
        "        && !std::env::var(\"INTARSIA_SPEC_DEFAULT\").is_ok_and(|v| v == \"0\")\n",
        "\n"),
    # the recipe default also turns on for --max-batch above 1 (MTP does not batch: the load would refuse)
    "default-batch-ignored": ("crates/intarsia-models/src/load.rs",
        "        && !opts.max_batch.is_some_and(|n| n > 1)\n",
        "\n"),
    # the recipe default ignores a configured prefix store
    "default-store-ignored": ("crates/intarsia-models/src/load.rs",
        "    if !mtp_default_allowed(opts, local) || (store_blocks && prefix_store_given(opts)) {",
        "    if !mtp_default_allowed(opts, local) {"),
    # the one-rank key answers for tensor parallelism too
    "default-world-any": ("crates/intarsia-models/src/load.rs",
        "    if world != 1 || !mtp_default_allowed(opts, local) {",
        "    if world < 1 || !mtp_default_allowed(opts, local) {"),
    # the macro prefill pass does not hand its hidden rows to the MTP drafter (lane mpf)
    "macro-head-skip": ("crates/intarsia-core/src/model.rs",
        "                    if st.mtp.is_some() {\n                        self.mtp_after_forward(st, &hn, total, true, profile)?;\n                    }\n                    ci += glen;",
        "                    ci += glen;"),
    # the KV-only head ingest writes its rows at position 0 instead of the cache length (lane mpf)
    "head-kvonly-pos": ("crates/intarsia-core/src/model/mtp.rs",
        "self.attn_block_kv_only(&head.attn, kv, &n, t, kv.len, false, profile)?;",
        "self.attn_block_kv_only(&head.attn, kv, &n, t, 0, false, profile)?;"),
    # the KV-only head ingest is used for the piece whose output the first draft needs (lane mpf)
    "head-kvonly-last": ("crates/intarsia-core/src/model/mtp.rs",
        "                if next.is_none() || done + c < n {",
        "                if true {"),
    # --- CUDA binary only (scripts/mpf/spark1.sh mutant): the verify attention goes back to the multi-row call ---
    "cuda-perrow-off": ("crates/intarsia-backend-cuda/src/device.rs",
        "        self.verify_per_row_pref.store(true, std::sync::atomic::Ordering::Relaxed);",
        "        self.verify_per_row_pref.store(false, std::sync::atomic::Ordering::Relaxed);"),
}

tree, name = sys.argv[1], sys.argv[2]
path, old, new = MUTANTS[name]
p = f"{tree}/{path}"
s = open(p).read()
if s.count(old) != 1 or old == new:
    sys.exit(f"mutant {name}: anchor found {s.count(old)} times in {path} (must be exactly 1): NOT APPLIED")
open(p, "w").write(s.replace(old, new))
print(f"mutant {name} applied to {path}")
