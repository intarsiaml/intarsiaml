#!/usr/bin/env python3
"""Mutations of the round-6 changes for the mutation check (scripts/cuda-prefill4/mutcheck.sh):

  * the 128-row flash blocks at head_dim 256 (kernels/cuda/attention_flash.cu), tests/flash_bq.rs + tests/attention.rs;
  * the w4a16 MoE threshold (crates/intarsia-backend-cuda/src/quantmoe.rs), the registry pin
    kernels::tests::moe_experts_selection_by_batch_size (`--lib`).

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list [file]           print the mutation names (only those of <file> if given)

Pattern of scripts/cuda-round5/mutate.py. A mutation whose description starts with EQUIVALENT cannot change any output bit
(the 64-row and 128-row blocks, and the two MoE kernels, are bitwise the same by construction): it is expected to survive
and is listed with the reason; its effect is on speed only, which the A/B measures.
"""
import sys

FL = "kernels/cuda/attention_flash.cu"
QM = "crates/intarsia-backend-cuda/src/quantmoe.rs"

# name -> (file, description, [(find, replace, count)])
MUTATIONS = {
    "bq-smem64": (FL, "the 128-row blocks get the 64-row dynamic shared memory size (Q rows 64..127 past the allocation)",
        [("    constexpr size_t smem = flash_smem_bytes<D, BK, BQ>();", "    constexpr size_t smem = flash_smem_bytes<D, BK>();", 1)]),
    "bq-threads64": (FL, "the 128-row blocks launch with 128 threads (warps 4..7, rows 64..127 of a block, never run)",
        [("<<<grid, 2 * BQ, smem, s>>>", "<<<grid, kThreads, smem, s>>>", 1)]),
    "bq-qstage-half": (FL, "only the first 64 Q rows of a block are staged into shared memory",
        [("        for (int idx = tid; idx < BQ * CHUNKS; idx += NT) {", "        for (int idx = tid; idx < kBQ * CHUNKS; idx += NT) {", 1)]),
    "bq-qlast64": (FL, "a block's key range ends at its 64th row (rows 64..127 miss their last keys)",
        [("    const int q_last = (q0 + BQ < T ? q0 + BQ : T) - 1;", "    const int q_last = (q0 + kBQ < T ? q0 + kBQ : T) - 1;", 1)]),
    "bq-mask-from": (FL, "the causal mask starts 64 keys late (tiles at the diagonal of the lower rows go unmasked)",
        [("    const int64_t mask_from = pos0 + q0 + 1; ", "    const int64_t mask_from = pos0 + q0 + 65;", 1)]),
    "bq-no-sync-k": (FL, "RACE: no barrier between the S = Q K^T reads of Ks and the copy of the next K tile into it",
        [("        __syncthreads();  // everyone is done reading Ks\n", "", 1)]),
    "EQ-bq-grid64": (FL, "EQUIVALENT: twice the blocks (grid from 64-row blocks); the extra blocks start past T, write nothing",
        [("    dim3 grid((T + BQ - 1) / BQ, N);", "    dim3 grid((T + kBQ - 1) / kBQ, N);", 1)]),
    "EQ-bq-default64": (FL, "EQUIVALENT: 64-row blocks by default (the test compares 64 with the default; same bits, slower)",
        [("    const bool bq64 = bqv && atoi(bqv) == 64;", "    const bool bq64 = !(bqv && atoi(bqv) == 128);", 1)]),
    "w4-1024": (QM, "the w4a16 MoE kernel from 1024 tokens again (the round-5 threshold)",
        [("pub const W4A16_MIN_TOKENS: usize = 768;", "pub const W4A16_MIN_TOKENS: usize = 1024;", 1)]),
    "w4-gt": (QM, "the w4a16 MoE kernel from 769 tokens (> instead of >=)",
        [(".when(|s| s.m >= W4A16_MIN_TOKENS && s.experts <= 1024", ".when(|s| s.m > W4A16_MIN_TOKENS && s.experts <= 1024", 1)]),
}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        f = sys.argv[2] if len(sys.argv) > 2 else None
        print(" ".join(m for m, v in MUTATIONS.items() if f is None or v[0].endswith(f)))
        return 0
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        print(__doc__)
        return 2
    root, name = sys.argv[1], sys.argv[2]
    fname, desc, edits = MUTATIONS[name]
    path = f"{root}/{fname}"
    src = open(path).read()
    for find, repl, count in edits:
        if src.count(find) != count:
            print(f"mutation {name}: `{find[:60]}` occurs {src.count(find)} times, expected {count}")
            return 1
        src = src.replace(find, repl)
    open(path, "w").write(src)
    print(f"applied {name}: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
