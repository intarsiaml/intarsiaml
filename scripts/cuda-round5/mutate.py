#!/usr/bin/env python3
"""Mutations of the pipelined split-KV attention (kernels/cuda/attention_splitkv.cu: splitkv_pipe_kernel,
combine_pipe_kernel and their launchers) for the mutation check of tests/splitkv_pipe.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list                  print the mutation names

Pattern of scripts/cuda-prefill4/mutate.py: every mutation is a deliberate bug the tests (or, for a dropped barrier, the
race check scripts/cuda-checks/racecheck.sh) must catch, except the ones marked EQUIVALENT, which cannot change any
stored bit. Only code of the new kernel is mutated (the base split-KV kernel is the reference the tests compare with).
"""
import sys

FILE = "kernels/cuda/attention_splitkv.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "no-wait": (
        "no cp.async.wait_group before a step's ring stage is read (the step may read bytes that have not landed)",
        [("        skv_wait<ST - 2>();  // this thread's copies of step `it` have landed\n", "", 1)],
    ),
    "wait-shallow": (
        "waits for one group too few (ST - 1 pending allowed: step `it` may still be in flight)",
        [("skv_wait<ST - 2>();  // this thread's", "skv_wait<ST - 1>();  // this thread's", 1)],
    ),
    "no-zero-fill": (
        "rows past the end are copied (from row 0) instead of zero-filled",
        [("    const int sz = valid ? 16 : 0;  // src-size 0 zero-fills the 16 bytes", "    const int sz = 16; (void)valid;", 1)],
    ),
    "past-end-rows": (
        "rows past the end are loaded from their own address (in the cache, past the visible keys) instead of zero-filled",
        [("            skv_cp16(st + u * kBlock + threadIdx.x, ok ? kb + j * D : kb, ok);\n"
          "            skv_cp16(st + (KB + u) * kBlock + threadIdx.x, ok ? vb + j * D : vb, ok);",
          "            skv_cp16(st + u * kBlock + threadIdx.x, kb + j * D, j < cap);\n"
          "            skv_cp16(st + (KB + u) * kBlock + threadIdx.x, vb + j * D, j < cap);", 1)],
    ),
    "issue-skip": (
        "the refill fetches step it + ST instead of it + ST - 1 (one step is never loaded)",
        [("        if (it + ST - 1 < niter) issue(it + ST - 1, wr);", "        if (it + ST < niter) issue(it + ST, wr);", 1)],
    ),
    "issue-same-stage": (
        "the refill overwrites the stage being read in this step",
        [("        if (it + ST - 1 < niter) issue(it + ST - 1, wr);", "        if (it + ST - 1 < niter) issue(it + ST - 1, rd);", 1)],
    ),
    "rd-rotation": (
        "the read stage wraps one stage early",
        [("        rd = rd + 1 == ST ? 0 : rd + 1;", "        rd = rd + 1 >= ST - 1 ? 0 : rd + 1;", 1)],
    ),
    "prologue-short": (
        "the prologue puts one step fewer in flight (the stage the first refill writes is read uninitialised)",
        [("    for (int s = 0; s < ST - 1; ++s) {\n        if (s < niter) issue(s, s);", "    for (int s = 0; s < ST - 2; ++s) {\n        if (s < niter) issue(s, s);", 1)],
    ),
    "ring-v-as-k": (
        "V rows are copied into the K slots of the stage",
        [("            skv_cp16(st + (KB + u) * kBlock + threadIdx.x, ok ? vb + j * D : vb, ok);",
          "            skv_cp16(st + u * kBlock + threadIdx.x, ok ? vb + j * D : vb, ok);", 1)],
    ),
    "key-swap": (
        "the KB keys of a step are read back in reverse order (wrong key for the validity mask and the softmax order)",
        [("            kk[u] = st[u * kBlock + threadIdx.x];\n            vv[u] = st[(KB + u) * kBlock + threadIdx.x];",
          "            kk[u] = st[(KB - 1 - u) * kBlock + threadIdx.x];\n            vv[u] = st[(KB + KB - 1 - u) * kBlock + threadIdx.x];", 1)],
    ),
    "issue-lane": (
        "a lane copies the rows of the next key lane",
        [("            const int64_t j = base + (int64_t)u * NG + gk;\n            const bool ok = j < end;",
          "            const int64_t j = base + (int64_t)u * NG + (gk + 1) % NG;\n            const bool ok = j < end;", 1)],
    ),
    "step-kb1": (
        "the 35B shape (group 8) steps one key per lane per step instead of 2 (another softmax grouping: other bits)",
        [("    if (G <= 8) { IK_PIPE_CASE(8, 2) }", "    if (G <= 8) { IK_PIPE_CASE(8, 1) }", 1)],
    ),
    "merge-sync-first": (
        "no barrier before a head's lane partials are written (the previous head's merge may still read them)",
        [("            __syncthreads();  // the ring / s_acc / s_m / s_l are reused for every head\n", "", 1)],
    ),
    "merge-sync-second": (
        "no barrier between writing the lane partials and merging them",
        [("            __syncthreads();  // every lane's partial is in s_acc / s_m / s_l\n", "", 1)],
    ),
    "combine-max-short": (
        "the combine's max skips the last split",
        [("#pragma unroll 8\n    for (int s = 0; s < nsplit_eff; ++s) M = fmaxf(M, ml[s].x);",
          "#pragma unroll 8\n    for (int s = 0; s < nsplit_eff - 1; ++s) M = fmaxf(M, ml[s].x);", 1)],
    ),
    "combine-sum-short": (
        "the combine's sums skip the last split",
        [("#pragma unroll 8\n    for (int s = 0; s < nsplit_eff; ++s) {\n        const float2 v = ml[s];",
          "#pragma unroll 8\n    for (int s = 0; s < nsplit_eff - 1; ++s) {\n        const float2 v = ml[s];", 1)],
    ),
    "combine-reverse": (
        "the combine sums the splits last to first (another fp32 order)",
        [("#pragma unroll 8\n    for (int s = 0; s < nsplit_eff; ++s) {\n        const float2 v = ml[s];",
          "#pragma unroll 8\n    for (int s0 = 0; s0 < nsplit_eff; ++s0) {\n        const int s = nsplit_eff - 1 - s0;\n        const float2 v = ml[s];", 1)],
    ),
    "EQ-no-final-wait": (
        "EQUIVALENT: no cp.async.wait_group 0 after the key loop (only empty groups can be pending there)",
        [("    skv_wait<0>();  // only empty groups can be pending here", "    // (no final wait)", 1)],
    ),
    "EQ-stages-3": (
        "EQUIVALENT: 3 ring stages by default instead of 4 for the 35B step size (the bits do not depend on the depth)",
        [("    const int st = env ? env : (KB >= 4 ? 2 : 4);", "    const int st = env ? env : (KB >= 4 ? 2 : 3);", 1)],
    ),
}


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print(" ".join(MUTATIONS))
        return 0
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        print(__doc__)
        return 2
    root, name = sys.argv[1], sys.argv[2]
    path = f"{root}/{FILE}"
    src = open(path).read()
    for find, repl, count in MUTATIONS[name][1]:
        if src.count(find) != count:
            print(f"mutation {name}: `{find}` occurs {src.count(find)} times, expected {count}")
            return 1
        src = src.replace(find, repl)
    open(path, "w").write(src)
    print(f"applied {name}: {MUTATIONS[name][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
