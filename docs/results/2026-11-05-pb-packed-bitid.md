# pb: CUDA packed prefill not bitwise for 16..31-row prompts (routed-experts kernel switch at 32 rows)

Lane `lane-pb/packed-bitid`, 2026-11-05. Base: `origin/main` 5390e65b4. Finding from lane bm
(`docs/results/2026-11-05-batched-mtp.md` on `lane-bm/batched-mtp`, raw `raw/gates-cuda3/`).

## Symptom (lane bm)

On main 602a1fcfe, `intarsia serve --backend cuda --max-batch 16 --max-seq 8192 --mem-budget 40`, Qwen3.6-35B-A3B NVFP4 tp1, six greedy
prompts submitted at once (`scripts/pb/ids.py`, a copy of lane bm's `scripts/bm/ids.py`): prompt 2 ("List the first twenty prime
numbers ...") differed from the solo spec-off server in 2 of 4 runs, both times identically: token 2 is 10033 instead of 3300
(ids `[7054, 11, 10033, ...]` vs `[7054, 11, 3300, ...]`). With `INTARSIA_PACK_PREFILL=0` every run matched.

## Cause (static analysis; GPU confirmation pending, see Status)

1. **Prompt lengths.** Tokenized with the checkpoint's `tokenizer.json`: 15 / 103 / **17** / 201 / 15 / 82 tokens. A packed pass carries
   each prompt's body (all but the last token, which runs alone with the head), so the bodies are 14 / 102 / **16** / 200 / 14 / 81 rows.
   `Model::PACK_MIN_ROWS` = 16, so prompts 0 and 4 never pack and prompt 2 is the only body in 16..31 rows. Whether it packs depends
   on arrival timing (the scheduler packs prompts waiting together, at most half the queue), hence the intermittency.
2. **What runs over the whole pass.** `Model::forward_packed` runs attention, KV append and the GatedDeltaNet recurrence per prompt, and
   on CUDA (`Backend::pack_split_gemms`) the dense GEMMs of `macro_split_ops()` (`Matmul`, `MatmulFp8`, `MatmulFp8Pair`, `MatmulPair`)
   per prompt too. Everything else, including the routed experts (`Op::MoeExperts`) and the NVFP4 shared expert
   (`Op::MatmulNvfp4`), runs ONCE over all packed rows.
3. **The kernel switch.** `crates/intarsia-backend-cuda/src/quantmoe.rs:95` `MMA_MIN_TOKENS = 32`: from 32 rows the grouped
   tensor-core `moe_experts_mma_bf16` serves the routed experts, below it the warp-per-row kernels. `docs/quant-moe.md` (batch
   invariance of `moe_experts`): the mma kernel "accumulates in tensor-core order" and differs from the single-token kernels by up to
   3.5e-4 rel_l2. So prompt 2 alone (16 rows) gets the warp kernel; packed beside anything (16 + 81 = 97 rows or more) it gets mma,
   and its hidden rows, KV and recurrent state get different bits. The last prompt token's argmax survives the small difference
   (token 1 matched), token 2 does not.
4. **Why the gates missed it.** `PACK_MIN_ROWS = 16` came from HIP (`crates/intarsia-backend-hip/tests/packed_rows.rs`: a 15-row segment
   differs, 16 and up do not, on HIP's kernels). CUDA's own gate `real_35b_packed_prefill_equals_the_local_engine` has its shortest
   prompt at 33 tokens, a 32-row body: exactly `MMA_MIN_TOKENS`, so every CUDA segment it packed took mma both alone and packed. The
   served identity checks (srv16) used 300..1300-token prompts.
5. **Other row-count switches on the shared path** (not split): `matmul_nvfp4` (shared expert) serves gemv to 8 rows, `*_mma` below
   33 (`GEMM_MIN_TOKENS`), `*_gemm` from 33, `*_gemm_reg` from 256; the gemm kernels are documented as the mma kernel's numerics
   (`kernels/cuda/nvfp4_gemm.cu` header) and the srv16 gate (32-row body packed into >= 256-row passes) passed through those
   switches. `moe_experts_w4a16_bf16` (from 768 rows) is documented bitwise `moe_experts_mma_bf16`. Router top-k, norms, rope and
   the combine are one warp / block per row in a fixed order (no atomics on outputs).

Same bound, second place: the macro pass (`Model::macro_plan`, several chunks of ONE prompt in one pass, default 8192 rows on CUDA)
takes a trailing chunk of at least `PACK_MIN_ROWS` rows into the pass; for a prompt whose last chunk body is 16..31 rows (length
2048k + 17..32) the plain schedule runs that chunk on the warp kernel, the macro pass on mma. Same mechanism, not yet measured.

## Fix (branch `lane-pb/packed-bitid`)

`Backend::pack_min_rows()` (default 16 = `Model::PACK_MIN_ROWS`, HIP's measured bound); the CUDA backend returns `MMA_MIN_TOKENS` (32).
`Model::pack_min_rows()` reads it (override `INTARSIA_PACK_MIN_ROWS=<n>`, for bisecting) and replaces the constant in
`forward_packed`'s check, `macro_plan` and the batched engine's `BatchModel::pack_min_rows`. On CUDA a prompt of up to 32 tokens
now prefills alone instead of packing (one extra 16..31-row pass per such prompt: small next to a pack), and a macro pass leaves a
16..31-row trailing chunk to the plain schedule. HIP and Metal unchanged.

Not chosen: splitting `Op::MoeExperts` per prompt (`INTARSIA_PREFILL_MACRO_SPLIT`) would run the experts once per prompt and give
up most of the packing gain (and `exec_call` splits only input 0 and the outputs, not the top-k ids / weights inputs); forcing the
mma kernel on short solo chunks would change every short prompt's bits against main.

## Probes (GPU, fleetq #8665 on spark1, queued)

`crates/intarsia-models/tests/batch_cuda_real.rs` (ignored, `INTARSIA_QWEN36_DIR`):
- `real_35b_packed_short_segment_probe`: short body 16 / 17 / 20 / 24 / 31 / 32 / 33 / 48 / 64 / 128 rows packed with a 16 / 81 / 200-row
  partner, first and second in the pack; logits of the last prompt row and 8 greedy steps vs solo, bit for bit, both prompts.
- `real_35b_macro_trailing_chunk_probe`: two 2048-row chunks plus a 16 / 20 / 31 / 32 / 40-row trailing body, macro 8192 vs plain.

`scripts/pb/probe.sh` runs both with `INTARSIA_PACK_MIN_ROWS=16` (main's bound: expected DIFF exactly for short bodies 16..31 and
the 16 + 16 pack), again with `INTARSIA_CUDA_DISABLE=moe_experts_mma_bf16` (expected: all equal, which pins the kernel), and with the
fix's default (expected: all run cases equal, 16..31 skipped); then 6 served runs each of main 602a1fcfe and this branch with
`INTARSIA_CB_TRACE=1`, ids vs solo spec-off, and which prompts packed together per run (`scripts/pb/packs.py`).

## Status

Fix and probes built and CPU-checked (`cargo check`, `intarsia-core --test scheduler` 40 passed); `cargo test --workspace` queued on
endor. GPU window #8665 not started yet (spark1 queue; estimated Fri 04:25).
