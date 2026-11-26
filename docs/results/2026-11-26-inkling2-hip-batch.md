# Inkling-2-Mini batched decode on HIP, one R9700 (lane ink-hip-batch, 2026-11-26)

Branch `ink/hip-batch` from main `87c9745c0`. Goal: `serve --max-batch N` (several streams decoding in one pass) works and is fast
for Inkling-2-Mini on the HIP backend. Today HIP refuses `--max-batch > 1` with `NoKernel`: no HIP kernel registers `.row_table()`
(`model/batch.rs` `row_tables_wanted`: Inkling-2 needs row-table kernels of `ConvMix` and `InkAttnDecode`, or of the pair).
Raw: `docs/results/2026-11-26-inkling2-hip-batch/raw/`.

## Briefs and messages received

- 2026-11-26 ~11:40 (coordinator): go ahead with the RT template parameter on `ink_row_kernel` / `ink_attn_dec_k`, not a copy.
  Two asks: (1) a test that pins the `RT = false` instantiations to main's bits: solo greedy md5 at 64 / 256 tokens, short / 4k,
  equal to main; (2) the `RowArgs` additions go at the end of the struct. Merge origin/main before the PR (pfflip will likely
  land first).
- 2026-11-26 ~11:45 (coordinator): `--mem` on tatooine-r9700 counts against the card's 32 GB, which radiance holds. Build #12035
  could never start and blocked every lane's queue. It was cancelled (`fleetq release 12035`) and resubmitted without `--mem`.
  No `--mem` on any tatooine job.

- 2026-11-26 ~12:00 (coordinator): two overlaps before the gate.
  1. ink-hip-spec also adds "hip" to `inkling2::ink_rows_exact`. Whichever lands second takes the other's line as is, without
     reordering. (This lane does not need that line: section 1.1.)
  2. PR #526 (HIP prefill flip) is merging. MoE WMMA prefill, flash prefill attention and chunk 2048 become the HIP Inkling
     defaults, and `tests/inkling2.rs` `backend()` "hip" pins `inkattn::set_flash(0)` and `inkmoe::set_wmma(0)`. Asks:
     - rebase onto main once it lands;
     - check that the batched rows (N rows at decode) never take the WMMA expert or flash attention path under the new defaults;
     - run the solo greedy md5 check (64 / 256, short / 4k) on the rebased binary with no env vars set;
     - say here which paths the batched forward took.
  Reminder: no `--mem` on tatooine jobs.

## 1. Design (written before any kernel)

### 1.1 What a batched Inkling-2 step needs on HIP

- `forward_batch` sets `exact_rows` for the whole pass (`Model::verifying()` is that flag, and `exec_call_one` copies it into every
  call). So every row-local op of the B-row pass takes the backend's `rows_exact` kernel where the op has one: `InkInProj`
  (`ink_in_proj_bf16_gemv_rows`), `Matmul` wo (`matmul_bf16_gemv_rows`), the router and unembed `Logits` (`logits_bf16_gemv_rows`),
  `MoeExperts` / `MoeExpertsMxfp8` (`_rows`, union-grouped). HIP already has all of them (used by the Qwen batching and verify).
  The ops without one (`AttnResMix`, `LoopBlend`, `MoeRoute`, norms, embedding, argmax) keep the solo kernel, one block per row.
  So `ink_rows_exact` (the core line lane ink-hip-spec owns) is **not needed** here, and this lane leaves it alone.
- The stateful ops run in their row-table form (`inkling2::ops`, "Row tables"): row r is the T = 1 call at device position
  `pos[r]` on sequence r's own rings and caches. In the HIP batched step these are `ConvMix` (twice per visit) and
  `InkAttnDecode` (op 75; on HIP the model takes op 75 at every decode visit, and a batch takes it whenever no row-table pair exists).
  `SconvRes` and the pair (`InkKvPrologue` + `RelSinkAttn`) are not called by the batched step, so they get no row-table kernel.
- New kernels: `conv_mix_bf16_rows` and `ink_attn_decode_bf16_rows`, registered `.row_table()`. With the `_rows` suffix they take
  the admission of the kernel they twin (`Registry::effective_admission`). Their gate is a bitwise test against the solo T = 1
  launches (`crates/intarsia-backend-hip/tests/ink_rows.rs`).

### 1.2 Kernel form: an `RT` template parameter on the solo kernels (Metal's pattern, HIP's `splitkv_kernel<.., ROWS>`)

- `ink_row_kernel<NE, FAST, DPP, RT = false>` (ConvMix). With RT, row t:
  - reads its ring from table entry `tab[t][index] + carrier offset`;
  - takes position `pos[t]`;
  - its conv history is itself (`qp == p` reads x row t) plus older positions from its own ring.
  The ring row is always written in-kernel (T = 1 rule), and `ring_after` never runs. The `[N, T, D]` planes, the mix and the
  blend (ctl GEMV, gate, norm: row-parallel) keep the call's global row index and T = B.
- `ink_attn_dec_k<GP, DPP, RT = false>` (op 75). With RT, block row r:
  - remaps its `InkP` to the T = 1 call of sequence r: row-indexed pointers shifted by r, its own scratch block, caches / rings
    from table entries `index .. index + 3` plus the carriers' offsets, `pos0 = max(pos[r], 0)`;
  - takes its ticket at global row r.
  The rows launcher always takes the fused kernel. The solo launcher's split path (full layers above `INTARSIA_INK_DEC_FUSED_KEYS`)
  gives the same bits as the fused kernel (both are bitwise the pair).
- The `RT = false` instantiations are the existing code. New fields go at the end of `RowArgs` / `InkP` only.
- Carrier offsets are passed from Rust (`TensorRef.offset` of each state tensor). An Inkling-2 ring slot puts both of its planes
  behind one table entry (`upload_row_table`: `(buf, buf)`), so plane 1 is reached only through its offset.
