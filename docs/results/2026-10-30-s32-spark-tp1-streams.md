# S32: Spark tp1 35B past 16 streams (code finding, no hardware run)

Date 2026-10-30, base `origin/main` d4936a3d4. Question: does Qwen3.6-35B-A3B NVFP4 at tp=1 on one DGX Spark (CUDA) keep
scaling past 16 concurrent streams? Planned: aggregate decode at 8/16/24/32 streams with `--max-batch 32`.

## Result

**No measurement was made.** Reading the code, the CUDA engine cannot batch more than 16 sequences, so 24 and 32 streams
cannot run as concurrent decode on current main. The refusal was **not tried on hardware**: spark1 was held exclusively by
fleetq job #7261 (another chat, `Ling-3.0-flash baseline`, expected 1h50m, still running overdue at 4h07m when this lane
stopped), so neither the CUDA build nor the probe got a slot. The error text below is therefore read from the source, not
captured from a run.

The current Spark tp1 35B line stays as measured by SCORE29 (10-29, `--max-batch 16`, MTP off in the batched arm):
1: 80.8, 4: 174.2, 8: 232.5, 16: 300.9 tok/s aggregate (8 -> 16 is still +29 %). Whether it keeps scaling past 16 is open.

## Where the cap of 16 lives

`--max-batch N` on CUDA is checked in this order; the first test that fails refuses the flag before any weight is read.

| # | place | what |
|---|---|---|
| 1 | `crates/intarsia-backend-cuda/src/device.rs:550` | `Backend::batch_row_cap()` returns `16` ("the `_rows` GEMV / MoE kernels are proven row-exact up to M = 16") |
| 2 | `crates/intarsia-models/src/load.rs:510-511` | passes that cap as `supported` to `Recipe::resolve_max_batch` whenever `--max-batch` is given |
| 3 | `crates/intarsia-core/src/recipe.rs:518-526` | `Some(n) if n > supported` returns `Error::Unsupported("--max-batch {n} is not implemented yet (the engine decodes {supported} sequence(s) at a time; ...); use --max-batch {supported}")` |
| 4 | `crates/intarsia-core/src/scheduler.rs:48,290,296` | `MAX_BATCH_CAP = 16`; `SchedulerConfig::validate` rejects `max_batch` outside `1..=16` (and `spec_off_at` above 17) |
| 5 | `crates/intarsia-core/src/model/batch.rs:23,157` | `Model::BATCH_MAX_ROWS = 16`; `enable_batch` sizes at `min(backend cap, BATCH_MAX_ROWS, LOGIT_ROWS, ws.max_tokens)`; the row tables, batch token/position buffers and top-k sampling buffers are allocated for `BATCH_MAX_ROWS` rows |
| 6 | `crates/intarsia-core/src/model.rs:130` | `LOGIT_ROWS = 16` (logits workspace is `LOGIT_ROWS * vocab * 4` bytes); `VERIFY_MAX_ROWS` derives from it |
| 7 | `crates/intarsia-backend-cuda/src/kernels.rs:74,77` | `ROWS_MAX = 16` and `SPLITKV_MAX_ROWS = 16`: the registry selects the row-exact GEMV / MoE / KV-append / split-KV attention kernels only for `1..=16` rows (`quantmoe.rs` uses the same `ROWS_MAX`) |
| 8 | `kernels/cuda/fp8.cu:178`, `kernels/cuda/matmul.cu:227-231` | device side: `kRowsMax = 16`, and `gemv_launch` returns `IK_ERR_SHAPE` for `M > max_rows` (16 for the row-exact twin) |
| 9 | `crates/intarsia-backend-cuda/tests/rows_exact.rs:28` | the bitwise row-exactness gate (`CAP = 16`) only proves rows 1..=16 |

The recipe's `max_batch = 32` (`recipes/qwen3_5_moe.toml:66`) is not the limit. It is a declared ceiling "that the engine
cannot do yet" (comment at `recipe.rs:514-516`); with `--max-batch` given, row 3 refuses first, and without it the batched
engine is not used at all (`supported` is 1). The `--max-batch` doc comment at `crates/intarsia-cli/src/cmd/serve.rs:22`
("CUDA and HIP 2..=16") is accurate.

Expected refusal for `--max-batch 24` or `32` on CUDA (from row 3, not captured; the error-kind prefix the CLI prints is unverified): `--max-batch 32 is not
implemented yet (the engine decodes 16 sequence(s) at a time; continuous batching is milestone M1 of
docs/research/continuous-batching-design.md); use --max-batch 16`.

## What lifting the cap to 32 would take (engine work, not done here)

1. Raise rows 1, 4, 5, 6 together (`batch_row_cap`, `MAX_BATCH_CAP`, `BATCH_MAX_ROWS`, `LOGIT_ROWS`); the buffers sized by them
   grow linearly (logits `LOGIT_ROWS * vocab * 4` bytes, row tables, sampling buffers).
2. Raise `ROWS_MAX` / `SPLITKV_MAX_ROWS` and the device-side row templates (`fp8.cu`, `matmul.cu`, the NVFP4 / MoE decode rows
   kernels, split-KV attention); check register and shared-memory use of the wider row instantiations on GB10, since the row
   count is a template parameter and more accumulators per thread may spill.
3. Re-prove bitwise row-exactness for 17..=32 rows (`rows_exact.rs`, plus the batch consistency gates in
   `docs/results/2026-10-23-rb2-cuda-batching.md`). The batched engine's contract is that each sequence's output is
   identical to solo decoding, so a kernel that is faster but not row-exact would not qualify.
4. Memory: KV for 32 sequences at the chosen `--max-seq` plus per-sequence recurrent state and snapshots
   (`enable_batch` charges them against the KV budget); with `--max-seq 8192` and `--mem-budget 40` as in SCORE29 this
   was expected to fit, but that was not checked.

Open question this leaves: the +29 % gain from 8 to 16 streams on Spark (against +4-6 % on R9700 and Mac) suggests headroom past 16,
but no number exists for it; a 24/32 measurement needs the engine change above first.

## fleetq and downtime

- #7269 (CUDA CLI build at d4936a3d4, `--load cpu`): queued behind #7261, then its client was killed by another lane
  at about 01:30; it never ran.
- #7328: the same build requeued; waited about two hours behind #7261, then released by this lane (cancelled, never ran).
- #7261 is not ours and was not touched. spark1's vLLM (flight, guidance, capcom) was already down because of it; this lane
  never stopped or started it, so there is no downtime or restore of ours to report.
- No binary was built, so there is no binary stamp.
