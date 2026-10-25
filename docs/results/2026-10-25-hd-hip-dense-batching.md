# hd: dense Qwen3-8B batched decode on HIP (lane hd, 2026-10-25)

Branch `lane-hd/hip-dense-batching` (base `main` 70ba3d22). The guard that refused `--max-batch N` (N > 1) for dense models on HIP
(`hip_batch_needs_hybrid`, commit 1ff822ac9, LOG row 2026-10-23 15:00) is removed, with its unit test `hip_guard_tests`. No kernel or
engine code changes: the dense batched step was never stuck. Lane hb's run of the test was stopped before the test could finish
(section "Why lane hb saw no output", with a control run on the tree of that day).

## Result

`real_qwen3_8b_batched_decode_is_bitwise_the_solo_step` (`crates/intarsia-models/tests/batch_cuda_real.rs`, assertions unchanged) on HIP,
tatooine card 0 (R9700), Qwen3-8B bf16, guard removed, fleetq #5345:

- **17,550 (row, step) pairs bitwise identical**: every N from 2 to 16, 130 consecutive steps each. The logits match bit for bit, the
  on-device argmax token matches token for token, and the state is identical.
- **577 s** for the whole test (571.6 s in the gate). Raw: [raw/2026-10-25-hd/run-1025-131205.log](raw/2026-10-25-hd/run-1025-131205.log).

| N | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate time for 130 steps (s) | 10.7 | 13.7 | 17.2 | 20.5 | 24.1 | 27.5 | 31.0 | 47.6 | 41.6 | 47.7 | 48.9 | 53.9 | 56.8 | 61.6 | 67.6 |
| batched step (ms) | 26.7 | 27.2 | 27.4 | 27.6 | 28.2 | 28.5 | 29.3 | 130.8 | 57.9 | 78.2 | 60.5 | 72.5 | 68.5 | 78.3 | 97.6 |
| its N solo steps (ms) | 55.3 | 77.7 | 103.6 | 129.3 | 155.4 | 181.2 | 207.4 | 233.1 | 259.4 | 285.6 | 312.1 | 338.5 | 364.9 | 391.5 | 418.4 |

Card 0 VRAM held steady at 25.4 GB (16.4 GB weights; 32 sequence states of 2,048 tokens of KV, 4.8 GB; workspace). GTT stayed at
20 MiB, so nothing spilled to host memory.

## Why lane hb saw "no output"

The gate prints nothing until it ends, and on HIP it runs every N from 2 to 16. That is 135 rows x 130 steps = 17,550 solo reference
steps of the dense 8B. Each one streams the 16.4 GB of bf16 weights at about 26 ms (about 630 GB/s, the card's bandwidth), so the solo
reference alone takes about 460 s. The batched steps take another 110 s. Lane hb stopped the test at 8 minutes (480 s, fleetq #4592),
before it could end. The 188% CPU it reported fits a host that is busy launching and waiting on the GPU, not a stuck host loop. For comparison, the
hybrid 35B gate, with the same pair count, finishes in 205 s: about 3B active parameters in NVFP4 make its solo step several times
cheaper.

Control run, so that the result does not depend on later HIP work (row tables, batch graphs, new rows kernels). The tree the guard
was added to (f3b9f19e5, the parent of 1ff822ac9, no guard yet) ran the same test unchanged, with a 25-minute cap (fleetq #5363):
**17,550 pairs bitwise identical in 606.7 s**, test 612.8 s. Raw: [raw/2026-10-25-hd/control-f3b9f19e5.log](raw/2026-10-25-hd/control-f3b9f19e5.log).
So the dense batched step worked on HIP when the guard went in. Lane hb's 480 s was simply shorter than the test.

The gate now prints a line per batch size (time for its 130 steps, the batched and the solo ms per step), the prefill time and the
batch KV plan (`common/batch_identity.rs`, `batch_cuda_real.rs` `load_with`). A long run is then visibly progressing. These are
progress lines only; no assertion changed.

## What changes for users

`serve --max-batch N` (2..=16) on HIP now batches dense models too, opt-in as before. Without `--max-batch` nothing changes. The dense
batched step runs the same kernels the gate proves bitwise: row tables, a graph per row count, `matmul_bf16_gemv_rows` and
`logits_bf16_gemv_rows`. The busy prefill chunk for a dense model stays 256. Prefill in a busy engine is therefore bitwise the solo
prefill only where the chunk schedules coincide, the same as dense on CUDA (`docs/tuning.md`, Batch row).

## Open

- **Batched dense step above 8 rows:** 29.3 ms at N = 8, then 57.9 to 130.8 ms at N = 9..16 (97.6 ms at 16). So 16 rows decode
  164 tok/s in the model loop against 273 at 8 rows. This is the same register cliff lane mr found and fixed for
  `logits_bf16_gemv_rows` (`gemv_kernel<M, 2, 4>` spills from M = 9, docs/results/2026-10-25-mr-hip-logits-rows.md).
  `matmul_bf16_gemv_rows`, which carries every dense projection, still uses that template. It is a speed lever, not a correctness
  issue: the rows stay bitwise. N = 9 is slower than N = 10; that is not explained.
- Not measured here: served speed of the dense 8B at 4 and 16 streams on HIP, and an engine-level identity test for the dense model
  (the 35B has `real_35b_batched_engine_equals_the_local_engine`; there is no 8B twin on any backend).
