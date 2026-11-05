# fh: packed multi-prompt prefill on by default on HIP

Lane `lane-fh/hip-pack-default` (branch off origin/main `602a1fcfe`), 2026-11-05. Qwen3.6-35B-A3B NVFP4, `serve --backend hip --max-batch 16`.
**Status: done (code, docs, endor gates, tatooine A/B at tp=1 and tp=2, GPU identity gates). Branch pushed; the coordinator opens the PR.**

## 1. Decisions and approvals

- User, 2026-11-05 17:06 EST, relayed in the lane brief: "flip 2". Scope: packed multi-prompt prefill becomes the default on HIP for the hybrid MoE, as it already is on CUDA (SRV16, `docs/decisions/LOG.md`).
- User approval, relayed by coordinator (2026-11-05, about 17:06 EST; the first relay said 17:08): "radiance down OK, start 2, keep building". It covers taking radiance down on tatooine for phase B. The lane first held phase B because the earlier relay was an agent message; the coordinator then repeated the user's words and the user's project memory note "Perf push 11-05" (17:08 EST: "deferrals lifted; flips 1+2, radiance down OK") says the same, so phase B went ahead.

## 2. Change

- `crates/intarsia-backend-hip/src/device.rs`: `HipBackend::pack_prefill_default()` returns `true`. `BatchedEngine` start-up (`crates/intarsia-core/src/engine/batch.rs`, `cfg.pack_prefill = model.backend.pack_prefill_default()` when `INTARSIA_PACK_PREFILL` is unset) does the rest. `INTARSIA_PACK_PREFILL=0` turns it off, `=1` forces it on.
- Doc comments in `backend.rs` and `batch.rs` now say CUDA and HIP. `docs/tuning.md` (the `INTARSIA_PACK_PREFILL` row and the Batch rows), `docs/decisions/LOG.md` (a new row quoting the approval, the 10-25 open-decision row for this question removed per the log's rule 6, and the HIP batched-decode line of "Defaults in force on main").

"Mirroring CUDA's conditions": CUDA returns `true` unconditionally; the narrowing is done by the scheduler, so HIP gets the same shape. A pass only forms when all of these hold (`Model::pack_capable`, `BatchEngine::pack_rows_max`, `Model::PACK_MIN_ROWS`):

- every sequence state has recurrent state (the hybrid 35B; dense models on HIP never pack);
- one rank (`tp.world == 1`): **`--tp 2` never packs, so the flip changes nothing at tp=2**; the tp=2 arm of phase B is therefore a no-op check (A should equal B within noise, zero packed passes);
- no Gemma layers, no plan recording, no rope table, no MTP head on the state, every segment at least 16 rows, the pass within the workspace and within `INTARSIA_PACK_ROWS` (3072 rows by default) and at most half the waiting prompts (`INTARSIA_PACK_HALF`), the same attention path for every prompt.

Differences from CUDA that matter here:

- HIP has `pack_split_gemms() == false`, so the vendor-GEMM split and the "whole one-chunk prompt from row 0 only" rule (`pack_whole_prompts`) do not apply: later chunks of long prompts can join a pass on HIP. The hg 35B test covered "every chunk packed" (section 3).
- SRV16b's bounds (3072 rows, half the queue) are default-on since 10-29, after the hg HIP measurement. They were not in the hg numbers.

## 3. Is the packed pass bit-identical to unpacked on HIP?

What the code and the docs show (nothing run on a GPU in phase A):

- By construction, each prompt's state and outputs are its solo prefill's if every row-wise kernel gives a row the same bits whatever else is in the call: the GEMMs, router, MoE and elementwise ops run once over the concatenated rows, attention, KV append and the GatedDeltaNet conv and delta rule run once per prompt on its own state (`Model::forward_packed`). The last prompt token runs alone, as in a solo prefill.
- hg (10-25, `2026-10-25-hg-hip-serve-gap.md` section 6) measured this on HIP: `tests/packed_rows.rs` (5 passed: `matmul_nvfp4_mma`, `matmul_bf16`, `moe_topk_bf16_v2`, `moe_experts_bf16_v2`; each row has the same bits packed, 6 segments of 4167 rows, as alone, **a 15-row segment differs, 16 and up do not**, hence `PACK_MIN_ROWS = 16`), plus `fp8_f8.rs::f8_rows_do_not_depend_on_m` for the W8A8 GEMM, `batch_recurrent` (11 passed, includes packed-vs-solo) and `real_35b_packed_prefill_equals_the_local_engine` on the HIP backend (24 requests, 6 sampled, 8 and 16 slots, every chunk packed: equal to the single-sequence engine).
- SRV16 on CUDA found a real exception (cuBLAS / cuBLASLt pick tiling by row count), which is why CUDA splits those GEMMs per prompt. HIP's GEMMs are in-tree kernels (`matmul_nvfp4_mma`, `matmul_fp8_mma_f8`, bf16 mma), which `packed_rows` covers.
- Measured on current main (this branch's binary, section 5): `real_35b_packed_prefill_equals_the_local_engine` on HIP passed (24 requests, 8 slots: 7 packed passes carried 16 chunks, 16 slots: 8 passes carried 21 chunks); `packed_rows` 5 of 5 and `fp8_f8` 5 of 5 passed; and in the served A/B the greedy text of every stream (distinct ~1k-token prompts, 1 / 4 / 8 / 16 streams, 3 bursts, two visits per arm) has the same sha256 with packing on and off, with 6 / 12 / 30 packed passes in the packed arm. So on HIP the packed pass is bit-identical to unpacked for what was tested (text hash as the stand-in for ids, see `scripts/fh/ttft.py`). The 10-30 HIP kernel changes did not break it.
- tp=2 is not packed at all, so there is nothing to be identical there.

## 4. Gates run in phase A (endor, fleetq #8533, `--load cpu`, 17:09 to 17:22 EST)

| Gate | Result |
|---|---|
| `cargo test --workspace --no-fail-fast` (endor, macOS, CPU / stub GPU backends) | exit 0: 1322 passed, 0 failed, 29 ignored (the GPU-only tests; `INTARSIA_PACK_PREFILL` is explicit in the packed tests, so the default flip is not exercised by them) |
| `INTARSIA_HIP_CHECK_ONLY=1 cargo check --locked -p intarsia-backend-hip --all-targets` (CI's check-hip, real-GPU cfgs on, no hipcc) | exit 0 |
| `INTARSIA_HIP_CHECK_ONLY=1 cargo check --locked -p intarsia-cli --features hip` | exit 0 |

None of these runs a HIP kernel: the default itself (`pack_prefill_default() == true` reaching `cfg.pack_prefill` on a HIP model) and the packed pass on current main are first exercised on the GPU in phase B. Logs: `.scratch/` of the lane worktree (not committed).

## 5. Phase B: the tatooine A/B and the GPU gates

Scripts: `scripts/fh/build.sh` (rsync of the tree to tatooine, which cannot reach GitHub; incremental release build in the HIP dev image; warms the test binaries), `scripts/fh/window.sh` (the window), `ttft.py`, `table.py`. Raw: `docs/results/2026-11-05-hip-pack-default/raw/` (localeval `result.json` per visit, ttft json with every stream's text hash, server logs, gate logs, `steps.log`). `raw-w0-aborted/` is the first, aborted window (below).

Setup: the branch at `a39404a79` plus script-only changes (build `fh` = `a39404a79`; the binary has the flip), Qwen3.6-35B-A3B-NVFP4, `serve --backend hip --max-batch 16 --max-seq 8192 --mem-budget 29`, tp=1 on card 0 and `--tp 2 --devices 0,1`. Arm A = default (packed), arm B = `INTARSIA_PACK_PREFILL=0`. Visits ABAB per tp, a fresh server per visit, so two runs per cell (n=2: min-max shown, treat differences under about 2 % as noise). localeval: 1k prompt, 128 generated, 1 reps + warm-up. TTFT p50 / p95: `ttft.py`, 3 bursts per stream count, distinct deterministic ~1k-token prompts, 64 generated, pooled over both visits. Packed passes: `/metrics` delta, summed over both visits of the arm.

| tp | streams | tok/s A | tok/s B | A vs B | TTFT p50 A / B (s) | TTFT p95 A / B (s) | LE TTFT median A / B (s) | packed passes A / B | text A == B |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 163.00 (162.50-163.50, n=2) | 164.70 (164.10-165.30, n=2) | -1.0 % | 0.12 / 0.13 | 0.16 / 0.16 | 0.11 / 0.11 | 0 / 0 | yes |
| 2 | 4 | 306.75 (305.50-308.00, n=2) | 305.90 (305.40-306.40, n=2) | +0.3 % | 0.29 / 0.31 | 0.49 / 0.46 | 0.28 / 0.28 | 0 / 0 | yes |
| 2 | 8 | 397.25 (397.20-397.30, n=2) | 409.35 (408.00-410.70, n=2) | -3.0 % | 0.53 / 0.51 | 0.91 / 0.91 | 0.50 / 0.49 | 0 / 0 | yes |
| 2 | 16 | 476.05 (475.30-476.80, n=2) | 477.20 (476.10-478.30, n=2) | -0.2 % | 0.99 / 0.97 | 1.86 / 1.86 | 0.94 / 0.96 | 0 / 0 | yes |
| 1 | 1 | 134.50 (134.00-135.00, n=2) | 140.40 (139.20-141.60, n=2) | -4.2 % | 0.17 / 0.15 | 0.22 / 0.20 | 0.14 / 0.13 | 0 / 0 | yes |
| 1 | 4 | 246.10 (246.00-246.20, n=2) | 246.55 (245.70-247.40, n=2) | -0.2 % | 0.36 / 0.36 | 0.50 / 0.55 | 0.35 / 0.34 | 6 / 0 | yes |
| 1 | 8 | 333.10 (329.00-337.20, n=2) | 318.35 (316.10-320.60, n=2) | +4.6 % | 0.56 / 0.62 | 0.96 / 1.09 | 0.57 / 0.60 | 12 / 0 | yes |
| 1 | 16 | 397.50 (396.30-398.70, n=2) | 361.70 (361.40-362.00, n=2) | +9.9 % | 1.05 / 1.19 | 1.74 / 2.25 | 1.03 / 1.18 | 30 / 0 | yes |

Read:

- **tp=1, 16 streams: +9.9 % aggregate tok/s** (397.5 vs 361.7), TTFT p50 1.05 vs 1.19 s, **p95 1.74 vs 2.25 s**. 8 streams +4.6 %, TTFT p50 0.56 vs 0.62, p95 0.96 vs 1.09. 4 streams: tok/s equal (-0.2 %), TTFT p50 equal (0.36), p95 0.50 vs 0.55. The 10-25 figures were +11 % at 16 streams and +5.7 % at 4 streams with TTFT 0.37 -> 0.46 s; the 4-stream gain and TTFT cost are gone with SRV16b's bounds (half the queue, 3072 rows), and nothing waits longer than before at 4 or 8 streams.
- **tp=2: no change, as expected** (zero packed passes in both arms; differences -3.0 % to +0.3 % are run-to-run noise, the 8-stream cell has the largest spread).
- **tp=1, 1 stream: A 134.5 vs B 140.4 tok/s (-4.2 %), TTFT p50 0.17 vs 0.15 s, with zero packed passes in both arms**. A single stream packs nothing, so this is either noise (the hg doc found the 1-stream tatooine figure bimodal inside one server start, 136 vs 127, cause not isolated) or a small cost of the default that is not the packed pass. It repeated in both A visits (134.0 and 135.0) against both B visits (139.2 and 141.6), but with n=2 and tp=2 showing -1.0 % at one stream it is not established. Not chased in this window; a longer 1-stream A/B at tp=1 would settle it.
- Text identity: every stream's greedy text equal between A and B in every cell.

GPU gates on the branch binary, card 0 (same window): `real_35b_packed_prefill_equals_the_local_engine` (hip, `INTARSIA_BATCH_TEST_BACKEND=hip`) 1 passed; `packed_rows` 5 passed; `fp8_f8` 5 passed.

Jobs and radiance: fleetq #8533 (endor cpu: workspace tests, HIP cfg check), #8556 (tatooine cpu,disk: build and dry-run, radiance up), #8566 (tatooine exclusive, takes down radiance, `--mem 30`, `--expect 90`: the window). `window.sh` was dry-run first (`DRY=1`, same ssh and docker steps, `serve --help`, tests listed only), which caught a bash 3.2 empty-array error. Radiance down twice: 17:27:39 to 17:30:33 (first window, killed by the lane after the ttft rows looked right but localeval had written nowhere, because the output directory was a relative path: fixed) and 17:30:43 to 17:39:57 (answers on :8080 after 15 s, `/v1/models` listing `Qwen3.8`), 12 minutes in all; the cap was 75. Radiance's status before the first stop (`radiance-status-original-15-04.txt`): active since 15:04:15 EST. Job #8566 released after radiance answered. Radiance's own restart was otherwise untouched.

Not tested: `--max-batch 16` with prompts longer than one busy chunk in a pack (the 4k prompt case), MTP, other HIP models (27B is dense and never packs). The flip is not exercised by `cargo test --workspace` (its packed tests set the knob).
