# fh: packed multi-prompt prefill on by default on HIP

Lane `lane-fh/hip-pack-default` (branch off origin/main `602a1fcfe`), 2026-11-05. Qwen3.6-35B-A3B NVFP4, `serve --backend hip --max-batch 16`.
**Status: phase A done (code, docs, gates on endor, measurement script). Phase B (the tatooine A/B and the GPU bit-identity gates) not run yet, see section 5.**

## 1. Decisions and approvals

- User, 2026-11-05 17:06 EST, relayed in the lane brief: "flip 2". Scope: packed multi-prompt prefill becomes the default on HIP for the hybrid MoE, as it already is on CUDA (SRV16, `docs/decisions/LOG.md`).
- User, 2026-11-05 17:08 EST, as quoted by the coordinator in a mid-task message (not seen by the lane as a user message): "radiance down OK". This covers taking radiance down on tatooine for phase B. The lane did not stop radiance on the strength of that relay alone; see section 5.

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
- Not shown by those runs: they predate the 10-30 HIP changes (hgv small-GEMV grid, hfp8 rows kernel, k4fp8, hload; none is a prefill-GEMM change as far as the log shows, but none was run against the packed pass) and the 10-27/28 prefill GEMM tile changes (`fa7e3e365` is MXFP4 / 27B). So bit-identity on current main is **expected, not re-verified**. The phase B script re-runs the three tests above on the branch binary (`GATES=1`) and compares the greedy text of every stream between arms (1 stream and 4 / 8 / 16 distinct prompts; at one stream nothing packs, so the multi-stream hashes are the ones that test packing).
- tp=2 is not packed at all, so there is nothing to be identical there.

## 4. Gates run in phase A (endor, fleetq #8533, `--load cpu`, 17:09 to 17:22 EST)

| Gate | Result |
|---|---|
| `cargo test --workspace --no-fail-fast` (endor, macOS, CPU / stub GPU backends) | exit 0: 1322 passed, 0 failed, 29 ignored (the GPU-only tests; `INTARSIA_PACK_PREFILL` is explicit in the packed tests, so the default flip is not exercised by them) |
| `INTARSIA_HIP_CHECK_ONLY=1 cargo check --locked -p intarsia-backend-hip --all-targets` (CI's check-hip, real-GPU cfgs on, no hipcc) | exit 0 |
| `INTARSIA_HIP_CHECK_ONLY=1 cargo check --locked -p intarsia-cli --features hip` | exit 0 |

None of these runs a HIP kernel: the default itself (`pack_prefill_default() == true` reaching `cfg.pack_prefill` on a HIP model) and the packed pass on current main are first exercised on the GPU in phase B. Logs: `.scratch/` of the lane worktree (not committed).

## 5. Phase B

Script (not yet run): `scripts/fh/window.sh <outdir>` under a fleetq slot that takes radiance down; `scripts/fh/build.sh <branch>` builds and warms the binary on tatooine first (CPU slot, radiance stays up); `scripts/fh/ttft.py` and `scripts/fh/table.py` are its helpers.

- Arms: A = the build's default (packed on), B = `INTARSIA_PACK_PREFILL=0`. Visits ABAB per tp, tp=2 first (`--tp 2 --devices 0,1`) and then tp=1 (card 0), a fresh `serve` per visit, `--max-batch 16 --max-seq 8192 --mem-budget 29`.
- Per visit: `localeval speed` (1k prompt, 128 generated, 1 / 4 / 8 / 16 streams: aggregate tok/s, TTFT median/min/max), then `ttft.py` (3 bursts at 1 / 4 / 8 / 16 streams, 64 generated, distinct deterministic ~1k-token prompts: TTFT p50 / p95 over all streams, sha256 of every stream's greedy text, and the `intarsia_batch_pack_passes_total` delta to prove the pass ran). localeval reports no p95 and no per-stream output, hence the helper.
- Then the 35B packed-prefill engine test and the HIP kernel tests on card 0.
- Expected GPU time: about 4 min per visit, 8 visits, plus about 10 min of gates and radiance's stop and restore: about 50 min with radiance down (the coordinator's cap is 75). Radiance is restored in an EXIT trap on every path, checked on its health endpoint, with a deadman on tatooine (starts it after 75 min if the script dies). `systemctl --user status radiance` is saved first (`radiance-status-before.txt`).
- Dry-run done here: `ttft.py` and `table.py` against a local fake SSE server (the pipeline, percentile and hash logic); `window.sh` and `build.sh` only syntax-checked, the ssh / docker steps are adapted from `scripts/35b-tp2-hip/stages-w3.sh` and `scripts/hg/win.sh` and untested.

Why phase B was held: the brief says radiance (tatooine's two R9700s, run by another session) stays up until the user approves, and the approval arrived as an agent message, not as a message from the user. A relayed quote is not consent from the user, so the lane stopped after phase A and asks for the user's own confirmation (or the permission system's) before `fleetq acquire --takes-down radiance`.
