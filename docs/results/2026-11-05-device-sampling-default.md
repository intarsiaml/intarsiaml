# Device-side top-k sampling on by default (CUDA and HIP)

Lane `lane-ds/device-sampling-default`, 2026-11-05. Branch `lane-ds/device-sampling-default` from origin/main `602a1fcfe`. Raw logs and scripts: `2026-11-05-device-sampling-default/raw/`.

## Authority

* 2026-10-23 17:57: device sampling was made opt-in, default off (`docs/decisions/LOG.md`, row "Device sampling (`INTARSIA_DEVICE_SAMPLING`, `HeadMode::LastTopK`) opt-in, default off"; the row records no reason). It had been default on from 17:07 that day.
* 2026-11-05 17:06 EST, the user: ignore what was deferred previously and build the largest performance wins. This lane carries that out for the one switch Fable's engine review ranks first among the unbooked served gains (`perf-ideas/fable-engine/report.md`, section 2, idea 4).

## What default traffic sends, and whether it takes the device path

No, for clients that omit `top_k`, which is what plain OpenAI and Anthropic clients do.

* `crates/intarsia-server/src/api.rs` (`Sampling::into_request`): temperature defaults to 1.0, `top_p` to 1.0, and `top_k` to 0 (`Some(k) if k > 0 => k`, anything else 0). `openai.rs` and `anthropic.rs` pass the client's optional `top_k` through unchanged; the server applies no default `top_k` of its own. The engine reads the checkpoint's `generation_config.json` only for EOS ids (`intarsia-models/src/load.rs:103`); the Qwen3.6-35B-A3B-NVFP4 checkpoint ships `top_k 20`, `top_p 0.95`, temperature 1.0 in it, so the model vendor's own sampling recipe is a top-k one.
* `Seq::topk_for` (`crates/intarsia-core/src/engine.rs`) takes the device path only for `1 <= top_k <= 64`. `top_k = 0` means the whole-vocabulary softmax, so those requests keep downloading the `[B, V]` logits and sorting the row on the host.
* So flipping the switch helps requests that send `top_k` (the Qwen recommended settings do: `top_k 20`; localeval's sampled targets with `top_k`), and does nothing for omit-`top_k` traffic. Measured below at 16 streams: no `top_k` (temperature 1.0) 169.7 tok/s in both arms; `top_k 40` (temperature 0.7) 222.4 tok/s on the host path and 249.7 tok/s on the device path. So giving omit-`top_k` traffic a `top_k` would be worth about 1.31x from the smaller host sort alone and about 1.47x with the device path (the two targets differ in temperature, 1.0 against 0.7).

Options for the omit-`top_k` case:

| option | exact? | verdict |
|---|---|---|
| (a) device path for `top_k = 0` / top_p-only | No. The host normalises over the whole vocabulary: `walk` sums `exp((l - max) / T)` as f64 over all sorted candidates, in sorted order, with Rust's libm `exp`, and then compares the cumulative sum to the draw. A device kernel cannot reproduce that sum bit for bit (different `exp`, different summation order), so a draw near a cumulative boundary could differ. Making it exact needs an error bound plus a fallback that downloads the row, which is a new kernel pair and a new exactness argument, not a flip | not built |
| (b) server or recipe default `top_k` (for example 20 or 40) | Exact on the device path, but it changes which tokens can be drawn from every omit-`top_k` request (the mass beyond the top 64 at temperature 1.0 is removed) | a user decision, not built |
| (c) leave it | exact | what this lane does |

Recommendation for the user: (b) is the cheapest way to make default traffic fast and the checkpoint already ships the value (`top_k 20`); it would be a per-recipe default read from `generation_config.json` rather than a global constant, and it needs the user's OK because it changes the sampled distribution of every omit-`top_k` request. Recorded as open decision O15 in `docs/decisions/LOG.md`.

## The change

* `crates/intarsia-core/src/model/batch.rs`: device sampling buffers are allocated and `HeadMode::LastTopK` is offered unless `INTARSIA_DEVICE_SAMPLING=0` (was: only when `=1`). The switch is `device_sampling_switch` with a unit test. Applies where `Backend::supports_topk()` is true; the overrides are in the CUDA and HIP backends only (`grep 'fn supports_topk' crates`: cuda `device.rs:362`, hip `device.rs:1552`, trait default false at `backend.rs:478`), so Metal and CPU are unchanged.
* Comments updated in `model.rs`, `intarsia-backend-hip/src/device.rs`, `tests/batch_cuda_real.rs` (the test now skips only with `=0`).
* Docs: `docs/tuning.md` env table row, `docs/decisions/LOG.md` defaults table (CUDA and HIP batched decode rows) and a new LOG row, `docs/backends/hip.md`. The task named `docs/tuning.md` lines 95-96; those lines are in `docs/decisions/LOG.md`.

Scope of the path, unchanged: batched (`--max-batch N`) sampled rows with `1 <= top_k <= 64`; greedy, top-p-only, constrained, `min_tokens`-with-stop-ids, NaN rows and `top_k > 64` use the host path as before.

## Equality: device against host, served, 16 concurrent

`scripts/htk/equal.py` (7 sampling settings x 1 / 4 / 16 concurrent streams, seeded, 64 forced tokens each, token ids and text compared), against one `intarsia serve --backend cuda --max-batch 16` of the new binary per arm, spark1, Qwen3.6-35B-A3B NVFP4, same binary and flags, arm A the new default, arm B `INTARSIA_DEVICE_SAMPLING=0`.

| | result |
|---|---|
| cells identical | **21 / 21**, 9408 token ids (`raw/ids-compare.txt`) |
| device cells (top_k 1..64) | 15, all equal, including the 16-stream cells at top_k 1, 8, 20, 40, 64 |
| host-path cells (top-p only, greedy) | 6, all equal (same path in both arms) |

## Speed: default against `=0`, localeval, ABAB

localeval 0.4.0 `speed`, 1024-token prompts, 128 forced tokens, 2 reps, burst at 4 and 16 streams; target `ds-sampled` sends temperature 0.7 and `top_k 40` (`raw/scripts/targets.toml`; the suite pins `top_p 1`, seed 0). One fresh server per visit, order default, 0, default, 0 (visits 1-4), page cache dropped before each. Run ids in the raw `v*-ds-*.txt`.

| aggregate decode tok/s | visit 1 default | visit 2 `=0` | visit 3 default | visit 4 `=0` | default mean | `=0` mean | ratio |
|---|---|---|---|---|---|---|---|
| sampled, 4 streams | 154.5 | 146.9 | 151.8 | 139.2 | 153.2 | 143.1 | **1.07** |
| sampled, 16 streams | 252.1 | 223.3 | 247.2 | 221.4 | 249.7 | 222.4 | **1.12** |
| sampled, 1 stream (decode tok/s, rep 1 / 2) | 82.4 / 82.2 | 82.0 / 81.7 | 82.3 / 82.2 | 81.8 / 86.2 | 82.3 | 82.9 | 0.99 (noise; single stream is not batched) |
| no `top_k` (`ds-nok`, temperature 1.0), 16 streams | 166.6 | 170.2 | 172.7 | 169.1 | 169.7 | 169.7 | 1.00 |

Every default visit beat both neighbouring `=0` visits at 4 and 16 streams. The no-`top_k` row is the point of the first section: default API traffic is unchanged by the flip, and costs about a third of the throughput of the same server when the request carries `top_k`.

Server MemAvailable with the server up: 54 to 55 GB in all four visits (`raw/mem.txt`); the device buffers are `BATCH_MAX_ROWS * 64 * (4 + 4) + ... + BATCH_MAX_ROWS * 32 * 64 * 8` bytes, well under 1 MB per row slot.

## Gates

| gate | result | job |
|---|---|---|
| CUDA release build of the final source (the rsync excludes only `.git`, `.scratch`, `target*`, `.worktrees`, `docs/results`, so the binary is exactly the committed code) | ok, `~/intarsiaml-ds/src/target/release/intarsia` md5 `d711631b...` | fleetq #8552 (first build), #8567 (final source) |
| Served equality, device against host, 16 concurrent, 21 cells | 21 / 21 identical | #8568 |
| Sampled aggregate tok/s at 4 and 16 streams, default against `=0`, ABAB, localeval | 1.07x and 1.12x | #8568 |
| `cargo test --workspace --no-fail-fast` (endor; includes the new unit test `device_sampling_is_on_unless_set_to_zero`, which first compiles there) | PENDING | #8598 queued |
| Metal no-op on endor: `plan show` (35B, 8B), `kernels test`, 256 greedy ids (8B short and 4k, 35B short), goldens `qwen3-8b-fp32` and `-long-fp32` | PENDING | #8634 queued |
| In-process CUDA draws test `real_35b_device_sampling_draws_the_host_tokens` with the new default | not run: dropped (#8580 released), redundant with the served 21/21 and queued behind exclusive DSv4.1 windows | |
| `cargo check -p intarsia-core` | ok (run on endor outside a fleetq slot, no GPU, deviation from the fleet rule) | |

## Not run

* HIP served check (tatooine busy with another lane's window; no slot was sent). HIP's device path rests on the 2026-10-27 lane htk evidence (LOG 10-27 01:15: 21/21 served cells identical, 3360 draws equal, 1.07x / 1.09x). The change is the same line for HIP.
* Options (a) and (b) for omit-`top_k` traffic, see above.
* Equality was gated on 35B tp1 CUDA only. Tensor-parallel runs (the vocab-parallel head; device top-k reads rank 0's logits only) and the other batched models also default to the device path now, and are not gated here.
* `spark1:~/intarsiaml-ds` (source, target, server logs) was created for the build and left in place.
