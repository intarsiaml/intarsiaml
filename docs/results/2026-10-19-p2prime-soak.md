# P2' (`MoeBlock`) default-on gate: max-batch-1 serve soak, 2026-10-19

Branch `perf/p2prime-default` on `perf/p2prime-moeblock` (`2adb9f29`). Not pushed. HIP backend, 2 x AMD R9700 (gfx1201, host tatooine, "radiance"). Parent doc: `docs/results/2026-10-19-p2prime-moeblock.md`. Raw data: `docs/results/2026-10-19-p2prime-soak/raw/`.

## Verdict

**PASS. P2' is on by default on HIP, `INTARSIA_HIP_MOE_BLOCK=0` is the kill switch.** Default-flip commit: `0cea68fc`.

* **No hang, no stall, no restart, no error.** 13 queueing replays plus an open-loop cancel load ran on each arm for 10.5 to 10.8 minutes. The slowest request end to end took 10.1 s (P2') and 11.7 s (control) over replays and open-loop load, against a 60 s limit. 0 HTTP 429/503, 0 stream errors, `/health` 200 in every one of ~350 polls per arm, `engine_steps_total` never stuck, uptime monotonic, container exit 0 with 0 restarts, `journalctl -k` GPU filter empty.
* **Identity.** Every one of 546 replayed requests per arm (42 distinct prompts, 32 greedy and 10 seeded-sampled; 131 to 4434 prompt tokens) has the same generated token ids as its solo run on the same server (client-cancelled streams are compared on the prefix received). A second solo pass after the soak is bitwise identical to the first. P2' and control solo ids are also identical to each other for all 42 prompts.
* **Metrics.** After the load both arms drained to `requests_running = 0`, `requests_waiting = 0`, `requests_total = sum(finished)` (848 = 848 on each), `rejected = 0`, `engine_failed = 0`; 62 (P2') and 69 (control) requests finished as `cancelled`.
* **Control.** Flag off passes every criterion too, so there is no pre-existing issue to report separately.
* **Quick gates with the flip on** (window W2): KL21 35b-short and 35b-long, 8B fp32 goldens, prefix reuse and the 256 ids after the 4408-token prompt all match the parent doc's digits (section 6). Mac checks pass.

## 1. What could not be run: HIP has no batched decode

The brief asked for `serve --max-batch 8` with B = 1, 2, 4, 8 and `--spec`. HIP cannot serve either, by construction (checked first, no GPU time used):

* `crates/intarsia-models/src/load.rs` (~line 176) refuses `--max-batch > 1` on any non-Metal backend before it reads the weights: "needs the Metal backend ... use --max-batch 1".
* `Model::enable_batch` (`crates/intarsia-core/src/model/batch.rs`) accepts dense attention-only models; the 35B (GatedDeltaNet and MoE blocks) is refused even on Metal.
* `--spec` is Metal-only (`load.rs` ~line 204). HIP registers no row-exact kernels (`quantmoe.rs` comment: "HIP has no batched step").

The user approved the replacement gate: the real HIP serving path at the default `--max-batch 1`, where concurrent clients queue and the engine serves one sequence at a time. What this exercises: MoeBlock's cross-block sync counters and graph replay across thousands of back-to-back requests, request switches, client cancels mid-flight (after N ms or N streamed chunks, queued or running), state left behind by abandoned requests, and the queue. What it cannot exercise: multi-sequence rows in one step (not a HIP feature). Speculative decoding on/off was not run (Metal-only).

## 2. Load shape

Two servers at once, one per GPU, from the same binary (the pre-flip build of tree `perf/p2prime-moeblock` code, `intarsia serve --backend hip --mem-budget 29 --max-seq 8192 --profile-every 0 --no-supervisor`, 35B NVFP4, graphs on). GPU0: `INTARSIA_HIP_MOE_BLOCK=1`. GPU1: `INTARSIA_HIP_MOE_BLOCK=0` (control). Both set explicitly, so the window means the same on the pre-flip and the post-flip binary. Radiance was down for the window (job 2743).

Per arm, driven by `tools/p2prime_soak.py` (lane driver), `tools/cbidentity.py` (identity harness from the Metal CB work) and `tools/cbsoak.py` (open-loop load), all stdlib python on the host:

1. Decode-speed sentinel (3 greedy 128-token requests) to prove which kernels ran: 164.6, 164.7, 164.7 tok/s (P2') against 149.6, 151.2, 151.4 (control). The served plan (`raw/w1/*/plan.json`) has 255 kernels per decode step with `moe_block_bf16` on GPU0 and 375 with the chain on GPU1.
2. Solo reference: 42 cases (stratified sample of the 203-case set, at least 2 per family: raw and chat prompts, greedy and seeded sampling, EOS, stop strings, `min_tokens`, `ignore_eos`, 4k golden prompts, cancel cases run to completion), a flush request before each so every case prefills cold.
3. Soak, 633 s (P2') and 649 s (control): back-to-back concurrent replays of the same 42 cases, rotating B (client requests in flight) 4 / 8, streamed / not / mixed, arrival times scaled x1, x0.5, x0.25 and 0 (a burst of all cases at once), 13 replays, mixed `max_tokens` 16 to 160, plus 4 cancel cases per replay (client closes the socket after N ms or N chunks). In the background the whole time, `cbsoak` open loop at 0.3 req/s (Poisson arrivals, prompts 128 to 4096 tokens, mixed `max_tokens`, 30 % of requests cancelled mid-flight). Requests overlapped freely: up to 12 to 13 were waiting at once.
4. Drain and `/metrics` invariants, then a second solo pass, then `cbidentity compare` of every replay and of the second pass against the first.

Every request had a 60 s no-data timeout; `/health` and `/metrics` were polled every 2 s (503s, restarts via `uptime_s`, engine stall via `engine_steps_total` frozen for 30 s with work queued).

## 3. Results per B

B = 1 is the solo reference and the post-soak solo pass; B = 2 was not run (rotation was 4 / 8 / 4-with-arrival-scaling). Latency is end-to-end request time including queue wait (client side). Identity compares token ids (all 42 cases, ids present for 40 to 42 per replay; cancelled streams are compared on the prefix received).

| arm | B | replays | requests | mismatches vs solo | hangs / errors | latency p50 / p99 / max (s) |
|---|---|---|---|---|---|---|
| P2' | 1 (solo, 2 passes) | 2 | 84 | 0 (pass 2 vs pass 1) | 0 | 0.66 / 2.85 / 2.85 |
| P2' | 4 | 7 | 294 (11 cancelled) | 0 | 0 | 1.78 / 5.90 / 6.00 |
| P2' | 8 | 6 | 252 (20 cancelled) | 0 | 0 | 5.21 / 9.23 / 9.81 |
| control | 1 (solo, 2 passes) | 2 | 84 | 0 | 0 | 0.70 / 2.94 / 2.94 |
| control | 4 | 7 | 294 (16 cancelled) | 0 | 0 | 2.01 / 6.47 / 8.10 |
| control | 8 | 6 | 252 (20 cancelled) | 0 | 0 | 5.57 / 10.41 / 11.39 |

The per-replay table (13 rows per arm) is in `raw/w1/*/lane_summary.json` (`replays`) and was printed with `tools/p2prime_soak_report.py`.

Open-loop `cbsoak` load, per arm (197 requests sent, 0 errors, 0 shed, HTTP 200 on all):

| arm | ok / cancelled | TTFT p50 / p95 / p99 (ms, includes queue) | ITL p50 / p95 / p99 (ms) | ITL max (ms) |
|---|---|---|---|---|
| P2' | 166 / 31 | 1950 / 6442 / 8420 | 6.13 / 6.22 / 6.37 | 25.5 |
| control | 164 / 33 | 2973 / 7123 / 8320 | 6.67 / 6.77 / 6.91 | 27.6 |

ITL is the per-token gap on streams: P2' is 8 % lower (6.13 against 6.67 ms), the same ratio as the +9 % decode tok/s of the parent doc. The two arms ran on different GPUs at the same time, so the TTFT difference mostly reflects different cancel timing and queue luck, not the kernel.

## 4. Control against P2'

* Control has the same results on every criterion (table above, 0 mismatches, 0 hangs).
* Cross-arm: the solo pass of the P2' arm and of the control arm have identical token ids for all 42 prompts (`raw/w1/crossarm_ref.txt`; the harness only warns that the plans differ in `ops`, as they should).
* Different cancel counts between arms (11 against 16 at B = 4) are expected: a request with `cancel_after_ms` is cut at a different token because the arms decode at different speeds.

## 5. Metrics and health

| | P2' | control |
|---|---|---|
| `requests_total` / sum(finished) at idle | 848 / 848 | 848 / 848 |
| finished stop / length / cancelled | 132 / 654 / 62 | 130 / 649 / 69 |
| running / waiting at idle | 0 / 0 | 0 / 0 |
| rejected, engine_failed | 0, 0 | 0, 0 |
| max waiting seen | 12 | 13 |
| `/health` polls, non-200 | 348, 0 | 357, 0 |
| container state | running, exit 0, 0 restarts, not OOM-killed | same |
| `serve.log` lines matching error / panic / warn | 0 | 0 |
| kernel log lines matching amdgpu / gpu / reset / timeout / fault / hang | 0 | 0 |

`cached_tokens` was 0 in every replay and the compare reports 0 cases with different `cached_tokens`, so prefix reuse is no confounder in the soak (the recurrent 35B only reuses a full-sequence prefix). Prefix reuse is covered by the accuracy gate below.

## 6. Quick gates with the default flipped (W2, job 2745)

The flip build (`0cea68fc` tree, `cargo build --release -p intarsia-cli --features hip` in a CPU slot, radiance up) ran in one window, two GPUs. GPU0 with no env (default, P2' on), GPU1: ids and plans, then the kill switch (`INTARSIA_HIP_MOE_BLOCK=0`) through the same accuracy stages. Logs: `raw/w2/`.

| gate | default (P2' on) | `=0` | parent doc (P2' on) |
|---|---|---|---|
| KL21 35b-short, prefill / decode | 7.417e-3 / 1.172e-2, PASS | 7.417e-3 / 1.172e-2, PASS | 7.417e-3 / 1.172e-2 |
| KL21 35b-long (chunk 4096), prefill / decode | 4.356e-2 / 9.273e-3 | 4.356e-2 / 9.273e-3 | 4.356e-2 / 9.273e-3 |
| prefix-reuse gate | PASS | PASS | PASS |
| 8B fp32 goldens, short prefill / decode | 2.335e-4 / 3.626e-4, PASS | 2.335e-4 / 3.626e-4, PASS | same |
| 8B fp32 goldens, long prefill / decode | 3.966e-4 / 3.573e-4, PASS | 3.966e-4 / 3.573e-4, PASS | same |
| 256 greedy ids after the 4408-token prompt, default against `=0` | `IDS4K IDENTICAL` (decode 158.9 tok/s default, 144.9 with `=0`) | | |
| `plan show` 35B | 255 kernels per decode step (`moe_block_bf16` x40) | 375 (the chain) | 255 |
| `plan show` 8B | 364 | 364 | 364 |

The 35b-long prefill KL21 "FAIL" (4.356e-2 against the 3.0e-2 floor) is the pre-existing prefill floor failure documented in the parent doc (section 6): prefill chunks of 3+ tokens never take `MoeBlock`, and the same digits show with `=0`. The prefix-reuse verdict in that run is PASS.

Mac checks on the final head (`raw/mac-checks.log`): `cargo test --workspace`, `cargo check --workspace`, `cargo check --workspace --features hip`, `cargo check -p intarsia-cli --features metal`, `cargo build -p intarsia-backend-metal`, the CUDA and Metal pin tests (`moe_block_is_absent_and_the_unfused_selection_is_pinned`) and the HIP unit test `moe_block_is_default_on_and_its_shapes_are_pinned`.

## 7. Default flip

`moe_block_from(v)` returns `v != Some("0")`: unset or any other value enables `moe_block_bf16`, exactly `"0"` disables it. The shape gate (`moe_block_shape_ok`) is unchanged. Env table updated in the parent doc (section 14). CUDA and Metal register no `MoeBlock` kernel, so their plans are untouched. `INTARSIA_HIP_DISABLE=moe_block_bf16` still removes the kernel too (mechanism not tested, as before).

## 8. Downtime (radiance)

| window | job | radiance stopped | radiance answers | minutes | what |
|---|---|---|---|---|---|
| W1 | 2743 | 14:47:19 | 14:59:37 | 12.30 | soak: 35B serve max-batch 1, P2' on (GPU0) and off (GPU1) |
| W2 | 2745 | 15:05:00 | 15:08:21 | 3.35 | quick gates with the default on, `=0` control, ids, plans |

Total 15.65 min of the 45 allowed; no window over 20 min; 5.35 min of radiance serving between the two windows. The fleetq lease for W2 (job 2745) was taken at 15:01:20, so fleetq listed radiance as down for about 1.7 min before it was actually stopped (15:05:00); the 5.35 min above is physical serving time. Radiance answered `/v1/models` before each release; `docker ps` showed only `radiance` (both soak containers were removed by the window trap) before each release. CPU-only slots (radiance up): 2740 (binary check), 2744 (flip build). Nothing was run on spark1 or spark2. `raw/downtime.log` has the same rows.

## 9. Caveats

* **Not a batching test.** HIP serves one sequence at a time; concurrency is HTTP queueing. Cross-sequence batching bugs cannot show here because HIP has none.
* B = 2 was not run, and B = 1 is the solo pass. Speculative decoding on/off could not be run (Metal-only).
* The two arms ran on different GPUs at the same time. A GPU-specific problem would show on one arm only; none did.
* The soak used the pre-flip binary with the env set explicitly (`=1` and `=0`). The flip changes only the default of the same flag, and W2 checked the default and `=0` on the flip build (ids, plans, accuracy); the unset-env soak itself was not repeated.
* One box, one window of about 11 minutes per arm (about 550 replayed requests and about 850 total per arm). It is a soak, not a multi-day run; the cross-block wait mechanism's ISA-level risk (parent doc, sections 5 and 13) is the same as before. The kill switch is the mitigation.
* `dmesg` is not readable for the user on tatooine, so GPU hang and reset lines were taken from `journalctl -k --since <window start>` filtered for amdgpu, kfd, gpu, reset, timeout, fault, hang, ring (empty on both arms).
* Seeded-sampled cases are in the identity counts; all 10 per replay matched.
* The hang rule is "no data for 60 s" (client timeout) plus "total time over 60 s"; the longest request in any replay or in the open-loop load took 11.7 s, but all 8 queued requests wait behind each other, so latency under B = 8 is queue time, not stall time.
* Parent doc item "35b-long prefill FAIL" and CUDA/Metal no-op (spark1 section 9 on `2adb9f29`) are unchanged by this branch.

## 10. Files

`tools/p2prime_soak.py` (lane driver), `tools/p2prime_soak_report.py` (tables), `scripts/hip-decode/p2prime-soak.sh` (window script), `scripts/hip-decode/p2prime-moeblock.sh` (stages `ids` / `plans` / `adm` now compare default against `=0`).
Raw: `raw/w1/{gpu0-blk,gpu1-off}/` (`lane_summary.json`, `poll.jsonl`, `ref*.jsonl`, `c*.jsonl`, `compare*.txt`, `cbsoak/`, `serve.log`, `plan.json`, `metrics_*.json`, `container_state.txt`, `kernel_gpu.log`), `raw/w1/crossarm_ref.*`, `raw/w1/window.log`, `raw/w2/`, `raw/mac-checks.log`, `raw/downtime.log`.
