# Metal phase A exit gate (2026-10-14)

**Outcome: `intarsia qualify --backend metal` on Qwen3-8B BF16 passes: 10 stages PASS, 2 SKIP, 0 FAIL, 0 ERROR, exit code 0.** Accuracy against the fp32 goldens is in the same range as HIP and CUDA. Decode is 33.3 tok/s at 128 tokens, 96-97 percent of mlx-lm (34.4) and llama.cpp (34.8). That comparison is across harnesses (qualify's `bench speed` vs localeval through a server), so it is indicative only. Phase B's exit is not met: it needs at least 85 percent of the byte floor (82 here) and decode not below llama.cpp, measured the same way as the baselines. Prefill is the gap: 311-326 tok/s warm up to 1k tokens and 95 tok/s at 16k, against 2000-3500 on HIP and 3189 (mlx-lm, 4k marginal). The 7.6 tok/s prefill seen on a 16-token prompt is not the kernels: it is disk page-in of the 16.4 GB of mmap'd weights on a cold page cache (section 3).

Host: endor (Apple M5 Max, 40-core GPU, 128 GB, macOS 27.0). Branch `spike/metal` at `708df23` plus the two harness changes in section 1. Model `/Users/josef/models/Qwen3-8B` (BF16, 16.38 GB), goldens `golden/qwen3-8b-fp32.safetensors` and `golden/qwen3-8b-long-fp32.safetensors` (found by model name). Build was unstamped (`INTARSIA_BUILD_STAMP` unset), so the baselines written under `baselines/` are marked unstamped.

Command (fleetq #1795, `--exclusive --mem 24`, `INTARSIA_CPU_FREE_MEM_GB=60`; qualify's cpu use is only the oracle inside kernel admission):

```sh
intarsia qualify --model /Users/josef/models/Qwen3-8B --backend metal --bandwidth-gbs 614 \
    --out docs/results/raw/2026-10-14-metal-phase-a/qualify
```

Raw logs: `docs/results/raw/2026-10-14-metal-phase-a/` (`qualify/` is the full report directory with `full-report.md`, `state.json`, `logs/`, `bench/`; `qualify-console.log` is the run's stderr).

fleetq: #1795 (qualify, 16m24s) and #1796 (prefill probes and profile), both exclusive on endor. `fleetq status` before the runs showed `endor-kev` (18 GB) and `endor-tensorfold` (32 GB) resident, no jobs. Neither has a health check, so idle background load cannot be ruled out. Thermal state was not measured.

## 1. Two qualify changes the run needed

Both are in `crates/intarsia-cli/src/cmd/qualify.rs`. No gate or tolerance was touched.

1. `qualify` rejected `--backend metal` ("unknown backend ... (cpu, cuda, hip)", exit 2). The whitelist now includes `metal`. The `--backend` help text and `docs/adding-a-model.md` say so too.
2. The `envelope` stage ran the `attn-simt` perturbation on every backend except cpu. That perturbation switches off the flash / split-KV attention kernels, which exist on cuda and hip only. `declare()` in `accuracy/envelope.rs` has `bail!("--noise-perturb attn-simt: {b} backend has no flash / SIMT attention (cuda, hip); use chunk or external:TEXT")` for any other backend, so on metal the stage would have ended in ERROR (not FAIL) and the full report would have been INCOMPLETE. The stage now runs `attn-simt` on cuda and hip, and the chunk perturbation alone on cpu and metal (what cpu already did). I did not run the unpatched stage; the claim comes from reading the code. This is a change to what the harness measures, not to a tolerance: the Metal envelope has less coverage than the HIP one (one perturbation instead of two), and a narrower envelope makes later baseline checks at least as strict, not looser. It is a harness decision the reviewer can veto. Operator decision, 2026-10-14: "keep it" (chunk-only envelope on metal stays).

## 2. Qualify stages

| # | Stage | Verdict | Wall | Result |
|---|---|---|---:|---|
| 1 | `doctor` | PASS | 0.5 s | Apple M5 Max (applegpu_g17s), 16 kernels registered |
| 2 | `recipe` | PASS | 0 s | `builtin:qwen3` covers the model (25/25 templates, bf16) |
| 3 | `plan` | PASS | 34 s | 14 ops resolved, exactness deterministic, all admitted. The 34 s load is the kernel admission run |
| 4 | `kernels` | PASS | 22 s | 16 kernels, 181 cases, 0 failures |
| 5 | `gate` | PASS | 16 s | perplexity 1.85, 5/5 sanity prompts, no non-finite rows |
| 6 | `acc-short` | PASS | 11 s | prefill top-1 100.00 percent, KL 2.61e-4; decode top-1 100.00 percent, KL 3.64e-4 |
| 7 | `bench` | PASS | 2m48s | decode 33.3 tok/s at 128 tokens (82 percent of the 614 GB/s byte floor) |
| 8 | `acc-long` | PASS | 54 s | prefill 99.86 percent, KL 3.74e-4; decode 99.60 percent, KL 3.05e-4; prefix reuse PASS |
| 9 | `envelope` | PASS | 22 s | chunk envelope written, baseline self-check PASS |
| 10 | `bench-long` | PASS | 10m56s | decode 32.8 tok/s at 1k; 16k prefill alone took 177 s per run |
| 11 | `ab` | SKIP | 0 s | no `--reference` server (none was attempted) |
| 12 | `tp` | SKIP | 0 s | `--tp` not given; Metal has one GPU and `common.rs` rejects `--tp` above 1 |

Total 16m24s (quick 4m11s, in-depth 12m13s).

**Graphs.** `supports_graphs` is false (ADR-0009 decision 3). qualify does not pass `--require-graphs`, so the bench stages run and report "N graph replays ... (decode ran eagerly)" as notes, not failures. Every row shows `graph.replays 0, eager_steps > 0`. The HIP report carries the same notes.

**Byte floor.** `--bandwidth-gbs 614` was needed. qualify's floor table has entries for gfx1201 and sm_121 only and otherwise falls back to doctor's `device_copy_gbs`, which is 139.5 GB/s on this host and would put every row above 100 percent of the floor. An `applegpu_g17s` entry in `floor_inputs` (or a real copy-kernel figure in doctor) is a follow-up; I did not add it.

### Kernel table (`plan.json`, `logs/kernels.out`)

All 16 registered kernels are ADMITTED against the cpu oracle. 14 ops are resolved for the model; calls are per decode step.

| Op | Kernel | Cases | max_abs | rel_l2 | Calls per decode step |
|---|---|---:|---:|---:|---:|
| embedding | `embedding_bf16` | 6 | 0 | 0 | 1 |
| rmsnorm | `rmsnorm_bf16` | 12 | 7.0e-4 | 2.7e-5 | 1 |
| rmsnorm_add | `rmsnorm_add_bf16` | 12 | 6.4e-4 | 1.4e-5 | 72 |
| qk_norm | `qk_norm_bf16` | 18 | 1.4e-3 | 2.2e-5 | 72 |
| rope | `rope_bf16` | 36 | 1.7e-3 | 2.2e-5 | 72 |
| matmul | `matmul_bf16` (tiled GEMM, prefill) | 21 | 3.2e-3 | 1.2e-4 | 0 |
| matmul | `matmul_bf16_gemv` (decode) | 7 | 2.9e-4 | 1.6e-5 | 252 |
| logits | `logits_bf16` / `logits_bf16_gemv` | 4 / 2 | 1.4e-6 / 1.3e-7 | 6.1e-7 / 8.9e-8 | 0 / 1 |
| kv_append | `kv_append_bf16` | 4 | 0 | 0 | 72 |
| silu_glu | `silu_glu_bf16` | 9 | 7.4e-7 | 2.0e-8 | 36 |
| argmax | `argmax_f32` / `argmax_bf16` | 4 / 4 | 0 | 0 | 1 / 0 |
| attn_prefill | `attn_prefill_bf16` | 21 | 2.0e-3 | 1.1e-4 | 0 |
| attn_decode | `attn_decode_bf16` | 15 | 3.4e-3 | 2.4e-4 | 36 |
| add | `add_bf16` | 6 | 0 | 0 | not used by this model |

### Accuracy against the fp32 goldens

Same goldens and gate class (bf16: top-1 at least 99.5 percent, mean KL at most 1e-3, no non-finite rows) for all three backends. HIP is `docs/results/2026-10-13-qualify/qwen3-8b-hip/` (R9700, gfx1201); CUDA is the 8B line of `docs/results/2026-10-14-cuda-decode-fusion.md` (GB10).

| Golden, mode | Metal (M5 Max) | HIP (R9700) | CUDA (GB10) |
|---|---|---|---|
| short, prefill (434 pos.) | 100.00 percent, KL 2.61e-4 | 100.00 percent, KL 2.34e-4 | 100.00 percent, KL 2.90e-4 |
| short, decode (128 pos., teacher-forced) | 100.00 percent, KL 3.64e-4 | 100.00 percent, KL 3.63e-4 | 100.00 percent, KL 3.24e-4 |
| long, prefill (1434 pos.) | 99.86 percent, KL 3.74e-4 | 99.58 percent, KL 3.97e-4 | 99.65 percent, KL 4.00e-4 |
| long, decode (500 pos.) | 99.60 percent, KL 3.05e-4 | 99.80 percent, KL 3.05e-4 | 100.00 percent, KL 3.33e-4 |

No non-finite rows. Full-vocab KL is within about 15 percent of HIP and CUDA in every row, and the Metal long-prefill top-1 is the highest of the three. The HIP phase 2a document also lists numbers against HF bf16 goldens (KL 5.6e-4 / 9.6e-4); those use a different golden and are not comparable. Prefix reuse on `long_prose_700` passed in all three cases (continue, restore, aligned; 32/32 greedy agreement). The 300-token and shorter prompts are skipped by the harness because the split is below one prefill chunk (256), as on the other backends.

## 3. Prefill anomaly

**Cause: disk page-in of the mmap'd weights on a cold page cache. Not the kernels.**

16-token prompt (`The capital of France is ...`), `intarsia generate`, same binary, `--max-tokens 8`. The cold-cache run used an APFS clone (`cp -c`) of the model directory, which has its own page-cache pages and was deleted afterwards; the pages of the original were already resident. This is the only way I found to get a cold cache without `sudo purge`.

| Run | Page cache | Prefill | Note |
|---|---|---:|---|
| clone, first run | cold | **2154 ms** (7.4 tok/s) | reproduces the reported 7.6 tok/s |
| clone, second run (new process) | warm | 497 ms | |
| original (new process) | warm | 536 ms | |
| original, after kernel admission (new process) | warm | 471 ms | pipelines already built; still ~470 ms |
| server, first request after start (53 tokens) | warm | 429 ms | |
| server, later requests (53-58 tokens) | warm | 251-260 ms | steady state |

So, for a 16-token prompt:

- About 1.65 s of the 2.15 s is reading 16.4 GB from disk (about 10 GB/s, NVMe sequential). It is paid once per cold file and is not Metal-specific; the first token after a reboot on any mmap-based engine pays it.
- About 0.18 s more is, by inference from timing, the first GPU touch of the weights with a warm page cache. The clean evidence is the server's first request against its later ones at the same length: 429 ms vs 251-260 ms for 53-58 tokens, same process, same path. It is not measured directly (no GPU page-fault counter was read), and the first request also includes any other one-time cost in the engine. The fresh-process `generate` runs (470-540 ms) are a different path and length and only agree roughly.
- The remaining ~250 ms is the real cost of a 53-token prefill (see below).

**Warm prefill throughput** (server, `/v1/completions`, unique nonce per prompt so the prefix cache cannot hit, wall time for `max_tokens=1`; log in `prefill-probe-server.log`, probe in `probe_ttft.py`, a diagnostic for page-in vs kernel time only, not a speed figure to compare engines (cross-engine numbers come from localeval); the nonce adds about 35 tokens to each prompt):

| Prompt tokens | Wall | tok/s |
|---:|---:|---:|
| 53-58 | 251-260 ms | 219-223 |
| 192 | 580 ms | 331 |
| 1352-1355 | 4.23 s | 320 |
| 3773-3774 | 16.0-16.2 s | 233-236 |

Qualify's `bench` (warmup discarded) agrees: 311 tok/s at 125 tokens, 326 at 1042, 213 at 4311, and in `bench-long` 344 at 988, 228 at 4072, **95 at 16717 (TTFT 177 s)**. Marginal cost is 2.4-3.1 ms per token to 1k tokens (about 320-410 tok/s), rising to 4.9 ms per token between 1.3k and 3.8k tokens as attention grows.

**Per-op prefill profile** (`generate --profile-every 1 --op-table --max-tokens 1`, 1320 tokens, chunk 256, 4.87 s; `prefill-op-table-1k.err`):

| Op | ms | Share |
|---|---:|---:|
| matmul (`matmul_bf16`, 1512 calls) | 3647 | 74.9 percent |
| attn_prefill (216 calls) | 722 | 14.8 percent |
| rmsnorm_add | 180 | 3.7 percent |
| qk_norm, rope, kv_append | 73 / 73 / 71 | 1.5 percent each |
| embedding, silu_glu, logits, rest | 42 / 40 / 18 / 2 | under 1 percent each |

The prefill GEMM runs at about 5 TFLOP/s (6.95 B non-embedding parameters times 2 times 1320 tokens in 3.65 s) and prefill attention at about 1.4 TFLOP/s at 1k context. Prefill attention is quadratic, which is why 16k is 95 tok/s. These are the kernels as written in A1/A2 (scalar-staged 32x32x32 simdgroup GEMM, flash-style attention with a K tile in threadgroup memory), correct and admitted but not tuned. That is the phase B work, not a bug.

**Would a warm-up or `MTLResidencySet` help?** Only the 0.18 s first-touch part, and I did not implement or test either.

- A residency set (`MTLResidencySet`, `requestResidency()` on a set holding the wrapped weight buffers, attached to the queue) would move the GPU first-touch of the weights from the first request to load time. It does nothing for the disk read.
- The disk part is only avoided by touching the file before serving: `madvise(MADV_WILLNEED)` or a sequential read at load, which costs the same 1.6 s but at startup instead of in the first request's TTFT. A one-time warm-up prefill after load would hide both. For `serve`, that is a startup-latency versus first-TTFT trade-off worth deciding in phase B.
- Not measured: whether a residency set also lowers steady-state GEMV time (it removes per-command-buffer residency tracking for many buffers; the weights are a handful of large no-copy buffers, so I would expect little).

## 4. Speed snapshot

qualify's `bench` stage is `intarsia bench speed` against `intarsia serve` (sizes 128/1024/4096, gen 128, 3 reps, 1 warmup, thinking off, `--gb-per-token 15.1368 --bandwidth-gbs 614`, default prefill chunk 256), so no separate speed run was made. The numbers are from fleetq #1795, exclusive. The baselines are the localeval runs of `docs/results/2026-10-14-metal-spike.md` (llama.cpp 0.5.0 b11146 with `-fa on`, mlx-lm 0.31.3); the harness differs from `bench speed`, and prompt sizes match only approximately.

| Prompt tokens | Decode tok/s (stdev) | % of byte floor (40.6 tok/s) | mlx-lm | llama.cpp | Prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 125 | **33.31** (0.10) | 82 | 34.4 (97 percent) | 34.8 (96 percent) | 311 |
| 1042 | **32.06** (0.42) | 79 | 33.8 (95 percent) | 33.4 (96 percent) | 326 |
| 4311 | **30.34** (0.22) | 75 | 31.0 (98 percent) | 31.7 (96 percent) | 213 vs mlx-lm 3189 / llama.cpp 1585 (marginal at 4k) |
| 16717 | 23.27 (0.02) | 57 | n/a | n/a | 95 |

Decode is 30.0 ms per token at short context. The measured GEMV-only ceiling was 26.07 ms, so about 4 ms per token goes to the other ops, launch and eager command-buffer encoding. Against the spike doc's phase B decode target as 90 percent of mlx-lm at each size (31.0, 30.4, 27.9 tok/s at 128, 1k, 4k), 128, 1k and 4k already pass; read as the absolute 31 tok/s it is met at 128 and 1k and missed at 4k (30.3). 16k cannot be judged because there is no mlx-lm number for it. The prefill gap is 15x against mlx-lm at 4k.

## 5. Phase B list

From the A2 notes and this run, in the order the numbers argue for:

1. **Prefill GEMM**: simdgroup-matrix kernel with bf16 operands loaded directly (no scalar bf16-to-f32 staging), larger tiles, M-aware tile choice. Matmul is 75 percent of prefill at 1k and runs at about 5 TFLOP/s. Decide here whether `mpp::tensor_ops` pays on M5 (open item 4 of the spike doc).
2. **Prefill attention**: simdgroup_matrix QK and PV. 14.8 percent of prefill at 1k, but quadratic: 16k takes 177 s.
3. **Split-KV decode attention** for 16k and longer. Decode falls from 33.3 to 23.3 tok/s between 125 and 16.7k tokens (30 ms to 43 ms per token).
4. **GQA-grouped decode**: Qwen3-8B has 32 query heads and 8 KV heads, so each KV head is read four times today. At 16.7k tokens the cache is about 2.5 GB, so that is about 10 GB per token instead of 2.5 GB.
5. **mmap-backed GEMV bandwidth**: the spike measured GEMV on heap buffers only. The 1k decode at 79 percent of floor against 94.6 percent GEMV-only suggests checking the mmap'd path, the 8-byte-aligned shard (`model-00001`, header length 8 mod 16), and eager encoding as separate costs.
6. **Weight warm-up at load** (section 3): residency set and a sequential pre-read, to move the 0.18 s first touch and the cold-cache read out of the first TTFT.
7. **Fingerprint coverage**: `Backend::fingerprint` hashes the embedded metallib (shader bytes) only. Changes to the Rust launchers (grid sizes, variant selection, argument packing) do not change it, so a baseline can survive a launcher change that alters numerics. Either hash the launcher source or add an explicit launcher version constant.
8. **qualify on Apple silicon**: an `applegpu_g17s` entry in `floor_inputs`, and the measured copy bandwidth (550 GB/s, 89 percent of 614) improved so `bandwidth_gbs` stops under-reporting.
9. **A/B stage and the second envelope perturbation**: `ab` needs a `--reference` server (none exists on this Mac for the 8B); the envelope has only the chunk perturbation until Metal has a second attention path to switch (the split-KV kernel in item 3 would provide one).

## Not run or not established

- `ab` (needs a reference server) and `tp` (one GPU) were skipped by design.
- A truly cold page cache was approximated with an APFS clone, not `purge`.
- The residency-set and warm-up ideas were not tried.
- Thermal behaviour and background load from `endor-kev` / `endor-tensorfold` were not measured.
- Speed numbers are one qualify run (3 reps per size). The stdev across reps is small, but there is no run-to-run repeat.
