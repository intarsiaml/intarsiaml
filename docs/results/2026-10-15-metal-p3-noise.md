# Metal 4k serve-path prefill noise at chunk 2048: a short tail chunk on the slow GEMM

Date: 2026-10-15 (07:00-08:25 EDT), endor (Apple M5 Max), Qwen3-8B BF16, branch `metal-p3-noise` off `metal-candidate2`
(d93127b = spike/metal a2102bf + P3, Metal prefill chunk 2048). fleetq job **#1975** (exclusive, released 08:16).
Raw data, scripts and logs: `docs/results/raw/2026-10-15-metal-p3-noise/`.

## Outcome

**Cause.** The spread (4k TTFT 1.25 / 1.08 / 1.05 s in the clock-logged slot) is not the GPU clock and not chunk 2048
as such. It is the length of the **last prefill chunk**. localeval calibrates its 4096-token prompt per pass, so the
prompt lands anywhere in 4083-4115 tokens. The engine cuts `n = prompt - 3` rows (3 chat-template tokens are always
cached) into chunks of 2048, so:

* n <= 4096 (prompt <= 4099): two forwards, 2048 + about 2040 rows. This is the fast case (1.04-1.09 s).
* n = 4097-4111 (prompt 4100-4114): 2048 + 2048 + a **tail of 1-15 rows**. A tail below `NAX_MIN_M` = 16 did not use the
  NAX GEMM but the tiled `matmul_bf16`, which streams the 16 GB of weights at a tenth of the bandwidth: **about 200 ms
  for an 11-row tail** (181.9 ms matmul + 14.4 ms attention against about 58 ms when the tail runs on NAX). That is 15-20 %
  of a 4k prefill, from 11 rows.

The same holds at chunk 256 (SAFE): 4096 is 16 x 256 as well, so SAFE has the same cliff. In the final-clock slot SAFE's
prompts never drew into the window (4083-4091), CAND's pass r2 did (4109-4112). Plain `generate` and the 4062-token
serve-prefill prompt (n = 4059) are in the fast case at both chunk sizes, which is why they look tight.

**Fix (backend-local, 3 lines of logic).** `NAX_MIN_M` 16 -> 2 in `crates/intarsia-backend-metal/src/gemm_nax.rs`, so
`matmul_bf16_nax` / `matmul_pair_bf16_nax` also take the 2..15-row tail chunk (M = 1 stays the decode GEMV). No change to
`intarsia-core`, CUDA or HIP. The NAX kernels' rows are bitwise independent of the call's M and tile
(`tests/gemm_nax.rs::nax_rows_do_not_depend_on_the_call`, now with 2/11/15-row calls), so a row gets the same bits whatever
chunk it lands in; the output is bitwise unchanged for every prompt whose chunking has no tail of 2..15 rows, and a tail of
2..15 now matches the tail-free chunking instead of differing from it (see "Output").

| tail 2..15 rows | before | after |
|---|---|---|
| tail forward (n = 4107, kernel profile) | 197 ms | 58 ms |
| engine prefill, chunk 2048, 12 clock-filtered requests each | 1274 ms | 1128 ms (-146 ms, -11.5 %) |
| localeval TTFT, prompt 4107-4115, 12 / 10 reps | 1.28 s (1.26-1.31) | 1.13 s (1.12-1.15) |
| localeval marginal (4108 row vs 1k), 4 passes each, median | 3049 tok/s | 3549 tok/s (+16 %) |
| prompts without such a tail | unchanged (engine 1074 vs 1076 ms, TTFT 1.08 vs 1.08 s) | |

What remains: a tail of 2..63 rows is still one extra weight-streaming forward, about **+55 ms (5 %)** over a tail-free
prompt (1128 vs 1074 ms). So the localeval marginal figure still depends on the prompt length it draws (about 3550 with a
tail, about 3720-3750 without, clock permitting). Removing it would need balanced chunking in `generate_one`
(`crates/intarsia-core/src/engine.rs`, shared by every backend, interacts with state snapshots and prefix reuse), which I did
not do; see "Follow-ups".

## 1. The discriminating variable: last-chunk size

Every 4096-row rep of the clock-logged slot (`docs/results/raw/2026-10-15-metal-final-clock`, spike/metal 3c92dda), with its
split. GPU MHz is the mean over the request's window (macmon, 100 ms). Table: `raw/.../final-clock-4k-reps.md`.

| pass | prompt tokens (3 reps) | forwards | last chunk (rows) | TTFT s | GPU MHz mean |
|---|---|---|---|---|---|
| CAND r1 | 4091 / 4089 / 4087 | 2 | 2040 / 2038 / 2036 | 1.07 / 1.09 / 1.09 | 1545 / 1530 / 1517 |
| **CAND r2** | **4112 / 4110 / 4109** | **3** | **13 / 11 / 10** | **1.24 / 1.25 / 1.25** | 1593 / 1578 / 1568 |
| CAND r3 | 4085 / 4086 / 4086 | 2 | 2034 / 2035 / 2035 | 1.04 / 1.05 / 1.06 | 1579 / 1589 / 1551 |
| SAFE r2 | 4087 / 4085 / 4088 | 16 | 244 / 242 / 245 | 1.15 x3 | 1605 / 1597 / 1608 |
| SAFE r3 | 4083 / 4085 / 4089 | 16 | 240 / 242 / 246 | 1.15 x3 | 1608 / 1605 / 1592 |
| SAFE r1 | 4089 / 4091 / 4091 | 16 | 246 / 248 / 248 | 1.24 / 1.68 / 1.86 | 1454 / 1081 / 1008 (slow clock, the known effect) |

CAND r2 ran at the highest clock of the three CAND passes and was the slowest, so the clock cannot explain it; r1 and r3
differ by 0.03 s at 1517-1589 MHz (r1 sits 30-70 MHz lower, a plausible but unproven reason). The spread in
marginal tok/s (3143 / 3719 / 3869) follows from 3072 / (TTFT_4k - TTFT_1k): 0.2 s on a 1 s denominator is 20 %.

## 2. Mechanism (kernel profile)

`INTARSIA_METAL_PROFILE=kernel:` on plain `generate --prefill-chunk 2048 --max-tokens 1` with exact-length prompts
(`identity-logits/w4086.txt`, `w4107.txt`: N single-token words, so tokens = N), per-forward GPU time by leading dim `t`
(`raw/.../kernel-profile/`):

| prompt | forward (rows) | GPU ms | of which |
|---|---|---|---|
| 4086 | 2048 / 2038 | 501 / 540 | NAX matmuls 456 / 449, attention 25 / 73 |
| 4107, before | 2048 / 2048 / **11** | 1042 / (same) / **197** | **`matmul_bf16` 181.9 (252 calls)**, attention 14.4 |
| 4107, after | 2048 / 2048 / 11 | same / same / **57.6** | `matmul_pair_bf16_nax` 22.3, `matmul_bf16_nax` 20.2, attention 14.4 |

The default tile pick is also the best for the tail (32x32w1 56.9, 64x64w2 63.2, 32x128w2 61.4 ms), so no new tile rule.
The 0 ms "first chunk" in the per-chunk trace is not GPU time: chunks are submitted asynchronously and the next chunk's
token-id upload (`Backend::upload` synchronizes) is what waits for the previous one.

## 3. Ruled out or not the cause

* GPU clock: CAND r2 had the highest clock; within one clock state the tail classes separate cleanly (below). The
  separate slow-clock effect (docs/results/2026-10-15-metal-serve-prefill.md) is still real: SAFE r1 above, and the first
  localeval pass of my own run1 sat at 1120 MHz / 36 W for all three 4k reps (1.44-1.51 s).
* KV allocation / growth, heap churn, attention tile pick: the workspace and KV cache are allocated at load; the
  per-forward kernel profile above shows the 2048-row forwards identical (501 vs 1042/2 ms) whether or not a tail follows.
* Splitting 2048 + 2048 vs remainder: this is the cause, through the tail size, not through the split itself
  (2048 + 2038 is as fast as the 256-chunk path: 1.04-1.09 s vs 1.15 s).

Sweep of the engine-side prefill time by last-chunk size (chunk 2048, run2, 16 prompt lengths, GPU clock scaled to
1600 MHz because the machine sat at 1350-1450 MHz that hour; the 1450 MHz filter alone leaves too few samples):

| last chunk | requests | median ms scaled to 1600 MHz |
|---|---|---|
| >= 2000 rows (two forwards) | 72 | 1042 |
| 64-1999 | 32 | 1048 |
| 16-63 | 50 | 1102 |
| **2-15** | **70** | **1241** |

(chunk 256 shows the same: tail >= 200 rows 1130, 64-199 1128, 16-63 1187, 2-15 1321.) Raw: `raw/.../run2/`, `an3.py`.

## 4. A/B

Same machine, fresh `intarsia serve` per entry, ABBA, `INTARSIA_PREFILL_TRACE` engine-side timings (the experiment-only patch
`raw/.../prefill-trace-plen.patch`, not committed), macmon at 100 ms, GPU clock per request, requests with a mean clock
below 1450 MHz dropped for the "fast" columns. CAND = the d93127b build (`final/intarsia-cand`), FIX = this commit.

**localeval (`speed --sizes 128,1024,4096,4108 --gen 256 --reps 3`), 8 passes ABBA ABBA, chunk 2048** (`raw/.../le1/`,
`localeval-ab.md`, `localeval-marginal.md`; mean GPU clock 1470-1590 MHz, only one rep below 1450, p5 rep 3 at 1195 MHz):

| row | last chunk | build | reps | TTFT median s (range) | engine prefill median ms, clock-filtered |
|---|---|---|---|---|---|
| 4096 | 2032-2039 (no tail) | CAND | 12 | 1.08 (1.04-1.38) | 1074 (n=11) |
| 4096 | 2032-2039 | FIX | 12 | 1.08 (1.05-1.11) | 1076 (n=12) |
| 4108 | 8-14 (tail < 16) | CAND | 12 | 1.28 (1.26-1.31) | 1274 (n=12) |
| 4108 | 11-15 | FIX | 10 | 1.13 (1.12-1.15) | 1128 (n=12 with the two 16-row reps) |

Marginal prefill vs the 1k row (localeval's own number for the 4096 row): CAND 3795 / 3713 / 3655 / 3804 (median 3754),
FIX 3690 / 3625 / 3753 / 3828 (median 3721), i.e. equal within the run-to-run range, as expected for prompts that never had
a tail. The 4108 row: CAND 2979 / 3049 / 3084 / 3049, FIX 3511 / 3545 / 3564 / 3554.

**Serve sweep with exact prompt lengths** (`raw/.../ab1/`, prompts of 4089 / 4103 / 4110 / 4115 / 4143 tokens, 6 rounds, both
chunk sizes; GPU at 1350-1400 MHz, so the scaled column is the usable one): tail 2-15 at chunk 2048 1230 -> 1097 ms (fast-only
1339 -> 1195), at chunk 256 1317 -> 1186 (1446 -> 1290); tail >= 1 chunk and 16-63 unchanged (1044 vs 1042, 1102 vs 1101 at 2048).

## 5. Output

* Greedy ids FIX vs CAND identical, 48 and 256 tokens, on the short prompt, the 4062-token prompt, and exact 4086 / 4107 /
  4110 / 4120-token prompts (`raw/.../gates/identity/summary.txt`: 12 of 12).
* Chunk invariance at n = 4107 (tail 11): top logits after the prompt (`identity-logits/`):
  CAND at chunk 2048, 4096, 1024, 256 (all leave a tail of 11) 803:20.4155 374:20.1295 ...; CAND at 8192, 1369, 2054 (no tail)
  803:20.4312 374:20.1308 ...; **FIX at 2048, 4096, 1024, 256, 8192, 1369, 2054: all 803:20.4312 374:20.1308 ...**, the
  tail-free value. So before, a 2..15-row tail changed the logits' low bits (greedy ids the same); now no chunking does.
  At n = 4086 (no tail) CAND and FIX are identical at every chunk size.
* So the change is bitwise-neutral for prompts without a 2..15-row tail (the case of every earlier measurement and of
  the existing baselines' long prompts) and moves a tail-carrying prompt to the value a tail-free chunking gives. Short
  prompts of 2..15 tokens are the same case (one chunk of M < 16): the golden "decode (teacher-forced)" KL moved
  3.496e-4 -> 3.313e-4 (short) and 3.517e-4 -> 3.371e-4 (long), most likely from the goldens' shortest prompts; I did not
  isolate which prompts. The shipped Metal baselines/envelopes (`golden/baselines/*.metal.*.json`) were recorded before this
  change; the gate passes against the existing ones as run (below) and I did not re-record them.

## 6. Gates (FIX binary 548bc7305b1ec8dd6961189d291ee932, all under fleetq #1975; `raw/.../gates/`, `gates.sh`)

| gate | result |
|---|---|
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | exit 0, all suites ok (tests extended: GEMM cases M = 2, 11, 15; row independence with 2/11/15-row calls; registration accepts M = 2 and 11, not 1) |
| `intarsia kernels test --backend metal -v` | exit 0, 23 kernels, 291 cases (was 272), all ADMITTED |
| accuracy short golden, bf16 class | prefill top-1 100.00 %, KL 2.832e-4 PASS; decode top-1 100.00 %, KL 3.313e-4 PASS (CAND 2.832e-4 / 3.496e-4) |
| accuracy long golden, bf16 class | prefill top-1 **99.72 %** (>= 99.6), KL 3.699e-4 (<= 5e-4) PASS; decode top-1 100.00 %, KL 3.371e-4 PASS (CAND 3.699e-4 / 3.517e-4) |
| greedy ids vs CAND | 12 of 12 identical, see section 5 |

Not run: `qualify` / the 4k golden (not in this branch; #1977 is running the CPU 4k golden), the full-prompt-length plain
`generate` ABBA, and tests for CUDA/HIP (not touched: the change is in the Metal crate only, `git diff --stat` below).

## Follow-ups

1. Balanced chunking in `generate_one` (equal-size chunks, e.g. 3 x 1369 for n = 4107) would remove the last 55 ms (5 %)
   for tails of 2..63 rows. It is a core change (all backends, snapshots, prefix reuse) and should be a per-backend recipe
   option; I measured nothing for it, the 55 ms is the gap above.
2. Comparing builds on localeval: log the prompt tokens per rep and compare only reps with the same last-chunk class
   (and the same clock class). A pass whose 4k prompt exceeds 4099 tokens carries a tail.
3. The slow-clock state (docs/results/2026-10-15-metal-serve-prefill.md) is a separate noise source and was present again
   in this session (run1 p1, run2 hour at 1350-1450 MHz); the "fast" columns above discard it, the scaled column removes
   it to first order.

## Files

`raw/2026-10-15-metal-p3-noise/`: `final-clock-4k-reps.md`, `localeval-ab.md`, `localeval-marginal.md`,
`localeval-results/` (8 localeval runs), `le1/ ab1/ run1/ run2/` (trace stderr, macmon.jsonl.gz, per-pass logs, summaries),
`kernel-profile/`, `identity-logits/` (stderr with top logits and ids, the exact-length prompts), `gates/`, `gates.sh`,
`scripts/` (drivers `passes*.sh`, `exp2.py`, analysis `an*.py`), `prefill-trace-plen.patch`, `binaries-md5.txt`.
Code: `crates/intarsia-backend-metal/src/gemm_nax.rs` (`NAX_MIN_M`), `src/kernels.rs` (comment), `tests/gemm_nax.rs`.
