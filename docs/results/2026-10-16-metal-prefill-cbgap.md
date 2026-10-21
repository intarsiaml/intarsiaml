# Metal prefill: where the "idle between command buffers" at chunk 256 comes from (2026-10-16)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main in `3bdd6804` (Merge metal-weight-residency b8d7f7d); "not pushed" / "nothing merged" wording in this page describes the state when written.

> Follow-up `2026-10-16-metal-weight-residency.md`: the pre-wire described here was replaced by a keep-warm thread (residency set re-requested every 400 ms) that
> removes the idle-gap stall for `serve`; `set_prewire` and `INTARSIA_METAL_PREWIRE` no longer exist.

Branch `metal-prefill-cbgap`, trimmed and rebased onto `spike/metal` 00abb36 (first version was off 70b15eb; the staged-upload
experiment was removed at the coordinator's request, see Trim). Qwen3-8B BF16, endor (M5 Max), `--backend metal`.
Raw logs and scripts: `docs/results/raw/2026-10-16-metal-prefill-cbgap/` (per-run traces of the first round are in
`traces.tar.gz`: `tar xzf traces.tar.gz` recreates `speed2/`, `cold/`, `diag/`, `base-gaptrace/`, which `speedsum2.py`,
`reqgaps.py`, `coldsum.py` read). Code: `crates/intarsia-backend-metal/src/{host,lib,profile}.rs`, `crates/intarsia-cli/src/common.rs`,
test `tests/prewire.rs`, examples `wire_probe.rs`, `pso_nax_attn.rs`. Not pushed, not merged. No CUDA/HIP file changed; `core` untouched.

## Outcome

1. **The 54 ms of "GPU idle between 48 command buffers" at chunk 256 (gemm-midm doc) is not a per-chunk cost.** It is the cold
   first use of the mmap'd weight shards: the driver wires a no-copy `MTLBuffer` the first time a command buffer uses it, at
   about **10 ms per GB** (38 to 45 ms for each 4 GB shard, 173 ms for all 15.6 GB, `wire/`). The first command buffer starts
   97 ms (chunk 256) to 140 ms (chunk 2048) after its commit; at chunk 256 the next two buffers each wait another 20 to 25 ms
   (other shards) behind a buffer (23 to 31 ms of GPU work) that is too short to hide it. At chunk 2048 the total is the same
   ~140 ms, but the 177 ms buffers hide most of it, so `span - busy` leaves it out. The old metric charged chunk 256 for a cost
   both chunk sizes pay.
2. **Warm, the idle between chunks is small**: 15 drains of 0.3 to 0.5 ms at 4k (5.4 to 8.5 ms of 1150, 0.5 %), 1.1 to 1.4 ms at
   1k. The host is never the bottleneck inside a chunk (encoding a 128-dispatch buffer 0.06 to 0.20 ms, `commit()` 0.01 ms, the
   2nd and 3rd buffer of a chunk committed 22 to 46 ms before the GPU reaches them). The only stall is the chunk boundary:
   `Model::run_layers` uploads the next chunk's token ids with `Backend::upload`, which on Metal drains the stream first.
3. **A pre-wire removes the cold stall when work starts right after the load, and does nothing for `serve`.**
   * `MetalBackend::set_prewire(true)` makes `wrap_host` commit a one-byte blit out of each new whole-file buffer, so the wiring
     runs while the loader carries on. Fresh `generate` with it forced on: cold prefill wall at 4062 tokens **-115 ms at chunk
     256, -89 ms at 2048** (1k: -112, -90), GPU busy unchanged (the pre-trim measurement, below).
   * **The wiring does not survive an idle gap.** `wire3/`: pre-wired shards are fast after a 300 ms gap (0.2 to 1.4 ms) but cost
     the full 40 ms / 4 GB again after 3 s, 10 s and 40 s, exactly like never-pre-wired shards. A server idles for 40 s of
     loading and then for seconds between requests, so it never sees the gain. Measured (`serve-cold/`, fleetq #2162): the
     first 4062-token request after startup takes 1149 ms with the pre-wire, 1149 ms with `INTARSIA_METAL_PREWIRE=0` and 1151 ms
     on 00abb36, at a 0 s and a 3 s gap (paired differences +0.1 .. +4 ms, noise). Its first command buffer starts 102 ms
     (off) / 104 ms (on) after its commit: the stall is there in both.
   * Consequence the coordinator should weigh: **with the pre-wire enabled only for serve, as requested, it has no measurable
     effect anywhere.** The `generate` gain exists only for the (default-off) `INTARSIA_METAL_PREWIRE=1` path. The same data show
     that a serve instance pays ~170 ms of re-wiring on the first request after any idle period of a few seconds or more, with
     or without this branch. Keeping the weights resident (an `MTLResidencySet`, or a keep-alive touch) is the thing that would
     help serve; I tested a residency set only back to back with the first use (`wire/residency-*`, same cost at the call), not
     across an idle gap.
4. **What is left of the 256-chunk cost is GPU busy time.** In a fresh process at 4062 tokens chunk 256 costs ~105 ms (+10 %) over
   chunk 2048 and GPU busy is +10 % (1144 against 1041 ms at a steady 1617 MHz); at 1024 tokens +8 ms wall (+3 %), +4.7 % busy.
   Attention explains ~23 ms of the 4k gap (isolated, below); the other ~80 ms is not attention and the isolated GEMM bench
   (0.97x) does not show it. It needs per-op `cb`-mode accounting inside the engine.
5. **256-row attention is a latency / wave-fill property, not a tile choice, and I found no bitwise-safe change worth making.**
   Analysis below.

## Trim (follow-up from the Metal spike session)

* Removed: staged upload (`upload_staged`, `set_staged_upload_ahead`, `INTARSIA_METAL_AHEAD`, `tests/staged_upload.rs`). Its
  measured effect was idle 5.9 to 0.03 ms at 4k chunk 256 and wall +6.8 +- 6.3 ms (unresolved); the pre-trim data are kept in
  `speed2-summary.md` (warm A/B section) and `gates/`, `gates-conc/`.
* Pre-wire is opt-in on the backend (`set_prewire`, default off). `ModelArgs::for_serving` (used by `serve` only) sets
  `prewire = Some(true)`, like the serve default `--profile-every 0`. `INTARSIA_METAL_PREWIRE=0` / `=1` override in both
  directions for every command (`prewire_wanted`, unit-tested on all platforms). `generate`, `plan`, `doctor`, `kernels`, the
  accuracy tools: no pre-wire, so no extra stall when work starts right after the load and no wait in `Drop` (they run the
  base code path).
* `profile.rs` host timestamps (`ho`/`hc`/`hr` in `cb` records) kept: they separated host stalls from GPU wiring latency here.
* Rebased onto 00abb36 and squashed into three commits (backend, cli, docs).


## Where the idle is (base, `ts` = base plus the new host timestamps, `diag/`, `base-gaptrace/`)

`INTARSIA_METAL_PROFILE=cb:` records now also carry `ho`, `hc`, `hr`: host time at which the buffer's first dispatch started to
be encoded, and `commit()` entered / returned, on the clock of `GPUStartTime`. One process, requests of 4062 and 1024 tokens
back to back (`cbgap-bench`, serving path `Engine::generate`, no prefix reuse):

| request | chunk | cbs | busy ms | idle in span ms | lead ms (request start to first GPU work) | the idle is |
|---|---|---|---|---|---|---|
| 4062, first in process | 256 | 48 | 1144 | 50.1 | **96.8** | cb 1 waits 19.5, cb 2 waits 24.8; 5.8 ms in sub-ms gaps |
| 4062, second / third | 256 | 48 | 1138 / 1155 | 8.5 / 5.9 | 0.3 | 15 chunk boundaries x 0.3..0.5 ms (+ one 3.1 ms outlier) |
| 1024, second / third | 256 | 12 | 268 / 265 | 1.4 / 1.2 | 0.3 | 3 boundaries |
| 4062, first in process | 2048 | 6 | 1069 | 0.5 | **139.6** | one boundary, 0.46 ms |
| 4062, second / third | 2048 | 6 | 1205 / 1116 | 0.5 / 0.5 | 0.3 | one boundary |

(Busy differs between rows mostly through the GPU clock, 1250 to 1617 MHz between processes, see below.)

Per-buffer view of the first request at chunk 256 (`gap-view-ts.txt`, from `reqgaps.py -v`): buffer 0 is committed 0.2 ms after
the request starts and the GPU begins on it 96.5 ms later; buffers 1 and 2 are committed 0.1 ms after buffer 0, and start 147 and
195 ms later. That is GPU-side latency, not encoding.

First use of the weight shards (`wire_probe`, `wire/none-*`, 5 shards of Qwen3-8B, a 16-byte GPU copy out of each):

| shard | size GB | first use ms | second use ms |
|---|---|---|---|
| model-00001 | 4.00 | 38.6 / 41.7 | 0.6 |
| model-00002 | 3.99 | 45.4 / 46.0 | 0.8 |
| model-00003 | 3.96 | 40.7 / 34.7 | 0.3..0.8 |
| model-00004 | 3.19 | 37.1 / 36.2 | 0.7 |
| model-00005 | 1.24 | 11.4 / 11.7 | 0.7 |

A touch pass or an `MTLResidencySet` (`requestResidency`) pays the same ~170 to 180 ms at the call (`wire/touch-*`,
`wire/residency-*`), then every first use is 0.2 ms: the cost is real, it can only be moved. The backend's pre-wire commits
the blit and returns (`wrap_host` of all 5 shards: 1.8 ms); `wire2/` with a delay between wrap and first use: 0 ms delay,
first use 178 to 185 ms; 100 ms, 70 to 90 ms; 300 ms, 0.8 ms. `generate` spends more than that between load and the first launch
(model setup, tokenizing the prompt); a server idles for much longer.


## What changed (final)

* `host.rs`: `prewire` (one-byte blit into a 16-byte scratch buffer, committed on the queue, not part of the stream) for every new
  whole-file buffer when `MetalBackend::set_prewire(true)`, up to half of the device's `recommendedMaxWorkingSetSize` in total.
  The command buffers are kept in `HostCache`; `Drop` waits for them before the mappings go away (the blit reads the mmap'd file),
  `synchronize` forgets finished ones; a failed one is reported once on stderr and ignored. GPU work committed after the
  pre-wire (the admission run at load) queues behind it. Off by default: with it on, a process that wraps weights and exits at
  once waits up to ~180 ms in `Drop`, and work that starts with no gap after the load waits for all five shards (about 40 ms
  worse than base, `wire2/` sleep 0).
* `lib.rs`: the `prewire` flag, `set_prewire`, `prewired_bytes`, `drain_prewire` in `Drop` / `synchronize`; host timestamps for
  the profile.
* `crates/intarsia-cli/src/common.rs`: `ModelArgs.prewire`, `prewire_wanted`, `open_backend_with`; `for_serving` sets it.
* No kernel, launch geometry or op order; `core` untouched. `LAUNCHER_SOURCES` hashes `lib.rs`, so cached admission results are
  invalidated once per binary.
* Tests: `tests/prewire.rs` (bytes read back before and after the wiring finishes, drop right after wrap, off by default and
  switchable), `common::tests::prewire_is_on_for_serve_only_and_the_env_overrides_both_ways`.

## Gates of the trimmed branch (HEAD after rebase on 00abb36)

| gate | result |
|---|---|
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` (fleetq #2161) | 106 passed, 0 failed |
| `intarsia kernels test` (#2161) | 27 kernels, 321 cases, all ADMITTED (`gates2/kernels-test.txt`) |
| CLI `generate` greedy ids, 48 tokens, L = 2049 (p2049s1, p2049s2) vs 00abb36 binary, chunk 256 and 2048 (#2161) | ids identical in 4 of 4, stdout byte-identical |
| `cargo check --workspace --all-targets` (plain, and with `intarsia-cli/metal`) | pass |
| `cargo test -p intarsia-cli --lib common` (with and without `--features metal`) | pass incl. the new prewire test |
| CUDA/HIP `cargo check -p intarsia-cli --tests` with `--features cuda` and `--features hip`, spark1 | pre-trim HEAD passed (#2152, #2156); not rerun after the trim, the trim touches `common.rs` only through code that compiles on all platforms (`cargo check --workspace` above) |

The pre-trim sweep (serving-path logits 1056 rows x 2 chunk sizes, `intarsia accuracy` short/long/4k, 812 concurrent-vs-solo
comparisons, #2153, #2154) ran on the version with staged upload and is not rerun, as agreed. The trim only removes code from
it and makes the remaining code opt-in.


## 256-row prefill attention (`attn/`, fleetq #2145; `flash_bench`, 32/8 heads, head_dim 128, 36 layers, NAX flash kernel)

Isolated, whole 4062-token prompt, attention only, summed over its chunks (default variant `attn_nax_d128_s4`):

| chunk | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| ms (36 layers) | 125.7 | 112.3 | 109.5 | 102.6 |
| TFLOPS | 38.7 | 43.3 | 44.5 | 47.4 |

So chunk 256 costs **+23 ms (+22 %)** in attention at 4k, not the +48 ms of the serialised kernel-mode trace in the gemm-midm
doc. Per layer, chunk T against 3840 cached positions (`attn/chunk-sweep.txt`), s4 / s8, microseconds:

| T | 128 | 192 | 224 | 240 | 256 | 272 | 288 | 320 | 384 | 512 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s4 | 374 | 386 | 392 | 399 | 402 | 409 | 411 | 426 | 681 | 760 | 3473 |
| s8 | 394 | 399 | 406 | 426 | 427 | 732 | 734 | 730 | 701 | 824 | 3543 |

`attn/pso.txt`: s4 has 4 simdgroups per threadgroup (`maxTotalThreadsPerThreadgroup` 128, no threadgroup memory) and makes
`T/16 x 8` threadgroups: 128 at T = 256, 160 at T = 320, 192 at T = 384, 1024 at T = 2048.

Reading: time per layer is flat from T = 128 to 320 (374 to 426 us) and doubles at T = 384. One simdgroup owns 16 query rows of one
head and walks the whole key range in 32-key tiles; its dependent chain (load K, MMA, softmax, load V, MMA) sets the time of a
wave. About 160 threadgroups of 128 threads (4 per core, 640 simdgroups) are resident (T = 320 fills them, T = 384 spills into a
second wave). At T = 256 the 512 simdgroup units fill 80 % of one wave and the layer takes one unit's latency: 41.4 TFLOPS,
against 49.3 at T = 320 and 47 to 50 at chunk 2048 (`pos-sweep.txt`: 17 / 34 / 37 / 40 / 41 TFLOPS at pos0 0 / 512 / 1024 /
2048 / 3840 for chunk 256, 50 / 50 / 49 / 48 / 47 for chunk 2048). It is an under-filled single wave of a latency-bound kernel,
not a tile that is too big or too small: s8 (64 threadgroups at T = 256) is 6 % slower and falls off a cliff at T = 272.

Bitwise-safe options: the variant-equality assert in `flash_chunk_invariant` shows grouping of (head, row slab) units into
threadgroups does not change bits, but regrouping cannot add units: the 512 units already fit one wave, so a threadgroup per query
head would not fill the missing 20 % and would lose the K/V sharing inside a GQA group. Anything that adds parallelism (split-KV,
a different key tile BK, a different rescale cadence) changes the accumulation order of a row and so its bits. What keeps the
arithmetic and shortens the wave is a shorter per-unit chain (prefetching the next key tile's K / V fragments while the current
one is in the MMA). Ceiling if chunk 256 reached the T = 320 efficiency: 15 % of 125.7 ms, about 19 ms of a 1150 ms prefill
(1.6 %). I did not implement it: it needs a kernel rewrite whose gain I cannot promise to be clearly measurable on a box that
swings 3 to 5 % per request, and it would not touch the ~80 ms of non-attention busy time.


## Speed

### Serve, cold first request (the trim's gate; `serve-cold/`, `serve-cold-summary.md`, fleetq #2162)

6 rounds, arm order rotated, one fresh `intarsia serve` per sample (chunk 2048, `--max-batch 1`), 4062-token completion with
`max_tokens 1` sent right after `/health` (gap 0) or 3 s later, timed with curl; arms: 00abb36, new with the serve default
(pre-wire on), new with `INTARSIA_METAL_PREWIRE=0`.

| gap s | arm | request wall ms (median [min..max]) |
|---|---|---|
| 0 | 00abb36 | 1151 [1146..1157] |
| 0 | new, `PREWIRE=0` | 1149 [1146..1152] |
| 0 | new, serve default | 1149 [1146..1152] |
| 3 | 00abb36 | 1153 [1147..1158] |
| 3 | new, `PREWIRE=0` | 1151 [1147..1155] |
| 3 | new, serve default | 1154 [1150..1161] |

New minus `PREWIRE=0`, paired: +1.2 ms (gap 0), +4.0 ms (gap 3), both inside the noise. **Serve does not get the gain.** First
command buffer, commit to GPU start: 102 / 104 ms (gap 0, off / on), 104 / 107 ms (gap 3). Load time 38.6 to 38.8 s in all arms.

### Why: the wiring is lost while idle (`wire3/`, fleetq #2168)

`wire_probe`: wrap the five shards, optionally pre-wire, sleep, then time the first 16-byte GPU copy out of each shard (ms per
shard, shards 1 to 5; 4.0, 4.0, 4.0, 3.2, 1.2 GB).

| pre-wire | sleep | shard 1..5 first use, ms |
|---|---|---|
| off | 0.3 s | 74 / 42 / 36 / 19 / 8 (run 1), 55 / 44 / 30 / 19 / 8 (run 2) |
| on | 0.3 s | 0.8 / 0.2 / 0.2 / 0.2 / 0.3, 1.4 / 0.4 / 0.4 / 0.2 / 0.2 |
| off | 3 s | 74 / 40 / 30 / 19 / 8, 86 / 40 / 31 / 20 / 9 |
| on | 3 s | 44 / 49 / 32 / 19 / 9, 51 / 54 / 35 / 25 / 9 |
| on | 10 s | 49 / 41 / 30 / 19 / 8, 60 / 44 / 30 / 19 / 8 |
| on | 40 s | 49 / 43 / 32 / 20 / 8, 53 / 47 / 32 / 20 / 9 |
| off | 40 s | 48 / 41 / 32 / 21 / 9, 49 / 45 / 33 / 21 / 9 |

### Generate with the pre-wire forced on (pre-trim measurement; now `INTARSIA_METAL_PREWIRE=1` only)

`speed2/` (fleetq #2155), fresh `intarsia generate` per sample, 8 rounds Latin square, burn-in before each cell, macmon over the
whole job (median GPU clock per arm and cell is in the first table). Arms: base (70b15eb), n1 (pre-wire always on, plus
staged upload), n1np (`PREWIRE=0`, equals base), n1ns (no staged upload; equals n1). The pre-wire alone is the gain:

Cold (fresh process), paired by round, median [min..max]; "new" = pre-trim n1:

| tokens | chunk | arm | prefill wall ms | GPU busy ms | idle in span ms | cbs | GPU MHz |
|---|---|---|---|---|---|---|---|
| 1024 | 256 | base | 396.5 [386..409] | 271.6 | 23.04 | 12 | 1553 |
| 1024 | 256 | new | **287.0** [280..290] | 271.7 | 0.01 | 12 | 1453 |
| 1024 | 2048 | base | 366.5 [355..390] | 259.7 | 0.00 | 3 | 1407 |
| 1024 | 2048 | new | **276.0** [272..280] | 259.5 | 0.00 | 3 | 1258 |
| 4062 | 256 | base | 1275.0 [1268..1370] | 1143.7 | 27.68 | 48 | 1617 |
| 4062 | 256 | new | **1159.5** [1156..1232] | 1144.1 | 0.02 | 46 | 1617 |
| 4062 | 2048 | base | 1149.0 [1139..1181] | 1042.2 | 0.49 | 6 | 1617 |
| 4062 | 2048 | new | **1057.5** [1053..1120] | 1041.1 | 0.00 | 6 | 1617 |

| tokens | chunk | new - base wall ms | new / base wall | no-prewire - base wall ms | no-staging - new wall ms | new - base busy ms |
|---|---|---|---|---|---|---|
| 1024 | 256 | -111.5 [-121..-99] | 0.721 | +11.5 [-17..25] | +1.5 [-5..8] | -0.1 |
| 1024 | 2048 | -89.5 [-117..-80] | 0.756 | -3.0 [-26..12] | 0.0 [-6..8] | -0.2 |
| 4062 | 256 | -115.5 [-138..-107] | 0.909 | 0.0 [-52..41] | +1.5 [-39..36] | -2.0 |
| 4062 | 2048 | -89.0 [-107..-34] | 0.923 | +4.5 [-10..31] | -1.0 [-45..3] | -0.9 |

The whole cold gain is the pre-wire (the no-prewire arm equals base, the no-staging arm equals new). GPU busy does not move: the
kernels are the same. The pre-wire saves more at chunk 256 than 2048 at 4k by the 26 ms of cold gaps it removes.

Whole `generate` process wall (start to exit, markers.txt), new minus base, median over 8 rounds: -91 ms (1k, chunk 256), -106 (1k, 2048),
-78 (4k, 256), -62 (4k, 2048). The pre-wire overlaps with load, so end to end it is a smaller gain than the prefill number alone,
but still a gain.

Warm "before" is the `off` row of the A/B below: with the weights already wired the pre-wire does nothing, and the only other
difference from base is the profile timestamps (profiling off: no code runs). So there is no separate warm base arm.

Extra cost of chunk 256 over 2048 inside an arm (paired by round):

| tokens | arm | wall c256 / c2048 | c256 - c2048 wall ms | busy c256 / c2048 |
|---|---|---|---|---|
| 1024 | base | 1.088 [1.015..1.111] | 32.0 | 1.046 |
| 1024 | new | 1.029 [1.007..1.055] | 8.0 | 1.047 |
| 4062 | base | 1.115 [1.099..1.187] | 131.5 | 1.097 |
| 4062 | new | 1.100 [1.091..1.102] | 105.5 | 1.099 |


The warm staged-upload A/B (idle 5.9 to 0.03 ms, wall +6.8 +- 6.3 ms at 4k chunk 256, unresolved) is in `speed2-summary.md`; that code is removed.

Not used as the metric, as asked: `localeval` TTFT.

## Not done / uncertain

* **The pre-wire has no user in the shipped configuration** (serve only, as decided): the wiring is lost during idle, so the
  serve default is dead weight (costs a wait in `Drop`, nothing gained). Options for the merger: (a) drop `host.rs` prewire and the
  CLI switch and keep only the timestamps and the analysis; (b) keep it as the opt-in `INTARSIA_METAL_PREWIRE=1` for short
  `generate`-style runs; (c) look for something that keeps the weights resident across idle (residency set or keep-alive),
  which would help serve's first request after any pause (~170 ms of wiring on top of the prefill).
  Untested: whether an `MTLResidencySet` survives an idle gap.
* Warm chunk-256 cost is unchanged by this branch (GPU busy +10 % at 4k); the 5.9 ms of boundary idle is not fixed after the
  staged-upload removal.
* Greedy ids at L = 1000 were covered only by the pre-trim harness serving ids (`docs1000`); the trimmed-branch id gate is L = 2049.
* Per-layer attention cost is flat from T = 128 to 320 and bitwise chunk invariance was verified only at 256 and 2048. A scheduler
  chunk of ~320 while decoding would fill the wave; that lives in `scheduler.rs` (off limits here, metal-cb-fix), and invariance at
  320 is unverified.
* `Model::forward` hashes for `code257` / `code513` differ between chunk 256 and 2048 in base and new alike (see
  `gates/compare.txt`, pre-trim); another session (metal-forward-hash) is working on it.
* The CUDA/HIP check was not rerun after the trim (see Gates).
* macmon pid 25718 (started 2026-10-15 20:03, before my first job) is not mine.

## fleetq jobs (all endor exclusive unless noted)

Pre-trim: #2137 baseline gap trace; #2144 cold / warm gap traces; #2145 attention sweeps; #2146, #2147 shard first-use and pre-wire
probes; #2148 cold generate; #2149 first speed run (superseded); #2152, #2156 CUDA/HIP `cargo check` (spark1); #2153 bitwise gates;
#2154 concurrent identity; #2155 position-balanced speed. Cancelled: #2142, #2143 (queued, runner killed), #2150, #2151 (killed
and rerun as #2153 / #2154).
Trim: #2161 (cargo test metal, kernels test, generate ids), #2162 (paired cold serve), #2168 (idle-gap probe).

## Reproduce

```
cargo build --release -p intarsia-cli --features metal        # copy target/release/intarsia away; other agents rebuild
fleetq run --host endor --exclusive --mem 24 --label "..." -- docs/results/raw/2026-10-16-metal-prefill-cbgap/serve-cold.sh OUT 6
python3 docs/results/raw/2026-10-16-metal-prefill-cbgap/servecoldsum.py OUT
fleetq run ... -- docs/results/raw/2026-10-16-metal-prefill-cbgap/wire3.sh        # idle-gap probe
fleetq run ... -- docs/results/raw/2026-10-16-metal-prefill-cbgap/gates2.sh OUT   # tests, kernels test, ids
```
(`serve-cold.sh`, `gates2.sh`, `wire3.sh` read binaries from the author's scratch dir: `intarsia-base2` = 00abb36, `intarsia-t1` =
trimmed HEAD, `wire_probe3`.)
