# Metal: keeping the weight wiring warm for `serve` (2026-10-16)

> **Update (2026-10-21, docs-accuracy pass, see `docs/decisions/LOG.md`):** Merged to main as PR #1 (`55bb1087`; the branch merge `3bdd6804`, Merge metal-weight-residency b8d7f7d, is inside it); "not pushed" / "nothing merged" wording in this page describes the state when written. C17: the keep-warm TTFT gain is quoted three ways (about 110 ms here, -58.5 ms at 10 s on the final build, -30 ms median at 10 s in `2026-10-17-metal-relcheck-main.md`); the relcheck never reproduced 110 ms. Use the relcheck numbers: -30 ms median at 10 s idle, -56 ms at 40 s. Keep-warm is on in `serve` (`INTARSIA_METAL_KEEPWARM` overrides).

Branch `metal-weight-residency`, rebased onto `spike/metal` 6d99a6f (includes the forward-hash head-grouping fix 6c077aa). Qwen3-8B BF16
(`/Users/josef/models/Qwen3-8B`), endor (Apple M5 Max, macOS 27.0 build 26A428, Metal 4), `--backend metal`. Not pushed, not merged. Metal
backend only: no CUDA/HIP file changed, `core` untouched.
Raw logs and scripts: `docs/results/raw/2026-10-16-metal-weight-residency/` (summaries: `probe-summary.txt`, `idle6-summary.txt`,
`itl-summary.txt`, `cbstart-summary.txt`, `power-summary.txt`). Follows `2026-10-16-metal-prefill-cbgap.md`, which found the stall.

**Status: done, with one gap in the requested gate.** A residency set that is re-requested every 400 ms from a background thread cuts the first 4k
request after idle by about 110 ms (TTFT 1165..1180 ms down to 1056..1063 ms) and it is the `intarsia serve` default; every other command is unchanged.
Pre-wire is gone (coordinator's instruction). Which build measured what:

* 3 / 10 / 40 s idle, 6 paired rounds each, all in one slot (#2178): measured on a **pre-final** build (`intarsia-kw`: the residency flavour selected with
  `INTARSIA_METAL_KEEPWARM_MODE=residency`, the old pre-wire code present but switched off). 36 of 36 paired differences negative.
* The **shipped** build (`intarsia-new`, residency by default, pre-wire code removed) was measured with 4 paired rounds at 3 s and 40 s only (the ITL control
  below, also #2178). The 10 s idle case was **not** run on the shipped build; the residency loop is the same code in both builds. Resume command under "Not done".
* All other gates ran on the shipped build.

## Mechanism

* Weights are `memmap2` mmaps of the safetensors shards, wrapped as one `MTLBuffer` per shard with `newBufferWithBytesNoCopy`,
  `StorageModeShared`, default (tracked) hazard mode (`crates/intarsia-backend-metal/src/host.rs`). Before this branch no `MTLResidencySet`,
  `useResource(s)` (only the ICB path in `decode.rs` uses it) or heap touched them.
* `objc2-metal` 0.3.2 exposes `MTLResidencySet`, `MTLResidencySetDescriptor`, `MTLAllocation`, `newResidencySetWithDescriptor_error`,
  `addAllocation`, `commit`, `requestResidency`, `allocatedSize`, and `MTLCommandQueue::addResidencySet / removeResidencySet`. Nothing else was needed.
* The driver wires a buffer for the GPU on its first use (~10 ms/GB, about 150 ms for the 15.6 GB of Qwen3-8B) and **drops that wiring about one
  second after the buffer's last use**. The decay is per buffer, not per device and not per queue, and it is not CPU page wiring:
  * a plain GPU touch of a shard held at a 1.0 s gap and was lost at 1.2 s (`probe3/touch-g1000..1800`, `probe2/touch-g*`);
  * touching shard 1 only every 200 ms kept shard 1 and lost shards 2..5; touching no weight buffer kept nothing (`probe2/keep-g10000-k200-first|none`).
* The cbgap doc measured the cost as ~102 ms before the first command buffer of a cold 4k prefill starts, plus ~25 ms before each of the next two.

## What was tried (idle-gap probe, `examples/wire_probe.rs`, first GPU use of the five shards after the gap)

Time in ms for the first use of shards 1..5 (4.0 / 4.0 / 4.0 / 3.2 / 1.2 GB) and their sum. A wired shard costs 0.1..1 ms. "never wired" is the cold
cost, 145..185 ms in total. All runs inside fleetq slots #2169 (a), #2170/#2171 (b), #2176 (c).

| approach | gap 0.3 s | 3 s | 10 s | 40 s | verdict |
|---|---|---|---|---|---|
| nothing, cold (reference) | 155 | 150 | 185 | 145 | first use costs ~150 ms |
| **a1** `MTLResidencySet` on the queue, `requestResidency` once, no re-request | 10 | 233 | 213 | 226 | worse than doing nothing after >= 3 s (the whole set is re-wired by the first command buffer) |
| **a2** same set, `requestResidency()` re-called every 500 ms (CPU only) | n/a | 4.5 | 7.5 | 4.8 | **removes the stall** (`probe4/residencycpu-*`) |
| a2 refresh every 200 ms / 1 s / 2 s, gap 10 s / 40 s / 40 s | | | 8.5 | 10.0 (1 s), 5.7 (2 s) | holds |
| a2 refresh every 5 s, gap 40 s | | | | 210 | lost (the refresh itself re-wires, ~197 ms per call) |
| a2 no refresh after an initial touch pass | | 204 | 218 | | lost, same as a1 |
| **b** `mlock` of all five mappings (CPU wiring, ~290 ms) | 160 | 171 | 164 | 170 | no effect: the cost is GPU-VM wiring, not page-level |
| **b** `madvise(WILLNEED)` + read pass | 160 | 163 | 161 | 156 | no effect |
| **c** copy into fresh shared `MTLBuffer`s (touched once, then idle) | n/a | 368 | 273 | 240 | no: fresh buffers decay the same way, load 1.7 s, +15.6 GB resident (page cache not counted) |
| **c** copy into `StorageModePrivate` buffers by GPU blit (touched) | n/a | 279 | 296 | 259 | no: decays too, load 0.7 s, +15.6 GB |
| **d** GPU touch of every shard, one command buffer, every 200 ms | 1.1 (no keep-warm yet, gap too short to decay) | 2.3 | 2.2 | 2.7 | holds (`probe2/keep-*`) |
| d every 500 ms / 1000 ms, gap 10 s | | | 1.2 / 1.4 | | holds; 2000 ms loses it (every touch pays ~150 ms) |
| d touching shard 1 only / no weight buffer | | | 121 / 176 | | partial / nothing |

Notes: (c) was first measured wrongly in the first probe run (`probe1/copy-*`: nothing had touched the new buffers before the gap) and redone as `copyshared`
/ `copyprivate` in `probe4` after the advisor's review. The a1 first-use cost of 220..230 ms is larger than the 150 ms cold cost because the whole
set (15.6 GB) is made resident by the first command buffer, not just shard 1. The initial `requestResidency` of a2 blocks for 150..175 ms (all shards
wired), so it must not run on the loader thread.
Rows for approaches that have a 0.3 s column come from `probe1`; `probe-summary.txt` has every run.

**Chosen: a2.** It needs no GPU work at all (the refresh is a ~15..20 us CPU call, measured mean 0.013..0.020 ms), has no extra memory (the set accounts
the same 15622 MB the mmaps already are), and no load cost on the loader thread. d works too and is kept as the fallback (below) for a macOS without
`MTLResidencySet`.

## What is in the code

* `MetalBackend::set_keepwarm(period_ms)` / `set_keepwarm_mode(period_ms, KeepMode::{Residency, Blit})` (`host.rs`). One thread `metal-keepwarm`:
  * `Residency` (default): all whole-file weight buffers go into one `MTLResidencySet` attached to the engine's queue; `requestResidency()` is called
    when a file is registered (this is the 150..175 ms wiring, in the thread) and every `period_ms` while the engine is quiet.
  * `Blit`: a one-byte blit out of every whole-file buffer in a command buffer on a queue of the thread's own (never the engine stream), committed,
    not waited for, never a second one while the previous has not run. Used when `respondsToSelector(newResidencySetWithDescriptor:error:)` is false.
  * "Quiet" = no engine commit for half a period (`Stream::last_commit`, one atomic store per commit); a registration always acts at once.
  * Bounded to half of `recommendedMaxWorkingSetSize` (the old pre-wire limit); files over `maxBufferLength` (per-tensor buffers) are not covered.
  * `Drop`: the thread is stopped and joined (and, for `Blit`, its last command buffer awaited) before the mappings can go; the registry holds the
    mapping `Arc`s so a queued touch can never read unmapped pages.
* CLI (`crates/intarsia-cli/src/common.rs`): `ModelArgs.keepwarm`, `ModelArgs::for_serving` sets it (serve only), `keepwarm_period_ms`
  (400 ms), `INTARSIA_METAL_KEEPWARM` overrides either way (`0|off|false|no`, `1|on|true|yes`, or a period in ms clamped to 50..1500 with a warning; anything else is warned about and ignored), `INTARSIA_METAL_KEEPWARM_MODE=blit` picks the GPU-touch flavour.
  `generate`, `plan`, `doctor`, `kernels` and the accuracy tools carry no default: no thread, no residency set, behaviour as before.
* Removed (coordinator's instruction, the keep-warm covers it): `set_prewire`, `prewired_bytes`, `drain_prewire`, the `for_serving` pre-wire default,
  `INTARSIA_METAL_PREWIRE`, `tests/prewire.rs`. Kept: `profile.rs` host timestamps (`ho/hc/hr` in `cb` records), `wire_probe`, `pso_nax_attn`.
* Tests: `tests/keepwarm.rs` (7: off by default; both modes touch a new file at once and again while quiet, bytes read back exactly; a busy engine is
  not touched; turning it on after wrapping covers earlier files and 0 stops it; 8 rounds of drop right after wrap with a 5 ms period), and
  `common::tests::keepwarm_is_on_for_serve_only_and_the_env_overrides_both_ways`.

## `serve`: first request after idle (6 paired rounds, fleetq #2178)

`serve-idle.sh`: per sample start `intarsia serve --max-seq 8192 --profile-every 0` (prefill chunk 2048 from the qwen3 recipe, `INTARSIA_METAL_PROFILE=cb:` on in every arm), wait
for `/health`, send one warm-up 4k request, then for idle in 3, 10, 40 s: sleep, send ONE streamed 4k request (4062 prompt tokens plus a nonce line so
no request shares a prefix with an earlier one, 33 tokens, temperature 0). One server per sample, 6 rounds, arm order rotated per round, all in
one slot. Arms: `base` = 6d99a6f (no pre-wire, no keep-warm); `kwb` = this branch with GPU-touch keep-warm; `rs` = this branch with the residency set
(the arm that became the default; this binary still had the pre-wire code, switched off with `INTARSIA_METAL_PREWIRE=0`).
TTFT ms (request sent to first token chunk at the client), median [min..max] of 6:

| idle | base 6d99a6f | blit keep-warm | residency keep-warm | paired minus base: blit | paired minus base: residency |
|---|---|---|---|---|---|
| 3 s | 1164.1 [1159.5..1185.4] | 1056.4 [1054.2..1057.6] | 1060.5 [1058.2..1063.2] | -107.7 [-129.1..-103.4] | -104.9 [-124.8..-96.3] |
| 10 s | 1173.5 [1167.3..1178.1] | 1056.6 [1055.7..1059.6] | 1061.4 [1060.5..1065.4] | -115.0 [-121.6..-111.0] | -110.6 [-116.5..-106.2] |
| 40 s | 1172.5 [1169.7..1181.9] | 1056.8 [1054.9..1060.1] | 1062.7 [1059.5..1069.7] | -115.7 [-127.0..-109.7] | -109.5 [-122.0..-100.1] |

Every one of the 36 paired differences is negative (`idle6-summary.txt`). The generated 33-token texts of the three arms are identical in all 18 prompt groups
(0 differing).

Where the 110 ms went (`spansum-summary.txt`, `cbstart-summary.txt`): the five prefill command buffers of the timed request (chunk 2048, 4062 tokens) have the
same GPU busy time (881..883 ms) and the same idle inside their span (0.4 ms) in all three arms, so the whole gain is **before the first command buffer
starts**. Commit-to-GPU-start of the first command buffer is 7..8 ms (blit) and 10..11 ms (residency) after 3 / 10 / 40 s idle in the arms that carry the
host timestamps (the base build has none; the cbgap doc measured ~102 ms for it). TTFT minus GPU span is ~178 ms for the residency arm against ~288 ms for the
base. The residual against a warm request was not measured in the same server (no back-to-back 0 s arm); the cbgap doc's warm 4062-token chunk-2048
prefill wall was 1041..1057 ms (a different harness), the residency arm's TTFT is 1061 ms. So: the stall is cut by ~110 ms; whether a few ms are left (the residency
arm trails the blit arm by 4..6 ms in TTFT and ~3 ms on the first command buffer) is unresolved.

### Decode ITL after idle

Same streams (33 tokens: mean of the 32 gaps; first-3 = the first three gaps), no profile trace, final binary, fleetq #2178 (4 arms `base`, `off` = new binary
with `INTARSIA_METAL_KEEPWARM=0`, `blit`, `rs` = the shipped default; idles 3 and 40 s). Median [min..max] ms:

| idle | base 6d99a6f | off (new binary, `KEEPWARM=0`) | blit | rs (default) |
|---|---|---|---|---|
| 3 s, mean ITL | 29.3 [28.5..29.8] | 29.1 [29.0..29.7] | 28.7 [28.3..29.4] | 29.1 [28.4..29.7] |
| 3 s, first-3 ITL | 29.0 [28.4..31.0] | 29.4 [28.9..30.1] | 28.8 [28.3..29.6] | 29.2 [28.3..30.6] |
| 40 s, mean ITL | 29.1 [28.4..29.2] | 29.1 [28.2..29.5] | 29.1 [28.4..30.0] | 28.9 [28.3..29.9] |
| 40 s, first-3 ITL | 29.1 [28.3..29.5] | 28.9 [28.1..29.2] | 29.2 [28.4..30.0] | 29.1 [28.3..29.9] |
| 3 s, TTFT | 1170.7 [1162.4..1172.5] | 1179.6 [1162.2..1195.3] | 1058.3 [1056.6..1058.7] | 1061.7 [1060.6..1065.8] |
| 40 s, TTFT | 1178.6 [1171.2..1182.5] | 1178.8 [1170.0..1185.4] | 1058.4 [1057.6..1062.4] | 1061.2 [1059.2..1064.6] |

(n = 4 rounds, no profile trace.) The `off` arm is the control: the new binary with the thread disabled behaves like 6d99a6f (TTFT paired minus base
+10.6 [-9.6..28.9] at 3 s, +2.3 [-7.4..4.9] at 40 s), so the 107..121 ms gain of `blit` / `rs` (paired minus base: blit -112.1 / -120.9, rs -107.0 / -117.7 at
3 / 40 s, every round negative) comes from the keep-warm and not from a different build. Texts identical across the four arms in all 8 prompt groups.

The earlier 6-round table (profile trace on) has ITL 29.2..29.4 ms in all three arms at all three idles, the base's first round (28.2..28.4) being the only
low outlier (`idle6-summary.txt`). Keep-warm does not change decode ITL, and no ITL cost of the stall existed to begin with: the first decode steps
after idle already ran at 29 ms.

## Cost

* **Memory:** no new allocation (the set accounts the 15622 MB the mapped shards already are; `blit` mode allocates a 256 byte scratch buffer), **but the weights
  stay wired while `serve` idles**, up to `recommendedMaxWorkingSetSize` / 2 (the cap; files past it are skipped and logged), and the OS cannot reclaim
  those pages under memory pressure meanwhile. Without keep-warm the driver dropped the wiring after ~1 s. endor showed ~11 GB of swap in use during these runs.
  The copy variants (c) would add 15.6 GB resident and 0.7..1.7 s of load.
* **Load:** the 150..175 ms wiring moves into the keep-warm thread, concurrent with the rest of the load; nothing waits for it. `Drop` joins the thread.
* **CPU:** one wake-up per 400 ms; the refresh is 0.013..0.020 ms. Skipped while the engine commits.
* **GPU / power:** `Residency` mode submits no GPU work. `Blit` mode (fallback) submits one tiny command buffer per 400..500 ms while quiet. GPU power
  (macmon, 60 s segments, `power-summary.txt`) A idle box 27..37 mW mean, B process holding the shards with no touches 61..75 mW, C blit every 500 ms
  46..119 mW, D blit every 1000 ms 68..113 mW: run-to-run spread is larger than the differences, so no cost is resolved at the ~0.1 W level (only
  the fallback was measured; the default does no GPU work).
* **Failure mode:** if the refreshes lapse (the thread starved, or `INTARSIA_METAL_KEEPWARM=<ms>` set to 5000 or more), the first use after the gap re-wires the
  whole set at once and costs 210..230 ms, worse than the ~150 ms of no keep-warm (a1, `probe5/residencycpu-g40000-k5000`). The set held at a 2 s refresh.
* **Coverage gap:** shards over `maxBufferLength` (per-tensor fallback buffers) and anything past half the recommended working set decay as before.
  KV cache and activation buffers are not kept warm; the 7..11 ms first-command-buffer latency left after idle suggests they cost little for Qwen3-8B at
  `--max-seq 8192`, but a much larger KV cache was not measured.

## Gates (final code, fleetq #2178 unless noted)

| gate | result |
|---|---|
| `cargo test -p intarsia-backend-metal --release -- --test-threads=1` | 110 passed, 0 failed (`gates/cargo-test-metal.txt`) |
| `intarsia kernels test` | 27 kernels, 321 cases, all ADMITTED (`gates/kernels-test.txt`) |
| `tests/keepwarm.rs` as a standalone binary | 7 passed (`keepwarm-test.txt` first version, `keepwarm-test2.txt` final) |
| CLI `generate` greedy ids, 48 tokens, L=2049, 2 prompts, chunks 256 and 2048 | new binary, new binary with `INTARSIA_METAL_KEEPWARM=1` (thread running), and 6d99a6f all byte-identical to the 00abb36 binary in stdout and ids (12 of 12 per arm; `gates/gen-*`) |
| `intarsia accuracy` short (`--mode both`) and 4k (`--mode all`) at chunks 256 and 2048, base 6d99a6f vs new | logs byte-identical (4 of 4; `gates/acc-*`) |
| `cargo check --workspace --all-targets` and with `--features intarsia-cli/metal` (rerun on the final HEAD ee93c0a code) | pass |
| `cargo test --release -p intarsia-cli --features metal --lib` | 79 passed |
| CUDA and HIP `cargo check -p intarsia-cli --features {cuda,hip} --tests` on spark1 (fleetq #2187) | both EXIT=0 (`spark1-check.log`; HIP builds the stub, no hipcc, as before) |
| serve output identity | the 33-token texts of all arms agree in every prompt group (18 groups in the 6-round run, 8 in the ITL run) |

## Review fixes (on top of 54df20a)

* `INTARSIA_METAL_KEEPWARM=off|false|no` now means off and `on|true|yes` on (it used to be ignored, leaving serve's default on); other words warn and are ignored;
  numbers clamp to 50..1500 ms with a warning (2 ms = ~500 touches/s, 5000 ms = the 210..230 ms failure mode). Unit test extended (off on serve, words, clamps).
* `residency_loop`: if `newResidencySetWithDescriptor` fails at runtime the thread falls back to `keepwarm_loop` (GPU touches, own queue and scratch)
  instead of exiting while `register` keeps filling the list. Not exercised by a test (the failure cannot be forced here).
* `residency_loop`: `seen` is taken from the length under the second lock, so a `register` between the two locks is added once.
* `Entry`: `whole` is declared before `_mem`, so the no-copy buffer drops before the file unmaps.
* The keep-warm thread runs at QoS `USER_INITIATED`.
* Every whole-file buffer is logged on stderr when registered (address, size) and, loudly, when skipped by the half-working-set cap.
* Memory section corrected above.
* Gates for these fixes: see the final report of the session; GPU gates (cargo test with `keepwarm.rs`, one 3 s serve spot check) are in
  `raw/.../review-fix.sh`, run status stated there.

## Final build at 10 s idle (fleetq #2203)

The final binary (`intarsia-new`, built from a4de0f2 with the review fixes) was run against base 6d99a6f in one exclusive slot, using `IDLES=10 NOPROF=1 serve-idle2.sh <out> 6 base rs`: 6 paired rounds, arm order alternating, and the profile trace off. The `rs` server logs show the five keep-warm registrations (4.00, 3.19, 1.24, 3.99 and 3.96 GB), which confirms the review-fix build ran.

TTFT in ms, median [min..max] over 6 rounds:

| idle | base 6d99a6f | final build (residency keep-warm) | paired: final minus base |
|---|---|---|---|
| 10 s | 1126.2 [1121.6..1139.5] | 1066.5 [1060.7..1078.4] | -58.5 [-78.8..-45.2] |
| warm (back-to-back) | 1128.6 [1123.5..1140.4] | 1076.1 [1064.9..1104.1] | |

The final build was faster in 6 of 6 rounds. The gain is about half of #2178's (-110.6 ms at 10 s). That is because base came in 47 ms faster than in #2178 (1126 ms against 1173.5 ms), while keep-warm came in roughly level (1066.5 ms against 1061.4 ms). This slot differs from #2178 in two ways: the profile trace was off (`NOPROF=1`), and the binary had the review fixes. Neither difference explains why only base moved, and that question is still open. Keep-warm does still win at 10 s on the shipped build, by about 60 ms (5%). Raw results: `raw/2026-10-16-metal-weight-residency/idle10-final/`.

## Fleet jobs

#2169 probe: none/residency/mlock/madvise/copy at 0.3/3/10/40 s. #2170 decay horizon and keep-warm variants. #2171 horizon 1.2..1.8 s, period
500/1000, power A/B (the second repetition of the probe2 rows in `probe3/*-2.txt` was lost to a shell bug and removed). #2176 copyshared / copyprivate /
residency with CPU-only refresh. #2178 one slot: refresh period sweep (`probe5`), keepwarm tests, serve idle A/B (round 1 checked, then 6 rounds), gates, ITL control.
#2187 CUDA/HIP check on spark1. No job of mine is left running or queued.

## Not done / uncertain

* The shipped build was measured at 3 s (#2188) and 10 s (#2203, above) idle, but not at 40 s.
* `commandBuffer.useResidencySet` (per-command-buffer instead of per-queue set) was not tried; the queue-level set already works.
* The residency set is attached to the engine queue. The ITL arms show no effect on decode, but the effect on very short command buffers (single-sequence
  batch-1 decode at high rates) was not isolated beyond that.
* KV / activation buffers not kept warm (see coverage gap). The refresh period (400 ms) was not tuned further: 2 s held and 5 s lost for the set.
* The hardware-unchanged claim "the driver drops the wiring about a second after last use" is measured on this box and this macOS build only.
* `smoke.sh` in the raw dir is the script that ran as job #2178 (it grew; the name is historical).
