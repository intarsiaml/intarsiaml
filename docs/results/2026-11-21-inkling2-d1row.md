# Inkling-2-Mini single-row Metal decode: fused attention and expert fixed latency (lane ink-d1row, 2026-11-21)

Branch `ink/d1row` from `ink/metal` 4d6199a05. Not pushed. Every endor GPU run takes a fleetq slot.

## Briefs received

1. 2026-11-21 (main, initial brief): cut the fixed latency of the fused attention (op 75: 32 us per visit in the stream
   against 22.9 us in the m-attn bench) and of the 1-row expert kernels (89 us per NVFP4 visit in the stream, 83 in the
   bench; target <= 75 in the bench and <= 80 in the stream, same bits); run Fable's measurement B (wo GEMV read twice on
   the same weights, first vs second read; go if second <= 0.6x first, then prototype prefetching the next dense weights
   during latency-bound dispatches). Same bits wherever possible; rows-exact; 1-16 rows keep working. Op contracts
   unchanged without main; new ops take 84-87. Stay out of controller kernels (dfuse) and cb encoding / argmax (hostgap).
   Decode measured with `--profile-every 0`.

2. 2026-11-21 ~22:10 (coordinator, queue tip): endor's quiet line is 14+ jobs (~9 h). Split the bundled job #10823:
   build + bitwise tests + G1-G3 as a default-load job with a short honest `--expect` (load jobs share the box and
   backfill), benches + measurement B + tok/s as one `--quiet gpu,cpu` job; CPU-only work to spark1 / spark2; merge
   ink/metal (rows_exact pin-list fix a0751f302). Done: #10823 released, job L #10850 (`--load gpu,cpu`, 30 min) and
   job Q (`--quiet gpu,cpu`, 35 min) queued; ink/metal merged at 3f3416d40.

3. 2026-11-21 ~22:40 (coordinator, reply to my build-window ask): endor has no CPU-free gap, so check on bespin: one
   `fleetq run --host bespin --load cpu --mem 6 --expect 25` slot; first `uname -s` (stop unless Darwin); rustup in my home
   dir only (`-y --profile minimal --no-modify-path`, toolchain from the repo, no sudo, no rc edits, call
   ~/.cargo/bin/cargo directly); code as a `git archive` of the exact commit to ~/intarsiaml-ink-d1row/src; run
   `cargo check --release -p intarsia-backend-metal --examples --tests` with CARGO_BUILD_JOBS=4; nothing to GitHub; fix
   errors in new commits (no amends) and re-check in the same kind of slot; report hash and result, and note here that a
   rustup toolchain now lives in my home dir on bespin. Also: the row-table form staying un-pipelined is fine.

4. 2026-11-21 ~23:20 (coordinator, reply to my results): opt-in pipelining with base bits as default is right.
   (1) Prefetch (a): **not now** (pays only below ~256 keys, which chat passes in the first turn; ~0.5 ms/token on the
   first few hundred tokens for an op-contract change): park it in this doc, "revisit if the 1-row floor work runs out",
   with the bench numbers and the B ratios. (2) Put the cold-cache finding in a short citable section (attn 15-19 us warm
   vs 22-34 cold, NVFP4 visit 83.4 bench / ~95 stream). (3) Send the single best remaining idea for the 1-row expert visit
   with ms/token and confidence before building it; if nothing beats ~0.3 ms/token at medium confidence, finish the doc,
   make sure 7d04f4ec2's check job passes, and stop.

5. 2026-11-21 ~23:35 (coordinator): agreed, finish the doc, confirm the check job #10933 on 7d04f4ec2 passes, then stop. If
   the single-dispatch expert visit is ever tried, use ink-dfuse's metal4 ticket pattern (tickets taken at threadgroup
   start, atomics in the metal4 lib), not metal3.1 visibility. Bench-pool finding goes in the cold-cache section (passed
   to the expert-tuning lanes). Leave the worktree in place until the opt-in commit is merged.

6. 2026-11-21 ~23:55 (coordinator, change of plan): the pipelined kernels lost, so their code won't merge; only this doc goes to
   ink/metal. Cancel #10933 (done: released, never ran). Put the doc on a docs-only branch `ink/d1row-doc` off ink/metal;
   leave the `ink/d1row` ref as is (it keeps the code). Clean up bespin in a `--load none` slot. Ops 84-87 are released:
   mark them free. Report the docs-branch hash and the bespin removals, then stop; don't delete the worktree.
7. 2026-11-21 (coordinator, correction to 6): keep ~/.rustup and ~/.cargo on bespin (ink-dfuse will use them for Metal
   compile checks); remove only ~/intarsiaml-ink-d1row, ~/rustup-install.log and ~/rustup-tc.log.

## Status

Done; results only. **Not merged code:** the pipelined kernels, their switches, the bench modes and the bitwise tests stay
on branch `ink/d1row` (code tip 7d04f4ec2, where they are opt-in with the base kernels as the default). Only this doc
goes to ink/metal, on the docs-only branch `ink/d1row-doc`, so the switches and bench modes named below do not exist on
ink/metal. The check job #10933 for 7d04f4ec2 was cancelled before it ran (brief 6): the opt-in flip itself is
**unverified** by a build. Its default path is the base code, which job L checked bitwise.

**Discriminants 84-87** (reserved for this lane in the brief) are **released and free**. No in-tree reservation comment
exists to edit: none in `crates/intarsia-core/src/ops.rs`, `inkling2/ops.rs` or the port plan, on any `ink/*` branch.
This line is the record.

**bespin:** the Metal cargo check (4f16e91f4, fleetq #10910) passed. Cleanup (brief 7): removed ~/intarsiaml-ink-d1row,
~/rustup-install.log and ~/rustup-tc.log. Kept ~/.rustup and ~/.cargo (rustup 1.29.3, toolchain 1.90.0) for ink-dfuse.

## What changed (same bits by construction; checked bitwise, below)

At one decode row the fused attention and the expert kernels are chains of dependent memory round trips with the bus
mostly idle (review items 5 and 7). The changes only move loads earlier; every arithmetic operation, its order and every
rounding are those of the existing kernels.

**Fused attention (op 75, `ink_attn_decp_g{1,2,3}`) and RelSinkAttn stage 1 (`ink_attn_s1p_g{1,2,3}`)**, both selected by
`inkattn::pipe()` (default on, `INTARSIA_INK_ATTN_PIPE=0` restores the plain kernels):
- The key loop keeps PD key steps in flight per simdgroup (a register ring refilled PD steps ahead; PD 4, or 8 with
  `INTARSIA_INK_ATTN_PD=8` on group-3 shapes). Before: one K / V load per step, each waited on before the step's math.
- The cache rows of the first PD steps are loaded at kernel entry, before the prologue work (q / r, conv, bias).
- Fused only: the relproj columns of a thread's first two bias elements are loaded at entry; the K / V conv of the call's
  own keys runs before the first barrier with all its loads issued first (`ink_conv4`: K = 4 unrolled, same fma chain);
  its sums go to their own threadgroup array, so 3 barriers precede the key loop instead of 4.

**Decode experts (`ink_moe_gu_*_p`, `ink_moe_dn_f_*_p`, `ink_moe_dn_u_*_p`)**, selected by `inkmoe::pipe_mask()` (default
both; `INTARSIA_INK_MOE_PIPE=0` restores the plain kernels, `gu` / `dn` pipelines gate+up / down alone, because m-moe
measured more loads in flight per lane as slower for gate+up, `INK_NB` 3: 83 -> 91 us, so the two halves are judged apart):
- gate+up: each lane issues its next chunk's gate and up loads before the current chunk's dot products.
- fused down (T = 1): a simdgroup's two slots (k and k + 4) run in one chunk loop, each pipelined one chunk ahead: a lane
  keeps up to four weight chunks in flight instead of one. Each slot keeps its own chain (chunk order, fma order,
  shuffle-xor reduction), so the per-slot values and the k-order token sum are unchanged.
- by-pair down (T >= 2): the same one-chunk-ahead pipeline per row.

Op contracts, op ids and the launch geometry are unchanged; the rows-exact rules hold (a row's work does not depend on T).
ink/batch's row-table form of op 75 (`ink_attn_dec_rows_g*`, its own launcher) is untouched and stays on the plain loop
(merges cleanly; porting the pipelining to it is mechanical). Told main 2026-11-21.

**Measurement B** (`ink_mix_bench reread`) and the **prefetch prototype** (`ink_attn_bench --pf CTX`, kernel
`ink_attn_decp_pf_g3`, benchmarks only: extra threadgroups in the attention launch read the next wo weights; the engine
never selects it).

## Correctness (job L, fleetq #10850, endor, build 4f16e91f4, rustc 1.99.0)

| Check | Result |
|---|---|
| `tests/inkling2_attn.rs` `pipelined_is_bitwise_the_plain_kernels` (new) | **396 cases bitwise** (pair and fused, T 1..16, host and device position, swa ring incl. wrap, full incl. 32 splits, PD 4 and 8) |
| `tests/inkling2_attn.rs` `rows_exact_real_shapes` (existing; pipelined now the default) | 36 cases bitwise |
| `tests/inkmoe.rs` `ink_moe_pipelined_is_bitwise_the_plain_loops` (new) | **NVFP4 500, MXFP8 1500 calls bitwise**, each under all 5 policies x masks gu / dn / both vs plain, T 1..16, aligned and unaligned |
| `tests/inkmoe.rs` rows-exact (existing) | NVFP4 1000, MXFP8 3000 multi-row calls bitwise the one-row calls |
| `tests/rows_exact.rs`, metal lib tests | pass (18 lib tests) |
| admission (`intarsia kernels test --backend metal`) | rel_sink_attn_bf16 63/0, ink_attn_decode_bf16 70/0, moe_experts_situ_bf16 10/0 (+_rows 7/0), moe_experts_mxfp8_bf16 19/0 (+_rows 14/0): all ADMITTED |
| G1 (`INTARSIA_INKLING2_BACKEND=metal`, tests/inkling2.rs) | 8 passed |
| greedy text, pipelined vs plain (`INTARSIA_INK_ATTN_PIPE` / `INTARSIA_INK_MOE_PIPE` 0 vs 1) | short prompt 64 tokens and 5k prompt 32 tokens: identical md5 |
| G2 short / long (oracle goldens, decode + prefill) | PASS: decode KL21 2.589e-3 (<= 4e-3) / 2.474e-3 (<= 6e-3); prefill 2.380e-3 / 3.418e-3 |
| G3 short / long (vLLM goldens, prefill) | PASS: 8.391e-3 (<= 1.05e-2) / 1.248e-2 (<= 1.65e-2) |
| self-goldens vs ink-metal's base run (`2026-11-21-inkling2-d1row/st_cmp.py`) | **every tensor bitwise identical** (G2 short 48, long 60; G3 short 24, long 30; only build metadata differs) |

So the output bits are the base's: no gate was re-recorded and nothing moved.

## Timing (job Q, fleetq #10852, `--quiet gpu,cpu`, build 4f16e91f4; raw in `2026-11-21-inkling2-d1row/`: bench outputs, job logs, per-op and step-period summaries, scripts; the jsonl traces were not kept)

"plain" is the base kernels (identical code to ink/metal), "pipe" the pipelined forms of this lane. Bench numbers are GPU
us per launch in the lanes' harnesses (many launches per command buffer, operands rotating); in-stream numbers come from
a per-dispatch trace of a real decode (`INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1
INTARSIA_METAL_FIRST_COMMIT=0 --profile-every 0`, short prompt, 24 steps, `2026-11-21-inkling2-d1row/cb1_ops.py`, summaries `cb1_ops_pipe{0,1}.txt`). They are one command
buffer per launch, so each in-stream number carries ~3 us of per-buffer cost the bench does not: compare plain vs pipe
within a column, not across.

### Op 75 (InkAttnDecode, fused attention), one row, us per call

| Context (keys) | bench warm plain | bench warm pipe | bench cold plain | bench cold pipe |
|---|---|---|---|---|
| swa 20 | 15.0 | 15.7 | 28.1 | 28.0 |
| swa 48 | 18.7 | 20.0 | 31.0 | 32.5 |
| swa 84 | 18.2 | 19.2 | 24.6 | 24.4 |
| swa 128 | 19.6 | 19.4 | 24.8 | 23.0 |
| swa 256 | 20.3 | 20.2 | 23.2 | 24.5 |
| swa 512 | 22.8 | 22.5 | 31.7 | 29.0 |
| full 20 | 15.1 | 16.0 | 33.5 | 27.4 |
| full 84 | 18.2 | 19.1 | 22.4 | 24.1 |
| full 512 | 22.1 | 22.4 | 26.2 | 27.4 |
| full 4096 (table run) | 62.4 | 81.5 | | |
| **in-stream (decode, < 300 keys)** | **29.9** | **32.0** | | |

PD 8 (8 key steps in flight) is slower everywhere: +1 to +4 us warm at short context, +30 us at full 4k.

**Where the missing ~9 us is.** The m-attn bench rotates 16 layers of operands that stay in the system cache. In the stream,
~200 MB of weights pass between two visits of a layer, so every operand of a visit comes from DRAM. That covers the pq row,
the conv ring and taps, relproj, the q / r scales and the K / V cache. `--cold` puts a 40 MB GEMV on its own weights before
every visit. It lifts the fused kernel at the stream's contexts from 15-19 us to 22-34 us, which brackets the stream's
29.9 us. Context is not the cause: warm, the fused kernel at 20-84 keys is 15-18 us, below the 22.9 us at 512.

Below 64 keys there is one split, so the fused grid is only 8 threadgroups. Cold, that grid is slower than the unfused
pair (swa 20: 28.1 fused vs 23.8 pair; full 20: 33.5 vs 28.0). From 64 keys on, two or more splits, and the fused kernel
wins again (swa 84: 24.6 vs 33.9).

**Why the pipelined form does not help.** Issuing loads early costs registers. The kernels' latency is hidden by having
other simdgroups resident, not by more loads in flight per lane, so the early loads only reduce occupancy: warm +0.3 to
+1.4 us, cold within noise, long context 10-25 % slower.

### Experts, one row (NVFP4 SiTU = 44 visits per token, MXFP8 = 4), us per call

| | bench plain | bench gu | bench dn | bench both | in-stream plain | in-stream both |
|---|---|---|---|---|---|---|
| NVFP4 visit (gate+up + down) | **83.4** (81.7, 82.1 repeats) | 85.2 | 82.8 | 86.4 | **95.3** | 94.2 |
| NVFP4 gate+up (probe) | 50.9 | 51.9 | 49.8 | 52.0 | 61.9 | 59.8 |
| NVFP4 down (probe) | 32.6 | 32.8 | 31.8 | 31.9 | 33.4 | 34.4 |
| MXFP8 visit | **127.4** | 126.6 | 130.1 | 128.6 | **140.7** | 142.4 |
| NVFP4 T = 4 (repeat runs) | 284.6 / 344.7 | 315.2 / 401.5 | 314.1 / 351.3 | 344.0 / 380.8 | | |

gu = pipelined gate+up only, dn = pipelined down only. The target (<= 75 us bench, <= 80 in-stream) is **not met**:
- Pipelining gate+up loses 1-2 us. That matches m-moe's `INK_NB` 3 result: more bytes in flight per lane is slower here.
- The pipelined fused down gains 0.8 us at one row, inside run-to-run noise. The by-pair down loses at 4 rows.
- In the stream, gate+up is ~8 us slower than in the bench, even after the ~3 us per-buffer cost. The down is not. A likely
  reason: in the stream the experts live in the mmapped checkpoint, in the bench in Metal-allocated buffers, so TLB /
  page-walk cost would hit the 16 random expert rows of gate+up first. That is a hypothesis; not measured here.

### Decode step period (engine, `generate --profile-every 0`, cb trace, `2026-11-21-inkling2-d1row/step_period.py`, `step_periods.txt`)

Short prompt (19 tokens), 256 tokens each run. ABBA order, two runs per config; median period over all steps of both:

| Attention | Experts | period us | tok/s |
|---|---|---|---|
| plain | plain | **15525** | **64.41** |
| pipe | pipe | 15773 | 63.40 |
| pipe | plain | 15820 | 63.21 |
| plain | pipe | 15677 | 63.79 |
| pipe | gu | 15751 | 63.49 |
| pipe | dn | 15831 | 63.17 |
| pipe PD 8 | pipe | 16325 | 61.26 |

5k prompt (5051 tokens), 128 tokens each run: plain 16598 / 17442 us (first / last of the ABBA, drifting) = 59.0 tok/s;
pipe 17157 / 17131 us = 58.3 tok/s. Every pipelined configuration is at or below plain, so **the pipelined forms are
opt-in** (7d04f4ec2): `INTARSIA_INK_ATTN_PIPE=1`, `INTARSIA_INK_MOE_PIPE=1|gu|dn`. The default runs the base kernels with
the base bits. The variants stay in the tree for the record. They add 29 pipeline states at backend creation; model load
stayed at 1.7-2.0 s.

### Measurement B (Fable section 3.3): the wo GEMV read twice on the same weights

`ink_mix_bench reread`: 24 wo weight sets (14 MB each, 340 MB) rotate. Each set is launched twice in a row, all in one
command buffer. t1 comes from the plain rotation, t2 = 2 x (paired list) - t1. Median of 3:

| GEMV | first read | second read | ratio |
|---|---|---|---|
| wo [2304, 3072] bf16, 14 MB | 25.70 us (551 GB/s) | 15.16 us (934 GB/s) | **0.590** |
| InkInProj [8664, 2304] bf16, 40 MB | 70.64 us (565 GB/s) | 54.88 us (728 GB/s) | 0.777 |

**wo: go** (0.590 <= 0.6, by a small margin). InkInProj: no-go; 40 MB does not stay in the system cache.

With one command buffer per launch (`INTARSIA_METAL_COMMIT_EVERY=1`), the per-call records give 36.90 vs 32.25 us
(0.874). Each of those carries ~11 us of per-buffer cost and a buffer boundary between the two reads, so it does not
measure the cache effect. The amortized number is the stream's case: one big command buffer per step.

### Prefetch prototype (go from B): read the next wo weights during the latency-bound attention

`ink_attn_bench --pf CTX`, kernel `ink_attn_decp_pf_g3`, benchmarks only. Each visit is [40 MB GEMV, fused attention, wo
GEMV on its own 14 MB weights], like the stream's order. The prefetch arms add z-planes of threadgroups to the attention
launch. Those read the first part of that visit's wo weights and discard the data. Both arms use the pipelined attention.

| | ctx 84 (short decode) | ctx 512 |
|---|---|---|
| no prefetch | 124.7 / 123.7 us per visit | 128.2 / 123.5 |
| prefetch 50 % of wo, 1 plane (16 threadgroups) | **113.2 (-11 us)** | 125.2 (~0) |
| 50 %, 2 planes | 116.9 | 128.0 |
| 50 %, 4 planes | 118.8 | 126.4 |
| 100 %, 4 planes | 122.4 | 132.1 |
| 100 %, 8 planes | 124.9 | 130.3 |

At the short context, a light prefetch of half the wo weights saves ~11 us per visit. That is the whole first-read vs
second-read difference of B. At 48 visits it would be ~0.5 ms per token (~3 %) at short context. At 512 keys there is
nothing to gain: with 8 splits the attention keeps the GPU busier. Heavier prefetch hurts. The bits are untouched (the
prefetch only reads). It does not belong in the engine as is: op 75 does not know the next op's weights. Integrating it
needs either an optional prefetch input on op 75 (an op contract change, main's call) or look-ahead in command-buffer
encoding (hostgap's area).

**Parked (main, brief 4): revisit if the 1-row floor work runs out.** It pays only below ~256 keys, and chat requests pass
that within the first turn. That is ~0.5 ms/token on the first few hundred tokens, for an op-contract change. The
evidence to restart from: B ratios wo 0.590 (go) and InkInProj 0.777 (no-go), and the table above.

### localeval speed-plain (intarsia serve --max-batch 1 --max-seq 20480, endor, job Q)

| Build | 142 tok | 1k | 4k | 16k | record |
|---|---|---|---|---|---|
| default (plain kernels = base bits) | **62.5** | **62.0** | **60.6** | **57.5** | `/opt/code/deepeval/results/20261121-224652-adhoc-endor-intarsia-metal-d1row-pipe0` |
| pipelined forms on (attn + experts) | 61.2 | 60.1 | 58.6 | 55.4 | `/opt/code/deepeval/results/20261121-225054-adhoc-endor-intarsia-metal-d1row-pipe1` |

Decode tok/s, median of 3 reps. Prefill at 4k / 16k is 1303 / 1302 tok/s (default). Both runs exit 3: the speed-plain
1024 cell was rejected on **prefill** spread (decode passed). These are the first localeval decode numbers for Inkling on
Metal. The engine's own 5k-prompt decode (`generate`, 128 tokens) is 59.0 tok/s plain against 58.3 pipelined.

## Cold caches explain the bench-vs-stream gaps (citable)

On one decode row, the gaps between the lanes' microbenchmarks and the decode stream on op 75 and on the experts come from
the cache, not from the context length or the kernel. In the stream, ~200 MB of other weights pass between two visits of a
layer, and every layer has its own expert pool, so a visit's operands come from DRAM. The benches keep part of them in the
system cache.

| | bench, warm (as the lanes measured) | bench, DRAM-cold | in-stream (per-dispatch trace) |
|---|---|---|---|
| op 75 fused attention, 20-512 keys | **15-19 us** (22.9 at 512) | **22-34 us** (40 MB GEMV before each visit) | **29.9 us** (< 300 keys) |
| NVFP4 expert visit (gate+up + down) | **83.4 us** (gate+up 50.9, down 32.6) | gate+up ~61 us if no expert reads hit the cache (469 GB/s x 23.9 MB scaled, below) | **~95 us** (gate+up 61.9, down 33.4) |

- Op 75: warm, the fused kernel at the stream's short contexts (15-18 us at 20-84 keys) is faster than at 512 keys. The
  stream's ~30 us is the cold case. Below 64 keys (one split, 8 threadgroups) the cold fused kernel is slower than the
  unfused pair (28-34 vs 24-28 us). From two splits on, it is faster again.
- Experts: `inkmoe_bench` draws every call's 8 experts from one 128-expert pool (~570 MB), so a few recent calls' experts
  sit in the system cache: about 16 % of expert reads, if the cache holds ~2.7 calls' worth. At no hits the bench's
  gate+up would take ~61 us, which is what the stream shows. Down (latency-bound at ~366 GB/s) moves little. Alignment is
  not the cause: in the real checkpoint the expert w13 / w2 and their scales are all 16-byte aligned, so the stream takes
  the aligned kernels like the bench. For the next lane tuning experts: rotate 24 per-layer pools in the bench.
- In-stream numbers are from one command buffer per launch and carry ~3 us of per-buffer cost each; the conclusions hold
  after subtracting it.

## Next steps (not done here)

- 1-row expert visit as one dispatch (down threadgroups wait on per-pair ready counters set by gate+up): same bits,
  ~4-8 us per NVFP4 visit = 0.18-0.35 ms/token, low-medium confidence. Sent to main; below the bar set in brief 4, so not
  built. **If tried: use ink-dfuse's metal4 ticket pattern** (tickets taken at threadgroup start, atomics in the metal4
  library), not metal3.1 cross-threadgroup visibility (brief 5).
- Down-geometry retune (down at ~390 GB/s in-stream against ~560 for a cold GEMV): changes bits, 0.1-0.2 ms/token.
- Prefetch of the next wo during op 75 (parked, above).
- Remove the opt-in pipelined variants if nobody wants them for the record (they cost 29 pipeline states at startup).
