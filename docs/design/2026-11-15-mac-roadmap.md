# Mac roadmap: where the Metal engine stands on the M5 Max and what to build next (2026-11-15)

Lane `mac-plan`, branch `mac-plan`, base `origin/main` `c5ce87bc8`. Box endor (Apple M5 Max, 40-core GPU, 128 GB, macOS 27.0);
`endor-kev` and `endor-tensorfold` untouched. **No engine code changed**: this is a plan plus the measurements behind it.
Scripts and raw output: [`scripts/mac-plan/`](../../scripts/mac-plan/) (`raw/` holds every log quoted here).
fleetq jobs: #9388 and #9389 (builds, `--load cpu`), #9391 (baseline sweep), #9398 (TTFT probe, short-pass cost),
#9401 (TTFT command-buffer trace), #9407 (per-dispatch decode attribution); every GPU run was `--quiet gpu`.
#9390 was a script bug (empty arm list, exited in a second, no data).

Models: Qwen3.6-35B-A3B NVFP4 (MoE; 40 layers = 30 GDN + 10 full attention; hidden 2048; 256 experts, 8 per token; 16 query /
2 kv heads of 256) and Qwen3.8-27B Quark MXFP4 W4A16 (dense; 64 layers = 48 GDN + 16 full attention; hidden 5120; FFN 17408;
24 query / 4 kv heads of 256). Both run `--spec mtp --spec-k 2` by default on Metal.

## 0. One page

1. **The 27B decode is the only place we lose on the Mac, and it is a row-cost problem.** Plain decode is at 81 % of the
   bandwidth roofline (30.6 tok/s), MTP k2 gives 47 to 50, TensorFold on the same box gives 87 at 4k (1.75x, lane dm,
   2026-11-06). Every extra verify or batch row costs almost a full step: 8 streams give 36 tok/s in total (1.2x of one
   stream). The matrix units (NAX) can do 16 rows for the FLOPs of about half a step, but today they are only reachable
   through a weight-decode pass that costs 250 ms per call (measured, section 1.4).
2. **The 35B loses a third of its speed at long context to attention that reads the KV cache several times.** With the
   default MTP path it decodes 129 tok/s at 4k and 87 at 34k; the KV bytes at 34k are worth 1.2 ms per pass at the
   roofline and the verify pass spends about 10 ms on them.
3. (**Withdrawn**, see the note in section 1.5: a hot box, not the MTP default.) **The MTP default costs 0.3 to 0.45 s of time to first token on the 35B under `serve`** (TTFT at 4k: 1.03 to 1.22 s
   against 0.73 to 0.77 s with `--spec off`). `generate` does not show it. The prefill command buffers themselves run
   longer (97 ms -> 111 to 166 ms), so something runs beside the prefill. Not diagnosed; found today.
4. Single-stream plain decode of the 35B (51 % of roofline) has no clean lever left: fewer dispatches was tried and lost
   (lane mdl), the kernel variants were tried and lost (lane mdec). It is ranked low on purpose.

Top 3 to start tonight (briefs in section 4): **A** row-invariant NAX rows kernel for MXFP4 (27B: verify rows and batch
rows at near-bandwidth cost), **B** KV read once per decode / verify pass (both models, long context), **C** the MTP TTFT
regression. They touch disjoint files.

**One decision the user has to make before A can ever be a default** (it can be built and measured opt-in without it):
A changes the bits of 27B decode. On 2026-11-06 the user ruled "keep bitwise" (every verify row bit-identical to the
M = 1 decode). A keeps a weaker but exact property instead: under its knob every row's bits are independent of the row
count and the row's position, so speculative and batched output is still bitwise the same engine's one-row output. Against
today's default it is numerics-changing and KL / MMLU-gated, like the NAX prefill kernels already are. Section 4, brief A.

## 1. Where time goes today

### 1.1 Roofline inputs, measured today (#9391, `raw/roof/`)

| quantity | measured | used below |
|---|---:|---:|
| copy kernel, read + write bytes / time (1 and 2 GiB) | 549 / 551 GB/s | 550 GB/s |
| best bf16 GEMV, weight + x + y bytes / time (chained, concurrent dispatch) | 588 GB/s | (upper bound) |
| NAX bf16 GEMM peak (4096 x 4096, M = 2048, tile 64x64w2) | 62.0 TFLOPS | 62 TFLOPS |
| NAX bf16 GEMM, long K (4096 x 12288 "down", M = 2048, best tile) | 36.3 TFLOPS | (a gap, idea 7) |

The task brief's 546 GB/s and 64.2 TFLOPS are confirmed within 4 %. ADR-0009's 614 GB/s is not reached by any kernel here;
"% of roofline" below uses 550.

### 1.2 Baseline against roofline (#9391; `serve`, localeval `speed`, filler prompts, 256 forced tokens, 2 reps)

Decode roofline = weight, state and KV bytes read per token / 550 GB/s. Prefill roofline = FLOPs per token / 62 TFLOPS
(GEMMs plus attention at the prompt's mean position; the GDN recurrence and element-wise work are not in it, so it is
optimistic). Bytes per token: 35B 2.4 GB at 4k (lane mdec's count: 1.65 GB labelled weights, 0.57 GB routed experts,
0.13 GB GDN state, 0.08 GB KV), 3.0 GB at 34k; 27B 12.95 GB MXFP4 + 1.29 GB fp8 lm_head (both from today's per-dispatch
trace) + KV 0.26 GB at 4k, 2.2 GB at 34k. FLOPs per prompt token: 35B 4.1 G + attention 0.34 G at 4k / 2.8 G at 34k;
27B 51.8 G + 0.8 G / 6.7 G.

| model, mode | measured | roofline | % of roofline |
|---|---:|---:|---:|
| **35B decode**, plain (`--spec off`), 1 stream, 4k | 117.7 tok/s (8.50 ms) | 229 tok/s (4.36 ms) | 51 % |
| 35B decode, plain, 34k | 91.0 tok/s (10.99 ms) | 183 tok/s (5.45 ms) | 50 % |
| 35B decode, MTP k2 (default), 4k | 129.3 tok/s (filler); 163 to 165 (`generate`, longctx prompt, 2.46 tokens / pass) | | |
| 35B decode, MTP k2, 34k | 87.3 tok/s | | |
| 35B decode, `--max-batch 16`, 1k prompts: 1 / 2 / 4 / 8 / 16 streams, aggregate | 94 / 110 / 164 / 212 / 254 tok/s | 16 rows: 909 tok/s (9.7 GB, lane mprof) | 28 % at 16 |
| **35B prefill**, 4k, `--spec off` | 5781 tok/s (TTFT 0.71 s) | 13.9k tok/s | 42 % |
| 35B prefill, 4k, MTP default | 3551 tok/s (TTFT 1.15 s) | 13.9k | 26 % (section 1.5) |
| 35B prefill, 34k, `--spec off` / default | 2943 / 2762 tok/s | 9.0k | 33 % / 31 % |
| **27B decode**, plain, 1 stream, 4k | 30.6 tok/s (32.7 ms) | 37.9 tok/s (26.4 ms) | 81 % |
| 27B decode, plain, 34k | 24.1 tok/s (41.5 ms) | 33.4 tok/s (29.9 ms) | 72 % |
| 27B decode, MTP k2 (default), 4k / 34k | 47.0 / 35.1 tok/s; `generate` 49.2 to 49.7 (2.44 tokens / pass) | | |
| 27B decode, `--max-batch 8`, 1k prompts: 1 / 2 / 4 / 8 streams, aggregate | 29.6 / 29.2 / 35.9 / 36.2 tok/s | 8 rows share one weight pass: about 290 tok/s | 12 % at 8 |
| **27B prefill**, 4k, `--spec off` / default | 747 / 687 tok/s (`generate`: 746 to 823) | 1179 tok/s | 63 % / 58 % |
| 27B prefill, 34k, `--spec off` / default | 621 / 630 tok/s | 1060 tok/s | 59 % |

Corrections to the task brief's numbers: 27B plain decode is 30.6 tok/s on today's main, not 20 to 22 (the fp8 lm_head and
GEMV work since 2026-11-02); 35B prefill "5.8k at 4k" holds only with `--spec off` (section 1.5).

Single window, 2 reps per cell (1 for the `--spec off` 27B and the batched arms). endor's window-to-window drift is 10 to
20 % (lane m27), so read differences below 10 % between arms as noise unless a paired check is quoted.

### 1.3 Per-dispatch decode attribution (#9407; `INTARSIA_METAL_PROFILE=cb` + `INTARSIA_METAL_COMMIT_EVERY=1`, `--spec off`, 4k prompt, 61 steps; `raw/ce1/table.txt`)

**35B: 8.22 ms GPU busy per token, 764 dispatches.**

| op | launches | us / token | share | us / launch | MB | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `matmul_fp8_gemv` (projections) | 130 | 3435 | 41.8 % | 26.4 | 1280 | 373 |
| `moe_experts_bf16` (gate-up, down) | 80 | 1664 | 20.3 % | 20.8 | (about 566) | (about 340) |
| `matmul_nvfp4_gemv` (shared expert, lm_head) | 121 | 729 | 8.9 % | 6.0 | 321 | 441 |
| `attn_decode_bf16_splitkv` | 10 | 447 | 5.4 % | 44.6 | (84) | (188) |
| `rmsnorm_add_bf16` | 80 | 416 | 5.1 % | 5.2 | 0 | |
| `moe_topk_bf16` | 40 | 328 | 4.0 % | 8.2 | 0 | |
| `gdn_delta_rule_bf16` | 30 | 310 | 3.8 % | 10.3 | | |
| `matmul_pair_bf16_gemv` (router, GDN a / b) | 70 | 306 | 3.7 % | 4.4 | 50 | 163 |
| the rest (conv, combine, norms, silu, gates, argmax) | 203 | 583 | 7.1 % | 2.9 | 0 | |

This reproduces lane mdec's table (2026-10-29) to within 3 %: nothing has moved in the plain 35B step in two weeks. The
step is 130 medium GEMVs of 10 MB each at 373 GB/s plus about 400 launches that move no bytes.

**27B: 31.46 ms GPU busy per token, 949 dispatches.**

| op | launches | us / token | share | us / launch | MB | GB/s |
|---|---:|---:|---:|---:|---:|---:|
| `matmul_mxfp4_gemv` | 496 | 24699 | 78.5 % | 49.8 | 12946 | 524 |
| `matmul_fp8_gemv` (lm_head 248320 x 5120) | 1 | 2219 | 7.1 % | 2183 | 1292 | 582 |
| `rmsnorm_add_bf16` | 128 | 1603 | 5.1 % | 12.5 | 0 | |
| `attn_decode_bf16_splitkv` | 16 | 1378 | 4.4 % | 86.0 | (260) | (189) |
| `gdn_delta_rule_bf16` | 48 | 638 | 2.0 % | 13.3 | | |
| the rest | 260 | 928 | 2.9 % | | 0 | |

The 27B step is one weight pass at 95 % of the copy bandwidth. There is nothing to win inside a one-row step beyond about
15 % (norms at 12.5 us each, attention, a 4-bit lm_head). **Everything above 38 tok/s has to come from more tokens per
weight pass.**

### 1.4 What an extra row costs today

* 27B, one short prefill pass through all 64 layers (#9398, `raw/shortpass/`, `generate`, includes about 300 ms of cold
  start in every cell): prompt of 3 / 7 tokens 334 / 370 ms; 12 / 16 / 24 / 32 / 64 tokens 558 / 552 / 573 / 553 / 575 ms.
  The step from the GEMV path (body of 8 rows or fewer) to the NAX path (`matmul_mxfp4_nax`: weights decoded to bf16
  into the plan scratch on every call, then the NAX GEMM) is **about 190 to 220 ms and flat in the row count**: the pass
  is the decode traffic (12.9 GB read, 52 GB written and read back), not the GEMM. So the matrix units are free for 12 to
  64 rows and unreachable for a verify pass at the same time.
* 27B verify rows on the bitwise ALU kernels (lane dm, 2026-11-06): 3 rows 39.5 ms, 8 rows 89 ms against 33 ms for one.
  fp32 `simdgroup_multiply_accumulate` runs at ALU rate (8.07 TFMA/s), so the bitwise route is closed (phase 3 there).
* 27B batched (today): 8 streams 36.2 tok/s aggregate, i.e. about 220 ms per 8-row step.
* 35B (lane mprof, 2026-10-30, not re-measured): 8 rows 20.7 ms, 16 rows 35.9 ms against a 16.7 ms byte floor; the fp8
  projections run 9..16 rows as two 8-row launches (10.15 ms against a 2.2 ms floor), lm_head 4.17 ms against 0.5.
* Long context multiplies the row cost through attention. From the table in 1.2: 35B plain +2.5 ms per token from 4k to
  34k (KV roofline at 34k: 0.68 GB = 1.2 ms); 35B MTP +3.7 ms per token = about +9 ms per 3-row verify pass. 27B plain
  +8.8 ms (KV roofline 2.2 GB = 4.0 ms); 27B MTP about +17.5 ms per pass. The verify pass pays roughly three times the
  one-row attention cost, so each verify row re-reads the KV cache; on the 35B the head split (`head_split`, 4
  threadgroups per kv head) reads it four more times, softened by the caches.

### 1.5 Found today: the MTP default costs TTFT under `serve` (35B)

> **Corrected by lane mtt (same day, [results](../results/2026-11-15-mtt-metal-mtp-ttft.md)).** This is not a cost of the MTP
> default. The "`--spec off`" arm of #9398 below was started without `--spec off` (an MTP server, and fast), and on a cool,
> unthrottled box the two arms are equal (MTP 754 ms against 746 ms). The slow arms were a hot box: after minutes of GPU load the
> machine throttles the GPU by 1.2 to 1.45x for whichever server runs, `--spec off` included. What the lane did find is a
> start-up cost: admission built every test case to read its label (27B with the MTP default: 65 s to start, now 2 to 3 s).

Same binary, same prompt (`prompt-4096.txt`, `/v1/completions`, fresh nonce per call, 3 calls per arm):

| arm | TTFT ms | job |
|---|---|---|
| default (MTP k2) | 1033 / 1221 / 1188; trace arms 1020 / 1205 / 1124 and 962 / 1074 / 1048 | #9398, #9401 |
| `--spec off` | 771 / 738 / 758; trace arm 768 / 728 / 748 | #9398, #9401 |

localeval agrees: TTFT 1.15 s against 0.71 s at 4k, 12.29 s against 11.51 s at 34k (35B); 5.97 s against 5.47 s at 4k on
the 27B (one window, weaker evidence). The 2026-11-05 flip gate recorded "prefill within noise"; its ABAB windows had this
difference inside a larger window drift.

Command-buffer trace (`raw/ttft/*.jsonl`, summary in `raw/cb-summary.txt`): the seven 128-dispatch prefill buffers of a
4096-row chunk take 96 to 107 ms each with `--spec off`, on every request. With the MTP default the first request's
buffers take 100 / 97 / 96 / 111 / 130 / 151 / 166 ms and the second and third requests' 124 to 154 ms each. These are GPU
start-to-end times, so the GPU is either sharing with other work or starved by a host thread. `generate` with the MTP
default shows 96 to 101 ms per buffer (no penalty; `raw/p35mtp/`), so it is specific to the serve path. The extra is about
0.3 s at 4k and 0.8 s at 34k: not proportional to the prompt. Cause not identified in this lane.

### 1.6 Competitors on the same box, honestly

| model | engine | prefill tok/s | decode tok/s | source |
|---|---|---|---|---|
| 35B | intarsiaml Metal (today) | 5781 at 4k, 2943 at 34k (`--spec off`) | 118 plain, 129 to 165 MTP | this doc |
| 35B | llama.cpp b11429, Q4_K_M (other weights) | 2392 at 4k, 2261 at 16k | 96.7 at 4k | localeval 2026-10-22 |
| 35B | oMLX, mlx-lm | no measurement on endor | | |
| 27B | intarsiaml Metal (today) | 687 to 747 at 4k, 621 to 630 at 34k | 30.6 plain, 47 to 50 MTP k2 | this doc |
| 27B | TensorFold (MLX 4-bit, DFlash2 tree) | 780 at 4k, 710 at 32k | 87.3 at 4k, 63.3 at 16k (same window as our 50.0 / 40.4) | lane dm #8866; localeval 09-28 |
| Qwen3-8B bf16 | intarsiaml / mlx-lm / llama.cpp | 3914 / 3103 / 2495 at 4k | 35.9 / 34 / 35.1 | scoreboard 2026-10-28 |
| DeepSeek V4.1 Flash | Sushi 1.4.1, EXL3, 96 GiB budget | 1051 at 4k | 56 | localeval 2026-10-31 |
| DeepSeek V4.1 Flash | oMLX 0.8.0rc1 expert offload, 12.5 % resident (their run, M5 Max 128 GB) | 28 | 5.6 (2.7 to 4.1 served) | jundot/omlx `docs/experimental/MoE_Expert_Offload.md` |

We lead on the 35B and the 8B and are at parity on 27B prefill. We are 1.75x behind TensorFold on 27B decode. There is no
oMLX number measured on endor for any of these models; its published numbers are its authors', mostly on an M3 Ultra.

## 2. Ideas, ranked by expected user-visible gain / effort

Effort is lane-hours of one Fable lane including gates. "Bitwise" = greedy ids and kernel bits unchanged against today's
default; "class" = bits change against the default but every row is bitwise independent of row count and position, so
speculative and batched output equals the same engine's one-row output; "numerics" = KL / accuracy-gated only.

| # | idea | expected gain | effort | exactness |
|---|---|---|---:|---|
| 1 | MTP TTFT regression under `serve` (brief C) | 35B TTFT at 4k 1.1 s -> 0.75 s (-30 %); 27B -0.5 s | 3 to 6 h | bitwise |
| 2 | KV read once per decode / verify pass (brief B) | 35B MTP decode at 34k 87 -> about 120 tok/s (+35 to +45 %), +15 % at 16k, +5 % at 4k; 27B MTP at 34k 35 -> about 43 (+20 %); plain +12 to +18 % at 34k | 10 to 14 h | bitwise (rule in the brief) |
| 3 | Row-invariant NAX rows kernel for MXFP4 (brief A) | 27B 8 streams 36 -> 150+ tok/s aggregate (4x); verify of 8 / 16 rows at 1.3x / 1.6x a step instead of 2.7x / about 5x; unlocks 4 and 5 | 12 to 16 h | class |
| 4 | DFlash2 tree verify on top of 3 (27B) | single stream 50 -> 60 to 75 tok/s at 4k (TensorFold 87) | 10 to 14 h after 3 | class (inherits 3) |
| 5 | NAX rows for the 35B's fp8 projections and lm_head | 16-row step 35.9 -> about 24 ms: 16 streams +40 to +55 %; MTP k 3..4 becomes worth testing | 12 to 16 h after 3's kernel form is proven | class |
| 6 | In-kernel weight decode for the NAX prefill GEMMs (no bf16 scratch pass) | 27B prefill +8 % solo, about 1.7x beside decoding streams (256-row busy chunks pay 200 ms of decode per chunk); removes the peel hack if 3's kernel serves all M | 6 to 10 h, shares 3's kernel | numerics (K order) |
| 7 | NAX GEMM for long K and small N (27B FFN down at 36 of 62 TFLOPS; 35B shared expert at about 14) | 27B prefill +8 to +12 %, 35B +5 to +8 % | 8 to 12 h | numerics (K order) |
| 8 | Prefix store with MTP on Metal, then on by default for `serve` | agent loops: a 32k prefix restores in about 0.2 s instead of 11 s (35B) or 54 s (27B) of prefill, across conversations and restarts; today a store turns MTP off on Metal | 6 to 10 h | bitwise restore (existing gates) |
| 9 | 35B plain step: fold the byte-free launches into each other | at best +15 % single stream (1.3 ms of 8.5); the earlier fold into the GEMVs lost 3.5 % | 14 to 20 h | bitwise |
| 10 | SSD expert streaming for models larger than RAM (DeepSeek V4.1 Flash on 128 GB) | a model that does not run on Metal today; bar: Sushi 56 tok/s at 96 GiB, oMLX 5.6 at 12.5 % resident | 40 to 60 h, needs a Metal DSv4.1 port and the checkpoint downloaded again | bitwise routing, latency only |
| 11 | KV cache compression (fp8 K / int8 V) | halves KV bytes: +5 to +8 % decode at 34k after 2; memory for batching | 10 to 14 h | numerics; **shelved by the user 2026-10-15** |
| 12 | ANE for the draft model | unknown; a 4 h probe only | 4 h probe | drafts only (exact) |
| 13 | Mac + Spark layer-split prefill | none on today's link | 0 (not now) | numerics (oMLX: KL 1e-3 to 4e-2) |

### 2.1 Notes per idea

Ideas 1 to 3 are the briefs of section 4, where mechanism, files, gates and stop rules are complete. The others:

**4. DFlash2 tree verify (27B).** Mechanism: `crates/intarsia-core/src/model/dflash_tree.rs` (host tree builder,
preorder, acceptance, unit tests; lane dm) exists and is wired into nothing. A tree of R nodes accepts at least what a
chain of R - 1 drafts does for the same verify rows; lane dm measured 3.6 tokens per pass for the k7 chain and TensorFold
keeps 3.2 to 4.4 from a 15-node tree. Why Apple silicon: the 27B step is bandwidth-bound and the matrix units sit idle, so
rows are nearly free once idea 3 lands; without it a 16-row verify costs about 5 steps and the tree loses (lane dm's
finding). Back of envelope with idea 3 at its gate (16 rows = 1.6 steps = 52 ms) and today's 19 ms draft: 4.2 / 71 ms =
59 tok/s; with the draft at 12 ms (65536-row draft head, one flush) and rows at 1.3 steps: 4.2 / 55 ms = 76 tok/s. Files:
`model/dflash2.rs` (verify plumbing over tree rows: position = len0 + depth, attention mask by ancestor), `dflash_tree.rs`,
`model/spec_pipe.rs` for the pipelined form, Metal `attn.rs` / `attention_splitkv.metal` for a tree-masked rows kernel
(coordinate with brief B). Gates: greedy ids == `--spec off` under idea 3's knob on 24 prompts, acceptance trace, `plan
show` identical with `--spec dflash` not passed, ABAB against MTP k2 (rule: >= 1.15x at natural-v1 4k, no cell < 1.0).
Stop: tree keeps < 3.8 tokens per pass at 16 rows, or the ABAB misses twice.

**5. NAX rows for the 35B.** Lane mprof ranked it: fp8 projections 10.15 -> 2.2 ms, lm_head 4.17 -> 0.5, small-N GEMVs
2.6 -> 0.2 at 16 rows. Two forms: brief A's staged-tile kernel with an e4m3 decoder (fp8.metal `fp8_rows_mmt` is the ALU
twin), or a persistent bf16 twin of the 1.28 GB of fp8 projections (+2.6 GB of RAM) fed to the existing `gemm_nax_fp8`
(55 to 62 TFLOPS, no new kernel) for 9..16 rows. Exactness is harder than on the 27B: a one-row step on the twin reads 2x
the bytes, so solo decode would stay on the GEMV and batched rows would stop being bitwise the solo step (KL-gated, a
policy change for `--max-batch` servers). Files: `kernels/metal/fp8.metal`, `nvfp4.metal`, `src/fp8.rs`, `src/nvfp4.rs`,
`lib.rs` `row_chunks`. Gates: `rows_exact.rs` within the class, `batch_metal_real`, KL vs the CPU oracle, served 16-stream
ABAB (rule: >= 1.3x aggregate). Stop: 16-row harness step above 30 ms.

**6. In-kernel weight decode for prefill.** `matmul_mxfp4_nax` / `matmul_nvfp4_nax` / `matmul_fp8_nax` decode the whole
weight to bf16 scratch per call (three dispatches). Measured cost on the 27B: about 200 ms per pass whatever the row count
(section 1.4), i.e. 8 % of a 2048-row chunk and most of a 256-row busy chunk. Brief A's kernel decodes per tile inside the
GEMM; if it holds its rate at M = 2048 it replaces the scratch path, and then prefill, decode and verify share one numerics
class, which makes `prefill_peel` (the 27B and 35B tail hack of 2026-11-04) unnecessary under the knob. Gates: prefill KL
vs golden, MMLU, chunk-invariance ids (chunk 512 == 2048), solo == behind-a-decoder ids at the exposed lengths. Stop:
slower than the scratch path at M >= 256.

**7. NAX GEMM for long K and small N.** The microbench shows 36 TFLOPS on the K = 12288 "down" shape against 62 on square
shapes at the same M; the 27B's down projection is K = 17408 and a quarter of its FLOPs. Likely cause: the K loop streams
A rows that no longer fit the tile's cache footprint; a K-split (two accumulation passes over K halves into the same
float destination) or a swizzled grid keeps both operands resident. The 35B shared expert (512 x 2048 NVFP4) runs at about
14 TFLOPS effective in prefill (10 % of prefill time for 5 % of the FLOPs). Files: `kernels/metal/gemm_nax_fp8.metal`,
`gemm_nax_bf16.metal`, `src/gemm_nax.rs` (`pick_variant`), `examples/gemm_nax.rs`. Gates: kernels test, prefill KL, MMLU,
served prefill ABAB. Stop: the microbench does not reach 48 TFLOPS on the down shape after two tile forms.

**8. Prefix store with MTP on Metal.** `crates/intarsia-core/src/prefix_store.rs` already persists KV pages and the
recurrent blob to disk, content-addressed (`serve --prefix-store`, off by default). On Metal a configured store turns MTP
drafting off (`load.rs` near line 710; `INTARSIA_MTP_STORE=1` forces it; lane mtp-store, 2026-11-13, did the CUDA side).
Why Apple silicon: unified memory makes a restore a page-cache read into buffers the GPU already maps, and the internal
SSD reads several GB/s; a 32k prefix is 0.7 GB (35B) or 2.2 GB plus state (27B). Work: run lane mtp-store's gates on
Metal, make the MTP head's KV part of the entry or rebuild it by a head catch-up at restore, then propose default-on with
a disk budget. Gates: the store's reuse gates (`accuracy --mode reuse`), ids warm == cold with MTP, restart test. Stop:
the head catch-up at restore costs more than 20 % of the saved prefill.

**9. 35B byte-free launch folding.** Section 1.3: about 400 launches move no bytes and take 1.3 to 1.7 ms. Lane mdl
(2026-10-22) cut dispatches from 764 to 455 by folding norms into the GEMVs and lost 3.5 %, because every threadgroup
repeated the norm. The untried form folds small ops into each other in one single-threadgroup kernel with barriers
(residual add + rmsnorm + q(x) prep; top-k + combine weights; conv + delta rule + gated norm). Also untried: one launch for
the pair of GEMVs that share an input and a concurrent dispatch encoder for independent neighbours (lane phase-b2b's list).
Honest expectation +5 to +15 %; two related attempts already failed. Stop: the first fold does not gain 1 % in ABBA.

**10. SSD expert streaming.** oMLX's design is sound and simple: a per-layer slot cache of resident experts, misses read
with `pread` from the checkpoint's own safetensors (they measured 0.23 -> 9.3 GB/s moving from an mmap gather to `pread`),
eviction by decayed routing count, prefill chunked on expert boundaries. Its result on our hardware class is 5.6 tok/s at
12.5 % resident, so the feature is "it runs", not "it is fast"; Sushi's 56 tok/s at a 96 GiB budget is the bar for a model
that mostly fits. For us it needs a Metal port of the DSv4.1 engine first (CUDA / TP only today; the checkpoints were
deleted 2026-11-12) and endor's 60 GB job budget next to the standing services is below the 82 GiB EXL3 file, so streaming
would be mandatory, not optional. A cheap first step that needs no new model: cap the 35B's resident experts artificially
and measure miss latency with `pread` into a Metal buffer (2 h probe). Ranked low on effort and on the missing checkpoint,
not on value.

**11. KV compression.** `--kv-dtype fp8|int8` exists on branch `metal-kv-quant` and was shelved by the user on 2026-10-15
("revisit with K-bf16 / V-int8 if KV memory ever limits batching"). After brief B the KV read is at the roofline and its
bytes become the long-context cost, so this is the follow-up, with the user's decision reopened first.

**12. ANE.** The ANE takes Core ML graphs with static shapes and fp16; it cannot read our weight buffers in place and it
shares the same memory bandwidth, so it cannot speed up a bandwidth-bound weight pass. The one fit is the drafter (DFlash2
block: 5 layers at width 5120, 8.7 ms on the GPU; MTP head): drafts only propose, so any numerics are exact for the
output, and the ANE's compute does not take GPU ALUs away from the verify pass. Probe: convert the DFlash2 block with
coremltools, time it on the ANE alone and beside a GPU decode loop. Stop: slower than 8 ms per block, or decode slows by
more than the draft saves.

**13. Mac + Spark split prefill.** oMLX sends 45 to 109 KB per token (hidden state plus the helper layers' KV rows). At
endor's 40 to 45 MB/s Wi-Fi that caps the helper at 400 to 900 tok/s. The 35B prefills locally at 2.9k to 5.8k tok/s and
the 27B at 620 to 750, so the link alone is at or below local speed before the Spark does any work. oMLX's own gains need
Thunderbolt or 2.5 GbE (their peak 308 MB/s) and are not bit-exact (top token changed on 2 of 8 prompts for two models).
Revisit only with a wired link of 2.5 GbE or better; then the 27B (Radiance prefills it at 6.1k tok/s on tatooine) is the
candidate.

**Evaluated and not listed.** *Fused decode megakernel / persistent kernel:* Metal has no cross-threadgroup
synchronisation, and one threadgroup cannot use the machine's bandwidth, so a megakernel can only cover the byte-free ops
(that is idea 9). *Command-buffer and encoder overhead:* GPU idle is 0.26 ms of 8.5 (35B) and 0.8 of 33 (27B) per token,
0.31 ms per MTP pass since lane mmtp; not a lever. *Weight formats:* the 35B's fp8 projections are 53 % of its bytes, a
4-bit form would cut the floor from 4.4 to 3.2 ms, but it needs a calibrated requantisation and the step is not at its
floor anyway; the 27B's fp8 lm_head (7 % of the step) as NVFP4 would give +3 %. *Unified-memory zero-copy:* weights are
already mmap'd and wrapped with `newBufferWithBytesNoCopy`, with a residency set under `serve`; the remaining item is a
cold first request in `generate` (130 to 170 ms of GPU idle before the first buffer), which `serve` does not have.

## 3. Why these three first

* They cover the three measured gaps: row cost (A), long context (B), TTFT (C).
* B and C are bitwise and can ship default-on after gates; A is opt-in and produces the evidence for the user's decision.
* They do not share files beyond two append-only lists (table below).

| lane | files it edits | GPU time on endor (fleetq `--quiet gpu`) |
|---|---|---|
| A `mnr` | new `kernels/metal/mxfp4_naxr.metal`; `crates/intarsia-backend-metal/src/mxfp4.rs`, `build.rs` (`METAL4_SOURCES`, one line), `kernels.rs` (`TABLE`, `all()`, `MAIN_KERNEL_LIST`: appended entries); new `tests/mxfp4_naxr.rs`, `examples/mxfp4_bw.rs` | about 15 microbench slots of 3 to 5 min, then about 90 min of gates and ABAB |
| B `mka` | `kernels/metal/attention_splitkv.metal`, `crates/intarsia-backend-metal/src/attn.rs`, `rowtab.rs`, `kernels.rs` (appended entries), `tests/` attention tests, `examples/attn_bw.rs` | about 10 microbench slots of 3 to 5 min, then about 60 min of gates and ABAB |
| C `mtt` | `crates/intarsia-core/src/model/mtp.rs`, `model/spec_pipe.rs`, `engine.rs`, `engine/batch.rs`, possibly `crates/intarsia-server` request path | about 6 slots of 5 to 10 min |

A and B both append to `kernels.rs` and its pinned list (a two-line merge conflict, resolve by keeping both). All slots
are short; the lanes queue behind each other in fleetq and none needs more than 32 GB.

## 4. Implementer briefs

Each brief is complete on its own. Common rules are repeated in each.

---

### Brief A: lane `mnr`, row-invariant NAX rows kernel for MXFP4 (Qwen3.8-27B on Metal)

**Goal.** Make M = 2..16 rows of an MXFP4 matmul cost close to one bandwidth pass on the M5 Max by running them on the
matrix units (NAX, `mpp::tensor_ops::matmul2d`) with the 4-bit weights decoded per tile inside the kernel. Ship it opt-in
behind `INTARSIA_METAL_NAX_ROWS=1`. With the knob unset nothing changes (`plan show` byte-identical).

**Why.** On endor the 27B's one-row step is a weight pass at 95 % of copy bandwidth (31.5 ms GPU, 30.6 tok/s). Extra rows
on the bitwise ALU kernels cost almost a step each: verify 3 rows 39.5 ms, 8 rows 89 ms (lane dm,
`docs/results/2026-11-06-metal-dflash2.md`); 8 batched streams give 36 tok/s in total. The existing NAX path
(`matmul_mxfp4_nax`, M > 8) decodes the whole weight to bf16 scratch on every call, about 200 ms per pass through the
model whatever the row count (`docs/design/2026-11-15-mac-roadmap.md` section 1.4). The FLOPs of 16 rows are 0.83 TFLOP
per pass = 13 ms at the NAX peak of 62 TFLOPS, against a 26 ms weight read. TensorFold reaches 87 tok/s on this box with
a 15-node tree, so the hardware can do it.

**Exactness contract (read this first).** The new kernel's bits differ from `matmul_mxfp4_gemv`'s (the matrix units
accumulate K in another order). The user ruled "keep bitwise" on 2026-11-06, so this stays **opt-in** and you do not flip
any default. What must hold under the knob:

1. Row invariance: a row's output bits do not depend on M, on the row's index, on the other rows' contents, on the tile
   choice or on buffer offsets. (The existing `matmul_mxfp4_nax` test 2 in `tests/mxfp4_nax.rs` proves the same for the
   scratch form; reuse its structure.)
2. Under the knob the same kernel serves M = 1 (decode step), M = 2..16 with `exact_rows` (verify and batch rows) and
   prefill passes of 8 rows or fewer, so speculative and batched output is bitwise the knob's own one-row output.
3. Against the default it is KL / MMLU-gated (gates below).

**Design.**

* Kernel `mxfp4_naxr_{bf16,f32}` in a new `kernels/metal/mxfp4_naxr.metal` (Metal 4, added to `METAL4_SOURCES` in
  `crates/intarsia-backend-metal/build.rs`). Start from `kernels/metal/moe_nax.metal` (`moe_mm_nax_impl`): it already
  decodes 4-bit blocks to bf16 once per tile into threadgroup memory and runs `matmul2d` in multiply-accumulate mode into
  a float cooperative tensor per simdgroup, and it runs the 35B's grouped experts at about 39 TFLOPS effective. Dense
  form: one threadgroup per 64 (or 32) weight rows; activation tile TM = 16 rows (rows >= M are zero and never stored;
  for M <= 8 try an 8-row tile padded to 16 as `moe_nax` does); K walked in steps of BK = 256 or 512; per step the lanes
  decode their 32-weight groups (16 weight bytes + 1 scale byte) to bf16 with the arithmetic decode of `mxfp4.metal`
  (`mx_dec8` / the e2m1 * 2^e form of `mxfp4_w_to_bf16`, exact in bf16) into the staged B tile; the A tile (activations,
  M x BK bf16) is staged once per step from device memory (it is tiny and cache-resident: 16 x 5120 x 2 B). Epilogue:
  bf16 round-to-nearest-even or f32 store, as `gemm_nax_fp8.metal`'s `epilogue`.
* W4A16 only (`act` 0); W4A4 calls keep the GEMV (the registration predicate excludes them, as `nax_ok` does).
* Second form if the staged tile misses the kernel rule: register-fed operands
  (`get_right_input_cooperative_tensor`, filled from decoded codes; `kernels/metal/attention_flash_nax.metal` `nax_mma`
  shows the fragment layout and measured 12.6 TFLOPS with 8-byte lane loads; 4-bit codes give four times the elements per
  loaded byte).
* Registration (`crates/intarsia-backend-metal/src/kernels.rs`, predicates in `src/mxfp4.rs`; read
  `crates/intarsia-core/src/registry.rs` lines 354 to 365 for how `rows_exact` selection works): with the knob set,
  `matmul_mxfp4_naxr_rows` (`.rows_exact()`, M 2..=16) and `matmul_mxfp4_naxr` (not `rows_exact`, M 1..=8, priority above
  the GEMV's). With the knob unset both predicates return false, so selection and `plan show` are unchanged. Append the
  names to `TABLE` and `MAIN_KERNEL_LIST` (test `kernel_list_is_unchanged_by_core_op_fusion_p1`). No change in
  `intarsia-core`.
* Stretch, only after every gate below passes: let the kernel take M > 8 under the knob (replacing the scratch path) and
  measure prefill; if it is within 3 % or faster, prefill, decode and verify share one numerics class and
  `prefill_peel` is unnecessary under the knob. Report it; do not change the recipe.

**Files.** New: `kernels/metal/mxfp4_naxr.metal`, `crates/intarsia-backend-metal/tests/mxfp4_naxr.rs`. Edited:
`crates/intarsia-backend-metal/src/mxfp4.rs`, `src/kernels.rs`, `build.rs`, `examples/mxfp4_bw.rs` (the microbench; add
the new kernel as a candidate). Lane B (`mka`) also appends to `kernels.rs` and `MAIN_KERNEL_LIST`; expect a two-line
merge conflict and keep both. Do not edit `attention_splitkv.metal`, `attn.rs`, `rowtab.rs` (lane B) or
`crates/intarsia-core/src/model/mtp.rs`, `spec_pipe.rs`, `engine.rs` (lane C).

**Pre-registered rules (write them into the results doc before measuring).**

* Kernel (shapes 17408 x 5120, 5120 x 17408, 12288 x 5120, 5120 x 6144; `examples/mxfp4_bw.rs`; GPU timestamps):
  M = 16 <= 1.6x and M = 4 <= 1.25x the M = 1 `matmul_mxfp4_gemv` time on every shape (today's bitwise rows: M = 8 is
  2.7 to 4.1x). M = 1 on the new kernel: report it (it is the knob's one-row step; expected about 1.2 to 1.4x the GEMV).
* End to end, same binary, knob off against on, served, ABBA, localeval `speed` (the `local-model-bench` skill):
  (a) `--max-batch 8`, 8 streams at 1k: aggregate decode >= 3x (36 tok/s today); (b) default MTP k2, 1 stream, natural-v1
  1k and 4k: no cell below 0.97; (c) `--spec dflash --spec-k 7` with the knob against MTP k2 without it at natural-v1 4k:
  report (lane dm measured 0.54x on the bitwise rows; above 1.0 makes idea 4, the tree, the next lane).

**Gates.**

1. `tests/mxfp4_naxr.rs`: row invariance (M 1..16, every row index, random co-rows, tile forms, 2- and 4-byte offsets,
   bf16 and f32 out, K 32 .. 17408, N with tails 37 / 130 / 515); against an f64 reference (bound: the
   `matmul_mxfp4_nax` case bounds, max_abs 2e-3 / rel_l2 5e-5 class); against the GEMV relative RMS < 1e-5; registration
   (knob unset: never selected; set: M and act ranges); a planted fault (one row's tile zeroed) is caught.
2. `intarsia kernels test --backend metal`: every existing row identical to main, new rows ADMITTED, 0 FAIL.
3. `intarsia plan show --model <M> --backend metal` byte-identical to main with the knob unset for
   `~/models/Qwen3-8B`, `~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`, `~/models/Qwen3.6-35B-A3B-NVFP4`, each with default
   flags, `--spec off` and `--spec mtp`.
4. Ids under the knob, 256 greedy tokens: MTP k2 == MTP k4 == `--spec off`; `--spec dflash --spec-k 7`
   (draft `z-lab/Qwen3.8-27B-DFlash2`, HF cache snapshot `50307d4c`) == `--spec off`; served `--max-batch 8` with 2, 4, 8
   concurrent identical prompts == solo; on a short prompt, 1k, 4k (two prompts) and 16k. Repeat runs equal.
5. Accuracy under the knob, the method of `docs/results/2026-11-02-m27-metal.md` section "Task 3, accuracy" (HF W4A16
   golden regenerated on endor CPU, `tools/golden_quark_mxfp4.py`): prefill and decode KL21 and top-1 reported beside
   the default's (5.9e-4 / 9.6e-4); pass = decode KL21 <= 1.5e-3 (HIP's level) and top-1 within 0.3 points. MMLU
   n = 1425: within 0.010 of 0.8161.
6. `cargo test --workspace --release --locked` (the endor-only
   `completions_logprobs_do_not_stall_the_async_threads` may be skipped, say so).
7. The speed rules above.

**Stop rules.** (1) If the staged-tile kernel misses the kernel rule, try the register-fed form once; if M = 16 is still
above 2.2x M = 1 on the two FFN shapes, stop, keep the code on the branch, write up the measured TFLOPS per form and what
limits it. (2) If row invariance cannot be made to hold on some shape, exclude that shape in the predicate and say so; if
it fails on the FFN shapes, stop. (3) If gate 5 fails, stop and report the KL table; do not tune tolerances.

**Deliverables.** Branch `lane-mnr/metal-nax-rows` from `origin/main`, worktree
`/opt/code/intarsiaml/.worktrees/mnr` (yours alone; build there, delete `target/` when done). Results doc
`docs/results/2026-11-15-mnr-metal-nax-rows.md` with raw logs under `docs/results/2026-11-15-mnr-metal-nax-rows/raw/`
and scripts under `scripts/mnr/`. A row in `docs/decisions/LOG.md` (status opt-in, the user's default decision open).
PR against main; report the PR number, the rule table and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mnr: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

---

### Brief B: lane `mka`, read the KV cache once per decode / verify pass (Metal, both hybrid models)

**Goal.** Make decode attention on Metal cost about one read of the visible K / V rows per layer and pass, for one row
and for 2..16 verify / batch rows, without changing any output bit. Ship opt-in behind `INTARSIA_METAL_ATTN_KV_ONCE=1`
first; default-on is the coordinator's call after the gates.

**Why.** Measured on endor today (`docs/design/2026-11-15-mac-roadmap.md` sections 1.2 to 1.4; served, localeval, filler
prompts):

| | 4k | 34k | extra per pass at 34k | KV bytes at 34k / 550 GB/s |
|---|---:|---:|---:|---:|
| 35B plain | 117.7 tok/s | 91.0 | +2.5 ms | 0.68 GB = 1.2 ms |
| 35B MTP k2 (default) | 129.3 | 87.3 | about +9 ms (3 rows) | 1.2 ms |
| 27B plain | 30.6 | 24.1 | +8.8 ms | 2.2 GB = 4.0 ms |
| 27B MTP k2 (default) | 47.0 | 35.1 | about +17.5 ms (3 rows) | 4.0 ms |

The 3-row verify pass pays about three times the one-row attention cost: each row re-reads the cache. On the 35B
(G = 8 query heads per kv head, head_dim 256) the one-row kernel also splits the query heads over `head_split` = 4
threadgroups per kv head, each reading the same K / V slice (`crates/intarsia-backend-metal/src/attn.rs` `head_split`;
`kernels/metal/attention_splitkv.metal` header, "Head split"). Lane mprof measured the batched kernel at 36 % of the KV
read roofline. Expected: 35B MTP at 34k 87 -> about 120 tok/s, 27B MTP at 34k 35 -> about 43, plain +12 to +18 % at 34k,
about +5 % at 4k, and the same saving on every batched step.

**What exists.** `kernels/metal/attention_splitkv.metal`: stage 1 `attn_decode_bf16_splitkv_g{1,2,4,8}` (one
threadgroup per (kv head, split); a lane loads one 16-byte chunk of a K / V row; KPS = 256 / D keys per simdgroup step,
i.e. one key at D = 256; slots merged with `simd_shuffle_xor`, simdgroups through threadgroup memory), stage 2
`attn_decode_bf16_combine` (partials combined in split order). `attn_rows_splitkv_g*` + `attn_rows_combine` serve the
`rows_exact` multi-row path, `attn_decode_rowtab_g*` the batched row-table path (`src/rowtab.rs`). Split policy:
`decode_splits` / `ask_nsplit` (`MIN_KEYS` 128, `TARGET_TGS` 120, `MAX_SPLIT` 64, chunk rounded up to 32 keys);
`INTARSIA_METAL_SPLITKV`, `INTARSIA_METAL_ATTN_HS`, `INTARSIA_METAL_ATTN_HSB` are diagnostic overrides.

**Design.** First measure, then build; the mechanism is yours to choose inside the bitwise rule.

1. Attribution (first slot): `examples/attn_bw.rs` at the two real shapes (35B: N 16, Nkv 2, D 256; 27B: N 24, Nkv 4,
   D 256), visible 4k / 16k / 34k, M = 1 / 3 / 8: time and effective GB/s against unique K + V bytes; split it into
   re-reads (rows x head split) and per-read rate (load-issue-bound at 16 bytes per lane and one key per step?). Write
   the table into the results doc before changing a kernel.
2. Share the read across rows. Verify rows of one sequence see keys 0 ..= pos0 + i, so they share every K / V tile; one
   threadgroup per (kv head, split) should load each tile once and apply it to all rows' query heads. Register pressure
   is the known limit (GP = 8 at D = 256 already spills, hence the head split), so stage the K / V tile in threadgroup
   memory (16 keys x 256 x 2 B = 8 KB per operand) and let R x G / GP simdgroup sets read it from there, or loop the
   rows over a tile held in registers. Rows whose split count or chunk differs from the first row's (it depends on the
   row's visible length) fall back to today's launches for that pass.
3. Remove the 35B's four-fold head-split read the same way (the four head groups of a kv head share one staged tile).
4. Only if 1 shows the single read itself is far below 550 GB/s: wider loads per step for D = 256.

**Bitwise rule.** Each (row, query head) keeps today's arithmetic exactly: the same keys per slot and simdgroup, the same
online-softmax stream per slot, the same `simd_shuffle_xor` and threadgroup merge order, the same split boundaries and
combine order. Only who loads which bytes, and how many rows a threadgroup serves, may change. The header already states
"each query head's arithmetic is independent of the others', so HS changes no bit at the same simdgroup count": keep the
simdgroup count per head group. If a faster form needs another reduction order, stop and report it as a
numerics-changing option with its measured gain; do not ship it in this lane.

**Files.** `kernels/metal/attention_splitkv.metal`, `crates/intarsia-backend-metal/src/attn.rs`, `src/rowtab.rs`,
`src/kernels.rs` (appended entries; lane A appends there too: two-line conflict, keep both), `examples/attn_bw.rs`,
attention tests under `crates/intarsia-backend-metal/tests/`. Do not edit `mxfp4.*` (lane A) or
`crates/intarsia-core/src/model/mtp.rs`, `spec_pipe.rs`, `engine.rs` (lane C). `kernels/metal/dflash.metal`'s windowed
decode (`attn_decode_bf16_window`) reuses the decode shaders over a window: keep it bitwise too (its test is
`tests/dflash2.rs`).

**Pre-registered rules.**

* Kernel: at visible 34k, M = 3 costs <= 1.3x M = 1 (today about 3x), and M = 1 reaches >= 70 % of unique K + V bytes /
  550 GB/s on both shapes (today about 40 %).
* End to end, same binary, knob off against on, served, ABBA, localeval `speed --sizes 4096,16384,32768 --gen 256`:
  35B default (MTP k2) decode at 34k >= 1.25x and at 4k >= 0.98x; 27B default at 34k >= 1.12x; `--spec off` no cell
  below 0.98.

**Gates.**

1. Bitwise kernel tests: every row of the new multi-row launch equals the one-row launch (M 2..16, visible lengths
   around every split boundary of `decode_splits` between 1 and 40000, both shapes plus the 8B's G = 4, D = 128, cache
   offsets); the new one-row launch equals today's one-row launch (same list); a planted fault (one tile skipped) is
   caught. Existing `rows_exact.rs`, `tests/dflash2.rs` and the row-table tests pass unchanged.
2. `intarsia kernels test --backend metal`: existing rows identical, 0 FAIL.
3. `intarsia plan show --backend metal` byte-identical to main with the knob unset: 8B, 27B, 35B x default /
   `--spec off` / `--spec mtp`.
4. Greedy ids, 256 tokens, knob on == knob off == main: 35B and 27B, default and `--spec off`, short / 4k (two prompts)
   / 16k / 32k (`docs/results/raw/2026-10-15-metal-longctx-profile/prompts/`); served `--max-batch 8` on the 35B with 4
   and 8 streams == solo; Qwen3-8B short and 4k.
5. `crates/intarsia-models/tests/batch_metal_real.rs` (35B) passes.
6. `cargo test --workspace --release --locked`.
7. The speed rules.

**Stop rules.** (1) If the attribution shows re-reads are not the cost (M = 3 already <= 1.5x M = 1 in the microbench),
stop and report where the +9 ms per verify pass at 34k goes (then it is the engine's verify path, not the kernel).
(2) If the shared-tile kernel is bitwise but slower than today at 4k on either model and cannot be gated by visible
length cleanly, ship it for visible >= the crossover only and say so. (3) Two kernel forms miss the kernel rule: stop,
write up.

**Deliverables.** Branch `lane-mka/metal-kv-once` from `origin/main`, worktree `/opt/code/intarsiaml/.worktrees/mka`
(yours alone; delete `target/` when done). Results doc `docs/results/2026-11-15-mka-metal-kv-once.md`, raw logs under
`docs/results/2026-11-15-mka-metal-kv-once/raw/`, scripts under `scripts/mka/`. A row in `docs/decisions/LOG.md`. PR
against main; report the PR number, the rule table and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mka: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

---

### Brief C: lane `mtt`, the MTP default's time-to-first-token cost under `serve` (Metal)

**Goal.** Find and remove the extra 0.3 to 0.45 s of TTFT that the MTP default adds to a 4k prompt on the 35B under
`intarsia serve` on Metal, without changing any output token. Then check the 27B and the first request after load.

**Evidence (all on main `c5ce87bc8`, endor, 2026-11-15; scripts in `scripts/mac-plan/`, logs in
`scripts/mac-plan/raw/`).**

* `scripts/mac-plan/ttft.sh` (one `serve` per arm, `/v1/completions`, `prompt-4096.txt` with a fresh nonce, 3 calls):
  default (MTP k2) TTFT 1033 / 1221 / 1188 ms, `--spec off` 771 / 738 / 758 ms (fleetq #9398; repeated in #9401:
  962 / 1074 / 1048 against 768 / 728 / 748).
* localeval `speed` (#9391): 35B TTFT 1.15 s against 0.71 s at 4k (prefill 3551 against 5781 tok/s), 12.29 against
  11.51 s at 34k; 27B 5.97 against 5.47 s at 4k (one window).
* Command-buffer trace (`INTARSIA_METAL_PROFILE=cb:<file>`, `raw/ttft/mtpcb.jsonl`, `offcb.jsonl`; print with
  `scripts/mac-plan/cbgap.py <file> 8`): a 4096-row prefill chunk is seven 128-dispatch buffers. `--spec off`: 96 to
  107 ms each on every request. MTP default: first request 100 / 97 / 96 / 111 / 130 / 151 / 166 ms; second and third
  requests 124 to 154 ms each. These are GPU start-to-end times.
* `intarsia generate` with the MTP default: 96 to 101 ms per buffer, prefill 907 ms against 881 ms off
  (`raw/p35mtp/`, `raw/p35/`). So the plain prefill path is fine and the cost is specific to the serve path with MTP on.
* The extra is about 0.3 s at 4k and 0.8 s at 34k: not proportional to the prompt length.

**Hypotheses, in the order to test them** (none verified):

1. GPU work submitted beside the prefill: the MTP head's catch-up over queued prompt rows
   (`crates/intarsia-core/src/model/mtp.rs`: `mtp_queue`, `mtp_catch_up`, `mtp_catch_up_rank`), run in small pieces whose
   command buffers overlap the prefill's, or the pipelined draft of `model/spec_pipe.rs` (`pipe_draft`, "drafted ahead")
   still running from the previous request.
2. A host thread that takes memory bandwidth during prefill under MTP (a readback, a zeroing or copy of the head's KV or
   workspace sized by `--max-seq`, the hidden-row queue). Check with `sample <pid> 1 -file ...` during a prefill and by
   repeating the probe at `--max-seq 8192`.
3. A different prefill schedule under MTP in the serve engine (`crates/intarsia-core/src/engine.rs`
   `Seq::prefill_step`; peel and tail passes): compare the dispatch count and ops per buffer of the two traces (`nd`, `t`
   fields).

**Method.** One `serve` per arm, the `ttft.sh` probe, the cb trace, and `INTARSIA_METAL_PROFILE=kernel:<file>` or
`INTARSIA_METAL_COMMIT_EVERY=1` with `cb` to see which ops share the GPU with the prefill buffers. Bisect with existing
knobs before reading code: `INTARSIA_METAL_MTP_GPU_ACCEPT=0`, `INTARSIA_BATCHED_MTP=0`, `--spec mtp --spec-k 1`,
`INTARSIA_SPEC_DEFAULT=0` (must reproduce the `--spec off` arm).

**Fix.** Whatever the cause, the fix must keep drafts and output bits: the head sees the same hidden rows in the same
order. Likely shapes: run the head's prompt catch-up as one grouped pass after the prefill chunk (prefill kernels at
M = chunk rows) instead of beside it; or defer it until after the first token is emitted, so TTFT is the prefill plus one
plain step; or remove a host-side copy. If the fix is a clear defect repair, make it the default with an off knob
`INTARSIA_METAL_MTP_TTFT_FIX=0` that restores main's behaviour for A/B; if it changes scheduling in a way that could move
decode speed, ship it opt-in (`=1`) and say so.

**Then, in the same lane, if time remains:** (a) the 27B (`~/models/Qwen3.8-27B-Quark-AWQ-MXFP4`): same probe, same fix
or its own cause; (b) confirm that under MTP the first token is streamed as soon as the prefill's logits exist and does
not wait for the first verify pass (if it waits, emitting it first is worth about 20 ms on the 35B and 50 ms on the 27B);
(c) measure a `--prefix-store` server with `INTARSIA_MTP_STORE=1` on Metal against the store alone (TTFT and decode on a
repeated 16k prefix): numbers only, no default change (it is idea 8 of the roadmap).

**Files.** `crates/intarsia-core/src/model/mtp.rs`, `model/spec_pipe.rs`, `engine.rs`, `engine/batch.rs`, possibly
`crates/intarsia-server` request handling. Do not edit `crates/intarsia-backend-metal/src/mxfp4.rs`, `kernels.rs`,
`attn.rs`, `rowtab.rs` or anything under `kernels/metal/` (lanes A and B).

**Pre-registered rule.** Same binary, fix off against on, `ttft.sh`-style probe with 6 fresh calls per arm, arms
interleaved A B B A: 35B median TTFT at 4k within 40 ms of the `--spec off` arm's (today +300 to +450 ms); at 34k within
150 ms. Decode tok/s (localeval `speed --sizes 4096,16384 --gen 256`, natural-v1 where the suite offers it) no cell below
0.98.

**Gates.**

1. Greedy ids, 256 tokens, fix on == fix off == `--spec off`: 35B and 27B, short / 4k (two prompts) / 16k, served; the
   acceptance line (`spec: verify passes=... accepted=...` on the server's stderr) identical between fix on and off for
   the same prompt (drafts unchanged).
2. `intarsia plan show --backend metal` byte-identical to main: 8B, 27B, 35B x default / `--spec off` / `--spec mtp`.
3. `cargo test --workspace --release --locked`; `crates/intarsia-models/tests/batch_metal_real.rs` (35B).
4. A regression test where CPU can show it (e.g. the number or order of head catch-up calls for a prompt of N rows);
   if only Metal timing shows it, say so and add the probe script to the results doc instead.
5. The speed rule.

**Stop rules.** (1) If the first two hours of bisecting do not isolate a cause, write up the traces and the knobs that
do and do not move it, and stop. (2) If the cause is on CUDA / HIP too and the fix touches shared scheduling that this
lane cannot test there, ship it Metal-only behind `Backend`-level gating or opt-in and flag it. (3) If the fix costs
decode speed (a cell below 0.98 twice), report both numbers and leave it opt-in.

**Deliverables.** Branch `lane-mtt/metal-mtp-ttft` from `origin/main`, worktree `/opt/code/intarsiaml/.worktrees/mtt`
(yours alone; delete `target/` when done). Results doc `docs/results/2026-11-15-mtt-metal-mtp-ttft.md`, raw logs under
`docs/results/2026-11-15-mtt-metal-mtp-ttft/raw/`, scripts under `scripts/mtt/` (copy `scripts/mac-plan/ttft.py`,
`ttft.sh`, `cbgap.py` from branch `mac-plan`, PR linked in the roadmap, if they are not on main yet). A row in
`docs/decisions/LOG.md`. PR against main; report the cause in one paragraph, the before / after TTFT table, the PR number
and the fleetq job ids.

**Rules.** GPU and timing work on endor goes through fleetq: `fleetq run --host endor --expect <MIN> --mem <GB> --quiet
gpu --label "mtt: ..." -- <cmd>`; builds and CPU tests use `--load cpu`. Never `--force`, `--no-queue` or `--exclusive`,
never release another owner's job, keep memory within 32 GB, do not stop `endor-kev` or `endor-tensorfold`. No tool call
or wait over 4 minutes: start long runs with `nohup ... &` inside the fleetq command and poll. macOS has no `timeout`: use
`perl -e 'alarm N; exec @ARGV'`. Commits end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`; the PR body
ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Never print secrets.

When you change code that can be run, built, or type-checked, run a real check that exercises the change before reporting it done: the project's tests, type-checker, or build, or the changed command itself. A syntax-only check, or a check command that failed to start, does not count; if all that is missing is the project's declared dependencies, install them with its own package manager and lockfile (e.g. npm install, pip install -r requirements.txt), never via sudo or the system package manager, unless told not to. Only if no real check can run here, say which one you did not run and why instead of reporting the change as done.

## 5. What this lane did not do

* No oMLX, mlx-lm or llama.cpp run today: the competitor rows are earlier records or the vendor's own numbers, dated in
  section 1.6.
* One measurement window; no ABBA. The TTFT finding is the exception (three separate serve pairs, all the same way).
* The cause of the TTFT penalty, and the split of the long-context verify cost between re-reads and per-read rate, are
  inferred from end-to-end numbers and left to lanes C and B to attribute first.
* The 35B's 16-row byte floor and per-kernel savings are lane mprof's of 2026-10-30, not re-measured.
* `scripts/mac-plan/raw/` does not keep the two per-dispatch traces (23 MB); `raw/ce1/table.txt` is their summary.
