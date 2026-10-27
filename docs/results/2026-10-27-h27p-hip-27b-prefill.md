# Lane H27P: Qwen3.8-27B MXFP4 tp2 prefill on the R9700s (HIP, gfx1201): profile, the 2.9x, and the first lever (2026-10-27)

Branch `lane-h27p/hip-27b-prefill`, built on origin/main `d877564c8` (PR #186, W4A8: fp8 activations are the 27B's HIP default). Host tatooine, 2x R9700 (gfx1201), `--tp 2 --devices 0,1`,
W4A8 (`mxfp4_act = fp8`), MTP k2 default for the served numbers. Every GPU number comes from a fleetq slot that takes radiance down (`--takes-down radiance --quiet gpu,cpu`), radiance restored
and `/v1/models` confirmed after each window (`raw/*/window.log`, `raw/w5/guard.log`...). Scripts: `scripts/h27p/`. Raw logs: `docs/results/2026-10-27-h27p-hip-27b-prefill/raw/<window>/`.

## Headline

| 27B tp2, W4A8, MTP k2, localeval `speed --sizes 1024,4096,16384 --gen 128 --reps 3`, 6 interleaved rounds | base (origin/main, chunk 256) | this branch | change |
|---|---:|---:|---:|
| prefill 1k tok/s | 2186 | **2808** | **+28.4 %** |
| prefill 4k tok/s | 2240 | **2974** | **+32.7 %** |
| prefill 16k tok/s | 2022 | **2913** | **+44.1 %** |
| decode 1k / 4k / 16k tok/s (median of 6 round medians) | 89.3 / 83.5 / 76.9 | 88.9 / 83.5 / 76.2 | x0.996 / x1.000 / x0.991 |
| decode of a stream beside an arriving 4k prompt: largest inter-token gap, `serve --max-batch 16` | 133 ms | 133 ms | same |
| greedy ids, 18 gate prompts x {`--spec off`, MTP default} | | identical to base (18/18) | |

* **The change**: `recipes/qwen3_5_vl.toml` `[tuning.hip]` gets `prefill_chunk = 2048`, `prefill_peel = 16`, `busy_chunk = 256` (a new recipe key, section 4). No kernel changes.
* **Where this leaves the gap to Radiance** (4k, tp2, 6133 tok/s): 2.74x before (2240), **2.06x** after (2974). What is left is mostly the all-reduce (section 3).
* Options measured, not shipped (section 6): bf16 all-reduce partials `INTARSIA_TP_PARTIAL=bf16` (existing knob) 3533 tok/s at 4k, +18.8 % over this branch, ids change on 12/12 prompts.

## 1. Profile: where a tp2 4k prefill goes (rocprofv3 kernel trace, one rank; 3858-token prompt, base chunk 256 and the new chunk 2048)

`scripts/h27p/w1.sh`, traces `raw/w1/p-c256.csv.gz`, `p-c2048.csv.gz`, `p-c4096.csv.gz`, classified by `scripts/h27p/trace_tp2.py` (per rank, the two ranks agree within 2 %; wall = the burst's span, profiler on, so
a few % slower than the plain runs below). Plain runs, `generate --max-tokens 1`, 3 reps: chunk 256 **1724 ms (2237 tok/s)**, 512 1423 ms (2709), 1024 1305 ms (2955), **2048 1258 ms (3067)**, 4096 1300 ms (2967)
(`raw/w1/t-*`, `raw/w2/t-*`; 3858 tokens).

Ranked, rank 0, % of the span:

| class | chunk 256 ms | % | chunk 2048 ms | % | achieved |
|---|---:|---:|---:|---:|---|
| GEMM, MXFP4 x fp8 (`a8_gemm_kernel`) | 812 | 44.8 | **541** | **42.4** | 135-140 -> **182-188 TFLOP/s** (37 -> 51 % of the 366 fp8 peak), table below |
| all-reduce (`ar2_kernel`, includes the wait for the peer) | 449 | 24.8 | **450** | **35.3** | 10.1 GB of f32 partials in 450 ms = **22.5 GB/s** of a measured 28.2 GB/s link (80 %) |
| GatedDeltaNet recurrence (`gdn_delta_rows_v3`, conv) | 112 | 6.2 | 108 | 8.5 | delta rule is a sequential scan over rows: 99 -> 99 ms (one launch per layer and chunk) |
| activation quant pre-pass + weight-span check (`mx_a8_quant`, `a8_wspan`) | 119 | 6.5 | 37 | 2.9 | 3 launches per GEMM (pre-pass, span check, GEMM); launch bound at 256 rows |
| flash attention (bf16 WMMA) + split-kv | 68 | 3.8 | 29 | 2.3 | 1.65 TFLOP causal / 29 ms = 56 TFLOP/s (30 % of the 190 bf16 peak) |
| norms (rmsnorm, gated rmsnorm, qk-norm-rope) | 36 | 2.0 | 40 | 3.1 | |
| elementwise, runtime fills/copies, lm_head gemv | 36 | 2.0 | 26 | 2.0 | |
| **idle (launch gaps, host)** | **182** | **10.0** | 44 | 3.4 | 52456 kernels per rank at 256, 9678 at 2048 |
| span | 1813 | | 1276 | | |

GEMM TFLOP/s per projection, from the real launches (the not-chosen twin of the fold / per-group kernel pair returns at once and is dropped; `trace_tp2.py --proj`, per rank FLOPs = 2 x tokens x N x K of the
rank's shard; grid y 34 and 68 are the 256- and 128-wide tile of the same gate+up op):

| projection (per rank) | chunk 256 | chunk 2048 | chunk 4096 |
|---|---:|---:|---:|
| gate+up (N 2 x 8704, K 5120) | 140.0 | 182.5 | 188.2 |
| down + gdn qkv + gdn out + attn o (N 5120) | 135.6 | 187.3 | 197.6 |
| gdn z (N 3072) | 108.6 | 181.3 | 196.9 |
| attn q+gate (N 6144) | 153.6 | 188.1 | 200.5 |

Measured ceilings of this box (lane mx, M = 4096): hipBLASLt fp8 259-321 TFLOP/s, our own W8A8 `f8_gemm_kernel` 188-218. The a8 kernel at 182-200 is inside the latter range.

**Link**: the two cards sit on separate PCIe 5.0 x16 root ports (`/sys/bus/pci`: 03:00.0 and 07:00.0, 32 GT/s x16 each). `scripts/h27p/p2pbw.hip` (`raw/w2/p2pbw.txt`): copy-engine, kernel stores into coarse or
fine-grained peer memory, 16 / 64 / 256 blocks, one direction or both at once: **28.2 GB/s per direction in every variant**. More blocks do not help; the link is the limit.

## 2. Roofline for a 4k prompt at tp2 (per rank)

Model: 64 layers = 48 GatedDeltaNet + 16 gated attention, hidden 5120, MLP 17408; 24.35 B linear parameters (MLP 17.11 B, GDN 5.56 B, attention 1.68 B) = 48.7 GFLOP/token.

| bound | per rank | time for 4096 tokens | tok/s |
|---|---|---:|---:|
| GEMM at the fp8 peak (366 TFLOP/s measured) | 99.7 TFLOP + 1.65 TFLOP attention | 277 ms | 14 800 |
| GEMM at hipBLASLt fp8 (259-321) | | 316-390 ms | 10 500-13 000 |
| GEMM at our a8 kernel (185) | | 548 ms | 7 500 |
| weights read once per chunk (6.5 GB at 640 GB/s) | | 10 ms per chunk: 160 ms at 16 chunks (256), 20 ms at 2 (2048) | |
| all-reduce, f32 partials (10.7 GB per direction at 28.2 GB/s) | 128 reduces of 4096 x 5120 x 4 B | **381 ms** | |
| all-reduce, bf16 partials | | 190 ms | |
| all-reduce at Radiance's 6 bits (wht6) | | ~71 ms | |

Without overlap the serial bound is GEMM + all-reduce: 277 + 381 = 658 ms (6 200 tok/s) with f32 partials even for a perfect GEMM, 277 + 71 = 348 ms (11 800 tok/s) at 6 bits.
Radiance's 6133 tok/s (668 ms) is therefore only reachable on this link by shrinking the all-reduce, overlapping it, or both. `GPU_MAX_HW_QUEUES=1` is set on this box (known-bad-default fix, `docs/ARCHITECTURE.md`): kernels of two
streams do not run side by side, so overlapping the all-reduce with the next GEMM needs a different queue setup first (not tried).

## 3. The 2.9x: what it is made of (design note, written from the profile before the change)

Radiance's own log (`/data/models/rad/native-serve.log`, read only): `quant=fp8_e4m3`, fused `ar_rmsnorm_quant_fp8` / `gate_quant_fp8`, prefill chunks of at most 2048, fp8 KV, `--tp-wire wht6`.

**Weight format accounts for ~0 of the gap.** Radiance loads 18.04 GiB from a 19.38 GB `.rad` and holds 9.68 GiB of static weights per rank (19.36 GiB total). The checkpoint's safetensors are 18.44 GiB
(MXFP4 codes + e8m0 scales + the bf16 embedding, lm_head and MTP head). fp8 weights would be 24.35 GB for the linears alone (22.7 GiB) before the 2 x 2.4 GB bf16 embedding and head: they do not fit in 19.4 GB.
Radiance runs the same MXFP4 weights with fp8 activations, W4A8, as this engine does since #186. Radiance's "fp8" is the activation and KV format.

Excess over Radiance for a 3858-token prompt (Radiance 629 ms = 3858 / 6133):

| source | base build (1724 ms plain run) | now (1258 ms) | status |
|---|---:|---:|---|
| chunking (16 chunks of 256 rows against Radiance's 2048). In the traces (span -537 ms): small-M GEMM tiles (135 -> 185 TFLOP/s: -271 ms of GEMM), launch gaps (-138), quant/launch (-82), attention (-39), the rest ~-7; the all-reduce does not change | **-466 ms closed here** (plain runs 1724 -> 1258) | | **measured**, 43 % of the gap |
| all-reduce: 450 ms against ~70 ms at 6 bits (wire time at 28.2 GB/s), if Radiance does not overlap | | **~380 ms** | estimated from the link measurement; **60 % of what is left** |
| GEMM: 541 ms against ~325 ms at a hipBLASLt-class 290 TFLOP/s | | ~216 ms | estimated from the ceilings above; 34 % of what is left |
| GDN scan, attention, norms, quant, gaps (215 + 44 ms ours) | | ~30 ms | not separable from outside Radiance |
| weight format | | 0 | arithmetic above |

Sum check: 629 (Radiance) = ~325 (GEMM) + ~70 (all-reduce) + ~230 (the rest, ours 259 incl. gaps) with no overlap. The 2.9x was therefore: chunking 43 %, all-reduce volume 35 %, GEMM efficiency 20 %, everything
else 2 %, weight format 0. The GatedDeltaNet scan (108 ms) is 8.5 % of our time and cannot be told apart in Radiance's; a chunk-parallel (WY) formulation would be a separate project (exact math changes).

## 4. The change

* `recipes/qwen3_5_vl.toml` (the 27B), `[tuning.hip]`: `prefill_chunk = 2048` (builtin 256), `prefill_peel = 16`, `busy_chunk = 256`.
  * **chunk 2048**: the whole +28..+44 %. 4096 is slower on this prompt (2967), 1024 and 512 are slower (2955 / 2709).
  * **peel 16**: without it one of 18 gate prompts flipped a token (`16k-3`, 16134 tokens, 6 tail rows at chunk 256: they run on the small-M row kernels, inside a 2048 pass they would run on the GEMM tiles; first
    attempt `raw/w3`, token 88). The lane-mx mechanism (`INTARSIA_PREFILL_PEEL`) cuts a 1..16-row tail into its own pass, so the tail schedule is the 256 one. With it: 18/18 and a sweep of 45 prompts whose length mod 256 covers 33 -> 0 -> 255.
  * **busy_chunk = 256** (new recipe key `[tuning.<backend>] busy_chunk`, `Recipe::resolve_busy_chunk`, read by `load.rs` when `--busy-chunk` is not given; `docs/recipes.md`): under `serve --max-batch` > 1 the loader
    made the busy chunk (a prompt that arrives beside decoding streams) equal to the model chunk. At 2048 that stalls the decoding streams for 650 ms per token (4k prompt arriving beside one stream), at 256 for 133 ms, the base build's number.
    This is the reason lane mx did not ship a larger chunk for this model (`raw/w10`: 653-661 ms, `raw/w12`: 132-134 ms for the recipe key, base 133-136 ms; the new prompt's TTFT 2.05 s with the key, base 2.06 s, 1.33 s without it).
    A prompt alone still prefills in 2048-row chunks.
* `crates/intarsia-models/tests/recipe_27b_chunk.rs` (new), `recipe.rs` unit test updated (it asserted "no builtin recipe sets a peel"; now: only the 27B on HIP).
* Kernels, `plan show` of the 35B and the 8B, CUDA and Metal: untouched (a recipe change). tp1 benefits too (section 5).

Why the bits do not move: the a8 GEMM is bitwise row-independent (M 1..4100, lane W4A8), the recurrence carries its state, attention and the norms are row-wise. Evidence beyond ids: `intarsia accuracy --mode both` on the W4A16 golden's
10 prompts (tp1) at `--prefill-chunk 256` and `2048` with `--write-self-golden`: **all 60 tensors (prefill and decode rows) byte-identical** (`raw/w11`, `dumpeq.py`). Those prompts are at most 700 tokens, so this compares 256-row chunking with a single pass; the 2048-row boundaries are covered by the 4k and 16k prompts of the ids gate.

## 5. Gates

| gate | result |
|---|---|
| CPU tests | `cargo test --workspace` on the Mac, final tree: 1031 passed, 0 failed, 24 ignored (the final tree, after adding the busy_chunk key) |
| kernel admission, >= 3 planted mutants | `intarsia kernels test --backend hip --quick` on the branch binary: 84 kernels, 414 cases: 74 admitted, 0 failed, 10 "no cases in --quick" (WMMA fp8 / bf16 pair variants; untested, not failed) (`raw/w8`, `raw/w14`); kernels untouched. Planted mutants of the change, **5 of 5 caught**: m1 no peel (`INTARSIA_PREFILL_PEEL=0`): `16k-3` ids differ; m4 all-reduce bug (`INTARSIA_TP_TEST_BUG=scale:1.003@1`): 3 of 3 prompts differ; m5 no collectives (`INTARSIA_TP_NO_COMM=1`): differs; m2 recipe chunk 2048 -> 1024 and m3 peel 16 -> 0: `recipe_27b_chunk` fails (`raw/w6`) |
| greedy ids at tp2, 18 prompts of `scripts/cmtp/ids-prompts` (6 short, 6 x ~4k, 6 x ~16k), 256 tokens, base chunk 256 vs branch `--spec off` vs branch MTP default | **18/18 identical** (`raw/w4`). First attempt without the peel: 17/18 (`raw/w3`, section 4) |
| tail sweep: 45 prompts cut from `4k-5`, lengths 4088-4129 tokens (mod 256 = 33 -> 0 -> 255), 48 tokens | 45/45 identical (`raw/w4`) |
| logit dumps chunk 256 vs 2048 (tp1, W4A16-golden prompts) | 60/60 tensors byte-identical (`raw/w11`) |
| prefill >= +10 % median over 3 interleaved rounds, decode >= 0.99x | 6 rounds (`raw/w5`, `raw/w7`, localeval, base and branch each a fresh `serve --max-batch 1`, order alternating): prefill **x1.284 / x1.327 / x1.441** (1k / 4k / 16k), decode x0.996 / x1.000 / x0.991. Decode is the same code; its round-to-round spread is +-5 % (MTP acceptance varies with the generated text), so the ratios are noise-level |
| G0: `plan show` of the 35B and the 8B unchanged | text and json byte-identical, and the 256 ids after a short and a 3.5k prompt identical, on HIP (`raw/w9`) |
| batched serving does not regress | context under `serve --max-batch 16 --mem-budget 29`: 19847 -> 19547 tokens (-1.5 %); `--max-batch 1` unchanged (`raw/w6`). Decode beside an arriving 4k prompt: 133 ms largest gap, equal to base (`raw/w12`). Texts of the arriving prompts (4k, 16k with a 6-row tail, 4k) identical on base `--max-batch 16`, branch `--max-batch 16`, branch `--max-batch 1` (`raw/w13`) |
| tp1 (card 0) | prefill 4k 1860 -> 2222 tok/s (+19.5 %), 16k 1716 -> 2130 (+24 %); ids identical on 4k-0, 4k-3, 16k-3 (`raw/w9/tp1`) |

Observation, not caused by this change: on the base build the first request of a fresh `serve --max-batch 16` produced a different text for the stream A than every later one (`raw/w13`, base16 `4k-0.txt`
A c158a.. against 8a3e..); the branch did not show it in one try. One datapoint, not investigated.

## 6. Options measured, not shipped

| option | numbers | why not default |
|---|---|---|
| bf16 all-reduce partials (`INTARSIA_TP_PARTIAL=bf16`, existing knob; half the bytes) | prefill 3290 / 3533 / 3425 tok/s at 1k / 4k / 16k = x1.50 / x1.58 / x1.69 over base, **+18.8 % over this branch at 4k**; decode x1.03 / x0.99 / x0.98 (`raw/w7`, 3 rounds) | rounds each rank's partial to bf16 before the sum: the ids change on 12 of 12 gate prompts (`raw/w8/bf16`). KL against the W4A16 golden at tp2 (`raw/w15`, `scripts/h27p/acc2.sh`; W4A8 vs W4A16 is the baseline offset): prefill 21-way KL 9.44e-3 (f32 partials, the default) -> 1.005e-2 (+6 %), decode 5.67e-3 -> 7.01e-3 (+24 %), top-1 prefill 98.3 -> 98.8 %, both far inside the 3e-2 static floor. MMLU n=1425 not run: an accuracy trade for the user, the gate for it is MMLU >= 0.815 |
| Radiance's wht6 (6-bit lossy all-reduce) | would take the all-reduce from 450 ms to ~70 ms: 4k about 1258 -> ~880 ms (~4400 tok/s) | lossy and not implemented here |
| exact reduce-scatter (f32) + all-gather (bf16) | 3 E bytes per direction instead of 4 E: -110 ms (-9 %) | a new pair of kernels; below the 10 % bar alone |
| all-reduce overlapped with the next GEMM | up to -340 ms | needs two HW queues (`GPU_MAX_HW_QUEUES=1` today) and a split of the row-parallel GEMMs |
| fp8 weights | 0 | do not fit (section 3) |

## 7. Next levers, in order of what the profile says

1. all-reduce volume (35 % of the time now): bf16 partials after an accuracy gate (+19 %), then something wht6-like; the exact RS+AG (+9 %).
2. overlap of the all-reduce with compute (needs the queue question answered first).
3. GEMM 185 -> hipBLASLt-class 250+ TFLOP/s (+10-15 %).
4. GDN scan (8.5 %): a chunk-parallel formulation.

## Windows and jobs

fleetq #5969 build; #5970 w1 profile + chunk A/B; #5980 w2 link microbenchmark + chunk 512 / 1024; #5983 w3 ids gate (17/18); #5985 w4 ids gate 18/18 + tail sweep; #5986 w5 localeval rounds 1-3; #5987 w6 mutants, admission, context; #5988 w7 rounds 4-6 + bf16;
#5989 w8 admission + bf16 ids; #5990 w9 G0 + tp1; #5991 w10 mixed latency (before the busy chunk key); #5992 w11 logit dumps; w12 mixed latency with the key; w13 texts beside a stream. Radiance restored and answering after each.

localeval runs (results under `/opt/code/deepeval/results/`, each also captured to unanimis by localeval): labels `h27p-w5-{base,dev}-r{1,2,3}` and `h27p-w7-{base,dev,bf16}-r{1,2,3}`, e.g. `20261027-022607-h27p-base-h27p-w5-base-r1`; pooled by `scripts/h27p/pool.py w5 w7`. `raw/w5` and `raw/w7` hold the serve start-up logs, the launch recipes and `steps.log`.
