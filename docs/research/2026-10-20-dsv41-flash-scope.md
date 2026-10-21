# Scoping: DeepSeek V4.1 Flash (EXL3 1.62 bpw, SAGE) in intarsiaml, with the Engram tables kept off the GPU

Status: research and design only (2026-10-20). No engine code changed, nothing downloaded (only HTTP range reads of safetensors headers), no GPU work, no fleet slot taken. Not committed.
Measurement scripts for every "measured" number are in `docs/research/2026-10-20-dsv41-flash-scope-tools/` (they write nothing but stdout and a header-only JSON).

Legend: **[M]** measured by me in this session, **[C]** cited from an external source (URL given), **[R]** recorded by another session in unanimis or the repo (id given), **[E]** my estimate or derivation (assumptions stated).

---

## 0. Headline

**Feasible: yes, but as a large project, not an onboarding.** The checkpoint itself fits the fleet (2 DGX Sparks, TP=2) only because 62% of its bytes (the Engram tables) live on NVMe. Our engine has none of the pieces the model needs except FP8 matmul, MoE and TP=2. Expect about **420 to 640 engineer-hours [E]** to reach correct TP=2 CUDA output, and more to reach the speed another engine already shows on the same boxes (TensorFold TP2, 61-67 tok/s decode [R a07d3d17]).

Three findings drive everything:

1. **Engram is a capacity problem, not a bandwidth problem.** A token needs 48 rows (24 per layer, 2 layers) of 256 B each, about 12.7 KB, against about 9.6 GB of other weights read per decoded token [M, E]. NVMe random-read capacity is 10^3 times above what decode needs. The only real constraints are (a) 203 GB of FP8 rows do not fit in RAM next to the model on two Sparks, and (b) on GB10 the page cache and the GPU share one 121 GiB pool, so `mmap` of the tables eats the budget.
2. **The lookup is a pure function of token ids** [C engram.py], so every read is known before the forward pass, and for decode a full layer of compute (1.9 to 3.5 ms) separates "token known" from "rows needed" at the earlier Engram layer [E]. This is what makes NVMe work. It is also what the other session measured: Engram on local NVMe gave 61-67 tok/s on TensorFold TP2; Engram on NFS gave 2.2 tok/s [R a07d3d17].
3. **The Engram tables are stored un-quantized (FP8 E4M3 + E8M0 scale) in the EXL3 file and in the base** [M]. EXL3 only covers routed experts, the head and the draft experts. So "1.62 bpw" describes the routed experts only; the 62% that is Engram stays at 8.25 bits per element.

**Recommended design: Variant A2** = row-sharded tables on local NVMe (one 101.4 GB shard per Spark), scales resident, 512 B `O_DIRECT` reads through `io_uring`, a frequency-pinned hot tier (about 1 GB), and the row fetch issued the moment a token id is known. Variant B (everything in RAM) does not fit at FP8 on two Sparks.

### 0.1 Byte budget (EXL3 file, 326.55 GB, 17 shards) [M]

Measured from the 17 safetensors headers (190,552 tensors) and `model.safetensors.index.json`. `total_size` in the index is 326,569,252,516 B; header sum is 326,549,980,508 B.

| Component | GB | % | Dtype / bit width in this file | Base repo (510.29 GB, 48 shards) |
|---|---:|---:|---|---|
| **Engram tables** (2 x `embed.weight` [~384M, 256] + `embed.scale` [~384M, 8]) | **202.76** | **62.1** | FP8 E4M3 rows, E8M0 scale per 32 elements. **Not EXL3.** 8.25 bits/element | identical 202.76 |
| Engram resident (`wkv` [25600, 6144] FP8, `q_weight`/`k_weight` [4, 5120] BF16) | 0.32 | 0.1 | FP8 block-32 | identical |
| **Routed experts** (40 layers x 384 x w1,w2,w3) | **109.93** | **33.7** | EXL3 trellis (I16) + `suh`/`svh` (F16) + `mul1` (I32). **1.608 bits/weight** over 543.6 B weights (trellis only) | 288.78 (MXFP4: I8 packed, E8M0 per 32) |
| Shared expert | 1.42 | 0.4 | FP8 block-32 | identical |
| Attention (wq_a, wq_b, wkv, wo_a, wo_b, compressor, sink, norms) | 5.11 | 1.6 | FP8 block-32 (BF16 compressor and norms) | identical |
| Indexer (8 layers) | 0.05 | 0.0 | FP8 + BF16 | identical |
| Hyper-connection (`hc_*_fn/base/scale`) | 0.16 | 0.0 | F32 | identical |
| Router gate (`weight` BF16 [384, 5120], `bias`, `bias_vl`) | 0.16 | 0.0 | BF16/F32 | identical |
| Embedding | 1.32 | 0.4 | BF16 | identical |
| LM head | 0.66 | 0.2 | EXL3 (`head.trellis` [320, 8080, 128] = K8) | 1.32 (BF16) |
| MTP / DSpark draft (3 blocks, 128 experts each, markov head) | 3.70 | 1.1 | experts EXL3 (3.10 GB), rest FP8/BF16 | 7.93 |
| Vision tower + aligner | 0.97 | 0.3 | BF16 | identical |
| **Total** | **326.55** | 100 | | **510.29** |
| **Resident without Engram tables** | **123.79** | 37.9 | | 307.53 |

EXL3 detail [M]: the experts use per-tensor K (bits per weight) of 1, 2, 3 or 4 (trellis last dim / 16). Parameter share: K1 53.8%, K2 33.6%, K3 10.6%, K4 2.0%. `suh`/`svh`/`mul1` overhead is 0.68 GB. Per-layer expert bytes range 2.32 to 3.37 GB.

### 0.2 Variant comparison

Decode bounds below are for the **dense path**; IO is never the binding term in any variant that touches NVMe sensibly (see section 6). Dense bound: about 9.6 GB read per token at bs=1 (6/384 of experts 1.72 GB + shared 1.42 + attention 5.11 + indexer 0.05 + hc 0.16 + gate 0.16 + Engram wkv 0.32 + head 0.66) [E from M bytes] / 221 GB/s measured GB10 bandwidth [R spark-tensorfold-tooling-lessons.md:17] = **about 23 tok/s on one Spark, about 42 to 46 at TP=2, before speculation** [E]. Measured by others: 13.4 tok/s no draft, 29.5-34 with DSpark k=5 on one Spark (vcruz recipe) [C]; 61-67 on TensorFold TP2 with the 2.9 bpw LibertAI file [R a07d3d17].

| Variant | Where the 203 GB lives | Per-token IO | Decode / prefill bound from the table path [E] | RAM cost per Spark (TP=2) | Effort over base port | Risk | Verdict |
|---|---|---|---|---|---|---|---|
| **A1** mmap of NVMe tables | NVMe, page cache | 48 faults x 4 KiB = 196 KB | decode fine; prefill capped by fault rate: 100-200K IOPS means 2-4K tok/s, close to compute prefill | page cache grows into the GPU pool; memguard trips seen [R a07d3d17] | 8 h | High on GB10 | **No** |
| **A2** row-sharded NVMe, `io_uring` + `O_DIRECT` 512 B rows, scales resident, hot tier, prefetch | NVMe, 101.4 GB per Spark | 48 x 512 B = 24.6 KB (24 reads per rank) | decode: IO 41K tok/s at 2M IOPS spec, so not binding. Prefill: 120K IOPS at 2500 tok/s before hot tier, 62K after (3% of spec) | 1 GB hot + 3.1 GB scales + 0.2 GB io buffers = about 4.5 GB | 40-60 h | Medium (graph capture, memguard) | **Recommended** |
| A3 same as A2 plus RAID0 across both Sparks | NVMe over network | n/a | none; row-sharding already aggregates capacity. Network block layer adds latency (NFS gave 2.2 tok/s [R]) | n/a | n/a | High | No |
| **B1** all rows in host RAM, FP8 | RAM | 0 IO | memory bound only (about 100 ns per row) | 101.4 GB per rank. Pool is 116 GB, model needs about 62 GB | n/a | **Does not fit** (203 GB vs about 90 GB free across both) | **Infeasible** |
| B2 all rows in RAM, requantized to 4 bit (4.25 bpw, 104.5 GB total) | RAM | 0 IO | same | 52 GB per rank on top of 62 GB model: no margin | 60 h + KL study | Very high (quality unknown, no margin) | Not now |
| B3 hot tier in RAM + NVMe tail | RAM + NVMe | cold fraction only | as A2 | 1-30 GB hot tier | included in A2 | Low | This is A2 with a bigger hot tier |
| B4 "Engram server" on endor RAM (about 60 GB free) or tatooine (30 GB) over the LAN | remote RAM | one batched request per layer, 6.3 KB per token-layer | latency 50-200 us per request, prefetchable; bandwidth trivial (0.8 MB/s at 60 tok/s) | 0 | 80 h (server + client) | Medium; endor is a Mac with other standing services | Only as a hot tier, not the full table |
| C1 CPU gathers and reduces, ships a vector | n/a | 51 KB per token-layer (4 keys + 1 value, 25600 dims bf16) | worse than shipping raw rows (6.3 KB); gate needs `h` on GPU | n/a | n/a | n/a | **No** |
| C2 PQ / low-rank / 3-4 bit rows | RAM or smaller NVMe | smaller | as A2 | 40-52 GB total | 60+ h | Quality unmeasured | Only if B is a hard requirement |
| C3 drop rare rows (zero fill) | n/a | fewer | as A2 | smaller | 20 h | Accuracy unmeasured, KL gate | Study after A2 works |
| C4 drop Engram entirely | n/a | 0 | n/a | 0 | 0 | Zeroing Engram output keeps only 29-44% of factual-knowledge benchmark scores (TriviaQA 29%) [C paper 6.3] | **No** (debug lever only) |

### 0.3 Decisions needed from you

See section 11. Short form: (1) is a 420-640 h project justified when TensorFold already runs this model at 61-67 tok/s on the same Sparks; (2) approve Variant A2 (spark2 is 93% full, but it already holds a full 203 GB Engram copy from #2853; halving it to a rank shard frees about 100 GB); (3) adopt EXL3 as a first-class quant format (ABI id 5 is already reserved); (4) text-only, 128k first, or full 1M; (5) fleet takedown windows for Phase 0.

---

## 1. Engram semantics (primary source: official `inference/engram.py` and `model.py`)

Sources: [engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/engram.py), [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/model.py), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/inference/config.json), paper [arXiv 2601.07372](https://arxiv.org/abs/2601.07372) ([html v2](https://arxiv.org/html/2601.07372v2)). I read both Python files in full (engram.py 8,138 B; the `ParallelEngramEmbedding` and `Engram` classes of model.py). There is no `modeling_deepseek_v41*.py` in the repo; the reference is `inference/`.

**Hashing [C].**
1. `compressed = token_map[input_ids]`. `build_compressed_token_map` normalizes each vocab entry with NFKC, NFD, StripAccents, Lowercase, whitespace collapse and Strip, and merges tokens that normalize alike (" The", "the", "THE"). It asserts the result has 99,092 entries. Image tokens are marked DEAD and break n-grams.
2. For each position, take the last 4 compressed ids (positions before the sequence start use `pad_id` = `token_map[2]`).
3. Per layer, per lookback `i`, a multiplier from `numpy.random.default_rng(10007 * layer_id)`, bounded by `(int64_max // 99092) // 2`, forced odd. `rolling = XOR_i (token_i * mult_i)`. The (i+1)-gram hash is the running XOR after step i.
4. Row = `rolling % prime[layer][ngram size][head] + offset`, with 8 distinct primes per (layer, n-gram size), searched upward from 16,000,000 and never reused. **All 8 heads of an n-gram share the XOR key and differ only in the modulus.** N-gram sizes 2, 3, 4 (so 3 sizes).

**I re-implemented this in numpy and it reproduces the checkpoint geometry exactly [M]:** compressed vocab 99,092, rows per layer 384,006,168 and 384,016,682 equal `engram_num_embeddings` in the config. This is strong evidence the hash is understood correctly (the tool also lets Phase 1 diff row indices against the Python reference).

**Rows and bytes.**
- Rows per token per layer: `(max_ngram_size - 1) * n_heads` = 3 x 8 = **24**. Two Engram layers (1 and 14), so **48 rows per token** [C].
- Row = 256 FP8 values + 8 E8M0 scales (one per 32 values). The scale is a **separate tensor** `embed.scale` [rows, 8] [M]. 264 B logical per row, 12,672 B per token.
- The 24 rows are concatenated (24 x 256 = 6144) and fed to `wkv` (FP8, [25600, 6144]), which outputs 4 keys (one per hyper-connection stream) and 1 shared value, each 5120 wide [C].

**Gate and merge [C model.py].** Per stream c: `dot = sum(h_c * q_w_c * k_w_c * key_c) * rstd(h_c) * rstd(key_c) * 5120^-0.5`; `gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))`; `h_c <- h_c + gate_c * value`. In V4.1 there is **no depthwise conv** (the paper's public demo has one; this class does not). The gate depends on the hidden state, but **the row addresses never do**.

**Prefetchability [C].** The reference computes all hashes once per forward from `input_ids` plus a token-history cache and passes them to the Engram layers. Rows for every token of a prefill chunk, or every draft token of a decode step, are therefore known when the step starts. The paper says the same (section 2.5: indices known before the forward pass, prefetch from host memory, placement at early layers so preceding computation hides latency).

**Sharding in the reference [C].** `ParallelEngramEmbedding` splits rows by contiguous range per rank, each rank gathers only its range (others masked to 0), then `all_reduce`s the dequantized bf16 rows. This is exactly the sharding we want for NVMe.

**Skew / hot-row cache: published vs measured.**
- Published: the paper cites Zipf's law and proposes frequent rows in HBM/DRAM and the tail on NVMe, but gives **no hit rates or cache sizes** [C paper 2.5, 6.4]. Its host-memory offload test (100B-parameter table in host DRAM, every lookup over PCIe, no cache) cost 1.9% (4B dense, 9,031.62 to 8,858.28 tok/s) and 2.8% (8B dense) on H800 [C paper 6.4].
- **Measured by me [M]** (script `engram_skew.py`): real DSV4.1 tokenizer and hash, on 3.0M tokens of text from `/opt/code` (783 files: markdown, Python, Rust, plain text), file-level train/test split, hot set chosen on train, evaluated on test (74.6M lookups). **The corpus is developer text, small, and domain-biased; treat the shape, not the absolute numbers.**

| Hot set (rows across all 48 tables) | Size at 264 B | Test coverage of lookups |
|---:|---:|---:|
| 10,000 | 2.6 MB | 15.4% |
| 100,000 | 26 MB | 30.8% |
| 1,000,000 | 264 MB | 47.7% |
| 4,000,000 | 1.06 GB | 55.9% |

  Per n-gram order with the 1M hot set: 2-gram 66.6%, 3-gram 43.0%, 4-gram 33.4%. Distinct rows touched / lookups on the test half: 0.32. A cold-start LRU over 150K tokens hit only 26-27% regardless of size (64K to 2M rows), so **a static, frequency-pinned hot set beats LRU** here. The numbers are sensitive to corpus size (a 0.3M-token run gave 33.8% at 1M rows), so the real profile must come from a representative corpus (Phase 0b). Log-trend extrapolation to 100M rows (26 GB) is 70-80% [E, low confidence].

## 2. Other novel blocks and how they map to engine ops

Source for all rows: official `inference/model.py` and `inference/config.json` (links above) [C], read by me; engine mapping from `crates/intarsia-core/src/ops.rs` (35 ops) [M].

| DSV4.1 piece (reference behaviour) | Engine status | Verdict |
|---|---|---|
| Attention: 64 q heads, **1 KV head, head_dim 512** (it is MQA with low-rank q, not MLA), q_lora 1280, partial rope 64, `attn_sink` per head, sliding window 128 | `AttnPrefill/AttnDecode/KvAppend` exist; kernels tuned for head_dim 128/256 | New kernel variants (hd512, sink, window), same op family |
| Sparse attention: window + compressed KV + top-512 indexed blocks via one `sparse_attn` | none | **New op** (gather-attention over an index list) |
| Compressor (gated pooling, `wkv`/`wgate` [512, 5120], ratio 2 on layers 2-19, ratio 1 on 20-39, none on 0-1) | none | **New op + new state** |
| Indexer: 32 heads, dim 128, top-k 512; `candidate_source_layer` 20 keeps top 2048 blocks of 8 | none | **New ops** (score einsum, top-k, candidate mask) |
| Cross-layer sharing: only layers 2, 8, 14, 20 own a compressed KV cache; indexers on 2, 8, 14, 20, 24, 28, 32, 36; `SharedAttentionRuntime` hands slots down the stack | `SlotSpec` is per layer (`state.rs:18`) | **New core concept**: shared KV/index slots with writer layer and reader layers |
| Hyper-connections: residual is 4 streams; `hc_pre`/`hc_post` with a 4x4 `comb` after 20 Sinkhorn iterations (fp32) | residual is a single stream (`Add`, `RmsNormAdd`) | **New core concept**: multi-stream residual in plan compiler + 3 new ops |
| Router: `sqrt(softplus(x W / T))`, `topk(scores + bias, 6)`, weights = unbiased scores, scale 1.5; 384 routed + 1 shared; extra `gate.bias_vl` tensor (purpose unclear; likely the vision-language bias, **check**) | `MoeTopk` assumes softmax-style | Attribute/variant of `MoeTopk` |
| `o_groups` 8: block-diagonal low-rank `wo_a` then dense `wo_b` | none | New op (grouped matmul) |
| SwiGLU with `swiglu_limit` 10 (up clamped both sides, gate clamped above) | `SiluGlu` | Attribute + kernel change per backend |
| YaRN x16 from 65,536 on compressed layers; plain theta 10,000 on layers 0-1; `norm_eps` 1e-20 (needs fp32 math) | `Rope` has no scaling (`adding-a-model.md:39` lists yarn as unsupported) | New rope variant with two tables |
| FP8 block quant with **E8M0 scales** | `MatmulFp8` exists | **Check** scale dtype and block shape assumptions |
| MTP + DSpark draft: block 5, taps layers 37-39 (mean over the 4 streams), 128 experts top-3, Markov head rank 256, confidence head | only a design placeholder (`ARCHITECTURE.md:232`) | New spec-decode machinery; Phase 4 |
| Engram (this doc) | none | New core concept: off-device table tier + gated merge op; graph capture must not wait on IO |
| Vision tower (32 layers, 0.97 GB) | anti-goal for now ("vision encoders are a block type, not a side product") | Skip; text only |

## 3. EXL3 and quantization strategy

Sources: [exllamav3](https://github.com/turboderp-org/exllamav3) (kernels `exl3_gemv_kernel.cuh`, `exl3_moe.cu`, `quant/codebook.cuh`), [vcruz305 model card](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) and [recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe); local MIT reference `/opt/code/TensorFold/src/tensorfold/cuda/exl3/` [M].

**Format.** A tensor is `trellis` (I16, [in/16, out/16, 16*K]), sign vectors `suh`/`svh` (F16), a codebook selector (`mul1`, or `mcg`; `mul1` here). Tiles are 16x16; each weight is a 16-bit window into a bit stream decoded through a multiply-based codebook (cb0 "3INST", cb1 MCG, cb2 MUL1); the reconstruction is `had_right(had_left(w) * suh) * svh` with 128-point Hadamards, so dims must be multiples of 128. Fractional bpw comes from mixing integer K across tensors (here K 1 to 4 in the experts, K 8 in the head) [C exllamav3, M header shapes]. SAGE, the mixed-K allocator, is unpublished [C model card].

**Is a decode port realistic?** Yes, with moderate effort:
- The decode math is integer bit-extract, a multiply-add codebook, and an FP16 butterfly; no tensor-core-only trick is needed except for speed. TensorFold already carries a MIT, torch-light implementation: `decode.cuh` (162 lines, tile decode into `mma.m16n8k16` B fragments), `linear.cu` (347), `experts.cu` (360), `experts_grouped.cuh` (356), and a **pure numpy reference decoder** `format.py` (386 lines: `unpack`, `hadamard`, `dequantize`, `forward`) [M]. The numpy decoder is our `dequant_ref` oracle (`QuantFormat::dequant_ref`, `ARCHITECTURE.md` section 3.4) for free. ADR-0002 already plans TensorFold as the first kernel pack; `IK_QUANT_EXL3 = 5` is reserved in `kernels/abi/ik_abi.h:77` [M].
- exllamav3 has HIP code paths (`ptx_mma_m16n8k16` substitute, RDNA MoE paths) [C]; I found no Metal code. A HIP/Metal port is plausible but not needed: no non-Sparks box in the fleet can hold the 124 GB (tatooine has 2 x 32 GB, endor's Mac 128 GB is too tight with its other services) [M, R 8da5d89b].
- Cost of the Hadamard: two 128-point butterflies per matmul (input side `suh`, output side `svh`); the upstream GEMV needs two `grid.sync()` calls [C]. vcruz measured the MoE kernel at 47.6% of DRAM peak, "bound by instruction issue", and a step breakdown of MoE 84 ms, non-MoE linears 36.6 ms, elementwise 8.5, head 5.7, draft MoE 6.1 of about 141 ms [C recipe]. So the dense path is far from the bandwidth floor (about 9.6 GB / 221 GB/s = 43 ms/token) and has headroom to improve.

**Dequantize at load into our existing formats? No.** EXL3 1.62 bpw experts are 109 GB; as NVFP4 (4.5 bits with scales) they would be about 305 GB [E: 109.25 GB x 4.5 / 1.608], which does not fit two Sparks (about 232 GB schedulable) and destroys the point of the checkpoint.

**Requantize from the base into our NVFP4/W4A16? No.** The base experts are already MXFP4 (I8-packed e2m1, E8M0 per 32) at 288.8 GB [M]; any 4-bit format is about 270 to 305 GB of experts alone, which only fits 4 Sparks. Sub-4-bit is mandatory for a 2-Spark fit. The engine's own formats are NVFP4 and FP8 only (`quant/scheme.rs` rejects INT4_AWQ) [R explorer, spot-checked]. Our own 2/3-bit requantizer would be a research project with quality risk; EXL3 K mixing is where the quality comes from.

**Quality ceiling.** The vcruz card reports mean KL 0.098, top-1 agreement 94.52%, top-5 overlap 81.16%; by category top-1 is code 99.1%, reasoning 96.0%, math 95.6%, general 89.9% [C model card]. These are teacher-forced against the FP8/FP4 original. The recipe README's own gate shows 78-97% top-1 on 32 positions [R 8da5d89b]. That is the model's quality, whichever engine runs it. Our KL gate can only prove we match the EXL3 reference, not the original.

## 4. intarsiaml readiness

From `docs/ARCHITECTURE.md`, `docs/adding-a-model.md`, `docs/accuracy.md`, `docs/decisions/`, `crates/intarsia-core/src/{ops,state,plan}.rs`, `crates/intarsia-models/src/{recipe,load,safetensors}.rs`, and a repo-wide grep (my own, after a delegated read whose output I spot-checked) [M].

- **Quant formats:** NVFP4 and FP8 implemented; EXL3, MXFP4, GGUF listed in the design only. No trellis, codebook or Hadamard-rotation weight code anywhere in `crates/`, `kernels/`, `recipes/`. The only Hadamard is a lossy all-reduce transport option.
- **Ops:** 35 in the `Op` enum (`ops.rs:60-100`). New ops must be appended at the end (admission sweeps seed case data from the discriminant).
- **Core concepts missing:** multi-stream residual; shared (cross-layer) KV and index slots (`SlotSpec` has only `AttnKv` and `Recurrent`, per layer); an off-device table tier; graph segments that can wait on IO.
- **Recipes:** `recipes/qwen3_5_moe.toml` is the nearest analogue (MoE, `pattern_by_layer`, per-backend tuning). Everything DSV4.1 adds is a code change, not a recipe (`adding-a-model.md` "When you need code").
- **TP:** requirement 12 in `ARCHITECTURE.md:13` is "TP=X" (the twelfth item of the priority list). Status per section 5 and ADR-0006: **TP=2 works** (NCCL for Sparks, RCCL for R9700, a one-shot all-reduce); self-golden TP budgets exist (`accuracy.md`). TP>2, pipeline parallel and **expert parallel are designed, not built.** DSV4.1 on 2 Sparks wants expert-wise or TP sharding of 384 EXL3 experts; EP-style sharding by expert is the natural fit and is new.
- **Host offload / mmap:** `safetensors.rs` mmaps sharded checkpoints and hands tensor windows through `HostMemory`; Metal wraps it zero-copy. There is **no weight streaming, no NVMe tier, no paged table**; every weight is expected device-resident.
- **Accuracy gate:** KL/top-1 against HF or vLLM goldens (`tools/golden.py`, `golden_vllm.py`; goldens exist only for qwen3-8b, qwen3-reranker, qwen36-35b-a3b-nvfp4). A DSV4.1 golden needs a reference implementation that can run it (see Phase 0).
- **"1-2 h new-model" target:** I found no "1-2 h" figure in the docs. The nearest is the onboarding pipeline in `ARCHITECTURE.md` section 7 (inspect 2 min, load and gate 5, quick bench 15, deep 2-3 h) and `adding-a-model.md` (recipe-only models). **Not applicable here:** DSV4.1 triggers nearly every "needs code" condition (new attention, yarn, unknown quant, multi-stream residual).
- **State of the tree:** CUDA round 7 and HIP round 10 are paused until the weekly quota resets (memory note `paused-until-reset`); Metal and CUDA are on main. Any DSV4.1 CUDA phase competes with those resume points.

## 5. Fleet hardware (read-only commands, measured 2026-10-20)

| Host | CPU | RAM | Local storage | Network | Notes |
|---|---|---|---|---|---|
| spark1 (spark-4e77) | GB10, `nproc` 20 | 121 GiB total, 91 GiB available at the time (**taken while fleetq #2864 was running on it**) | Samsung MZALC4T0HBL1-00B07, 3.7 TB, **PCIe 32 GT/s x4 (Gen5)**, logical/physical block **512 B**; `/` 82% used, 664 GB free | 4 CX-7 interfaces (`enp1s0f0np0`, `enp1s0f1np1`, `enP2p1s0f0np0`, `enP2p1s0f1np1`), each **200,000 Mb/s**, RoCE (`rocep1s0f0` etc., all PORT_ACTIVE, fw 28.45.4028), subnets 10.100.152-155.x | #2864 holds 110 GB exclusive |
| spark2 (spark-fa00) | GB10, 20 | 121 GiB, 49 GiB available (retro service up) | same SKU, same Gen5 x4; `/` **93% used, 272 GB free** | same four 200G ports | already holds a full 203 GB Engram copy from #2853 (confirmed), which is most of why it is full; vcruz saw 509 ms/step on a 97%-full disk [C recipe] |
| tatooine | x86_64, 24 threads | 30 GiB | WD_BLACK SN7100 2 TB + Samsung 970 EVO Plus 250 GB | n/a | 2 x R9700 (32 GB each) |
| endor | Apple M5 Max, 18 cores | 128 GB unified | Apple SSD AP4096Z 4 TB | n/a (ssh to itself skipped) | about 60 GB free with kev and tensorfold standing [R fleetq] |

NVMe spec (no fio run, per instruction): the exact SKU `MZALC4T0HBL1-00B07` has no public datasheet that I could find; the sibling `MZVLC4T0HBL1-00B07` PM9E1 4 TB lists up to **14,500 MB/s sequential read and 2,000,000 IOPS random read** ([retailer listing](https://esaitech.com/products/samsung-pm9e1-mzvlc4t0hbl1-00b07-4096gb-m-2-pci-express-gen-5-x4-solid-state-drive)) [C, unverified; vendor "up to" figures at deep queue depth]. Gen5 x4 gives a 15.75 GB/s ceiling [E]. GB10 memory bandwidth: 221 GB/s measured [R `spark-tensorfold-tooling-lessons.md:17`]; the doc uses only the measured figure. **GPUDirect Storage is not supported on DGX Spark** (`gdscheck` flags GB10 as an unsupported model; secondhand [forum thread](https://forums.developer.nvidia.com/t/gds-support-for-dgx-spark/350328)). It is also unnecessary: host and GPU share the pool.

## 6. Prior art in the fleet

- **fleetq #2853** ("tonyd2wild DSv4.1-Flash TensorFold TP2 test") **finished 2026-10-19 23:16 UTC** and both Spark vLLM services were restored; the brief's "running" is stale. Outcome [R unanimis a07d3d17, agent-run, unverified by unanimis]: the TensorFold TP2 lane serves; **decode 61-67 tok/s (2x our vLLM TP2) at 256K context and 2 streams**, 16K prefill +91% vs vLLM; MMLU 0.878. The engine reads Engram rows with **`O_DIRECT` AIO 4 KiB random reads**. **Engram over NFS: 2.2 tok/s and no first token on a 4K prompt in 657 s** (the stack and latency matter, not the link). Engram on NVMe with the repo's 1M-context, 4-stream config tripped a 2 GiB memguard (MemAvailable 1.66 GiB); at 262,144 context and 2 streams the low point was 2.9 GiB over 1,486 requests. Weights there were the LibertAI 2.9 bpw file, not this one, and **each node held a full 203 GB Engram copy** (2 x 101.5 GB per node), which is why spark1's root disk (111 GB free then) was a problem. Row-sharding per rank, as the reference model does, halves that.
- **fleetq #2864** (started 2026-10-19 23:31 UTC, session claude-32ebb62b): "vcruz305 DSV4.1-Flash 1.62bpw SAGE-EXL3 one-Spark recipe: download, base image build, serve, localeval", running on spark1 with flight, guidance and capcom down. **This is the exact file this doc scopes, and it is most of our Phase 0.** Still running at 04:2x UTC (`fleetq show 2864`: running, 52 min), and no outcome was recorded at the time of writing (unanimis recall at 04:17 UTC returns only the earlier plan record 8da5d89b). Re-check before acting; do not contact or interfere with that session.
- **unanimis 8da5d89b** (proposed, unverified): the vcruz one-Spark recipe measured 29.5-34 tok/s with DSpark k=5, about 720 tok/s real-text prefill, 3 min 11 s boot, about 4 GiB MemAvailable left at serve time; needs a base image built from source (about 64 GiB MemAvailable on an idle node); endor cannot run it (CUDA-only).
- **TensorFold:** the local checkout `/opt/code/TensorFold` has the MIT EXL3 decoder (section 3), a `deepseek_v4` family (that is **V4-Flash, not V4.1**), and `qwen4_exp/cuda/exl3_pack.py: NgramTable` (4-bit-row n-gram shards, memory-mapped, with prefetch) [M]. A grep with quoted globs found no "engram" string in the checkout; the Engram code for V4.1 lives in the separate `bertholomus/TensorFold` `deepseek-v41-zig` engine used by #2853 [R]. `NgramTable` is design precedent for prefetched, quantized n-gram rows.
- **vcruz recipe:** Engram "streamed from NVMe" through `io_uring` (needs `seccomp=unconfined`); a second off-GPU tier holds the 1,380 coldest experts plus the embedding in an 8.59 GiB 64-byte-aligned alias file; engine reservation 110.5 GiB of 121.7 GiB; `guard.sh` floor 0.5 GiB MemAvailable [C recipe].

## 7. Design: where everything lives (TP=2 over two Sparks) [E unless stated]

Per Spark, 116 GB schedulable (fleetq roster, 121 GiB box).

| Item | Per rank | Notes |
|---|---:|---|
| Non-Engram weights, sharded | about 62 GB | 123.8 GB / 2; embed, head, hc, gate, 1-KV-head KV replicated add a few GB. Experts: shard by expert (EP) or by moe_intermediate |
| Engram resident (wkv, q, k) | 0.16 GB | FP8, sharded by output column or replicated (0.32 GB) |
| Engram scales (E8M0, 6.14 GB total) | 3.07 GB | resident, so a row costs one 256 B read, not two |
| Engram tables on NVMe | 101.4 GB disk | contiguous row range per rank, as the reference does; spark2: reuse and halve the 203 GB copy #2853 left in `/var/tmp/tf1017-tp2/engram` (confirmed 190 GiB), no new download |
| Hot tier (frequency-pinned rows) | 1-4 GB | 4M rows = 1.06 GB = 56% of lookups on my dev corpus [M] |
| KV + indexer state, bf16 | 0.55 GB at 128K, 4.4 GB at 1M | derived from reference buffer shapes: window 128 x 512 on 40 layers (5 MB); compressed KV only on layers 2, 8, 14 (ratio 2) and 20 (ratio 1) = 2.5 x 512 elements/token; index_k on 8 layers = 6.5 x 128 elements/token; total 2,112 elements/token = 4.2 KB bf16 (2.1 KB if FP8). Single KV head cannot be sharded by head, so replicated. Per sequence. |
| Workspace, CUDA context, graphs, draft model | 6-8 GB | guess |
| OS + guard reserve | 4 GB | memguard floors seen: 0.5 and 2 GiB [R, C] |
| **Total** | **about 75-85 GB** | leaves 30-40 GB; could hold a larger hot tier or several sequences |

One-Spark (TP=1) does not fit with margin: 123.8 GB of non-Engram weights exceed what is schedulable, which is why the vcruz recipe pins a cold-expert alias on NVMe. We do not need that trick at TP=2.

**Collectives.** Engram adds 2 small exchanges per step (layers 1 and 14): each rank reads its half of the 24 rows; either `all_reduce` the bf16 rows (12 KB per token-layer, as the reference does) or exchange raw FP8 rows plus scales (6.3 KB), then run `wkv` replicated or column-sharded. Both fit into the existing per-layer all-reduce rhythm (80+ per step); at about 25 us each [E] the Engram share is under 2% of a step.

## 8. Variant A in detail: NVMe

**Addressing and layout (E).**
1. Repack once at load (or offline): per layer, per rank, a flat file of 256 B rows (FP8 payload only, 16 rows per 4 KiB page, no straddling) with scales in a separate resident array. One logical row = one 256 B read inside one 512 B sector. 101.4 GB per rank. No padding is needed at 256 B per row.
2. **Frequency-sorted ordering** (optional, profile-driven): sort rows by measured access count so the hot head is one contiguous region that can be pinned with one read and cached with no per-row map. Needs the Phase 0b corpus profile.

**Read path.**
- `io_uring` with `O_DIRECT`, registered buffers, `IORING_SETUP_SQPOLL` optional, QD 64+, 512 B reads, into a pinned buffer the GPU can address (on GB10 the same pool). `mmap` is rejected: the page cache is GPU memory on a unified-memory box, fault-driven IO tops out near 100-200K IOPS [E], and the other session's memguard trips point the same way [R].
- Logical block size is 512 B [M], so each row costs 512 B at the host interface (24.6 KB per token) instead of 4 KiB (196.6 KB per token). Flash internally still reads NAND pages (typically 4-16 KiB), so device IOPS, not bytes, is the budget; the 2M IOPS figure is at 4 KiB.
- Requests per step: 48 x (tokens in step) minus hot-tier hits, minus duplicates.

**Decode schedule.** The step's token ids are known at step start (verify block of 5 DSpark drafts, or the single sampled token). Issue all reads for layers 1 and 14 immediately.
- Layer 1 Engram: one layer of compute (layer 0 attention + MoE) hides it: 1.9 ms/layer at the measured no-draft rate (13.4 tok/s = 75 ms/token) and 3.5 ms/layer at 141 ms per DSpark step [C recipe, E]. A QD-24+ burst of 512 B reads completes in 100-300 us [E, typical Gen5 TLC]. About 10x slack.
- Layer 14 Engram: 13 layers (25-45 ms) of slack.
- Speculative prefetch from top-k logits is unnecessary at this slack; keep as a fallback if layer 1 ever stalls. DSpark drafts already give 5 tokens of look-ahead for free.
- Graph capture: do not capture the IO wait. Either split the captured graph at layers 1 and 14 (segments 0-1, 1-14, 14-end) or poll a host-visible completion flag from a device-side wait kernel. This is the only change to engine control flow and is a new core concept.

**Prefill.** All 48 x N reads are known up front. Sort and deduplicate per chunk (global unique/lookup is 0.32 on my corpus, per prompt it will be higher [M, E]), service the hot tier first, issue the rest in sorted order as one large batch for layer 1, then layer 14 while layers 2-13 compute. At 2,500 tok/s compute prefill (TensorFold, [R a07d3d17]) the raw demand is 120K IOPS, 62K IOPS with a 264 MB hot tier [E from M], 3% of the 2M IOPS spec, and half of that per rank at TP=2.

**Bounds [E].**
- Decode IO: 48 reads per token, about 2.9K IOPS at 60 tok/s; ceiling at 2M IOPS = 41.7K tok/s; at a pessimistic 200K IOPS, 4.2K tok/s. Never binding.
- Prefill IO: ceiling 41.7K tok/s at spec (no hot tier), 80K with a 48% hot tier; pessimistic (500K IOPS) 10K tok/s. Compute-bound at 0.7 to 2.5K tok/s.
- Latency tail is the real risk: a single 2-5 ms NAND stall (garbage collection on a 93%-full disk) lands in the step. Mitigation: keep the disks below 85% full and log read p99. Zero-filling a late row is not an option because it changes the output; the step must wait.

**RAID0 across both Sparks.** Row-range sharding already uses both NVMes. A network block device would add 100 us to ms of latency per read (NFS measured 2.2 tok/s [R a07d3d17]) for zero throughput benefit at these rates. One-sided RDMA READ of a remote hot row (about 2-5 us [E]) is the only network path worth building, and only for a hot tier larger than local RAM.

## 9. Variant B in detail: RAM

On GB10 there is no "host RAM versus GPU memory": one LPDDR5x pool. "Off the GPU" can only mean "outside the weight budget and the CUDA allocator", i.e. pinned host buffers.

- **B1 (full FP8 in RAM): infeasible.** 203 GB vs about 90 GB free across both Sparks (about 40-45 GB each after weights, KV, workspace and guard) [E]. Even with the model at TP=2 and nothing else on the boxes, 101.4 GB per rank plus 62 GB weights is 163 GB per rank against 116.
- **B2 (requantize rows).** 4-bit (E2M1 + E8M0 per 32 = 4.25 bit): 202.76 x 4.25 / 8.25 = 104.5 GB total, 52 GB per rank, which leaves 5-10 GB of slack on the 40-45 GB estimate above: no margin. 3-bit would fit (78 GB total) but needs a new kernel and is unmeasured. The rows are summed through a linear `wkv` over 24 concatenated vectors and then gated, so noise may average out [E, plausible not shown]; this is exactly what the KL gate decides. Precedent for 4-bit n-gram rows: TensorFold `NgramTable` for Qwen4-exp [M].
- **B3 (hot tier in RAM, tail on NVMe)** is Variant A2. The memory above says 30-40 GB per rank is spare, so the hot tier can be sized by measured hit rate, not by what fits.
- **B4 (engram server on another box).** Endor has about 60 GB free RAM, tatooine 30 GB. Neither holds a table. As a hot tier: 60 GB / 264 B = 227M rows, which would be about 30% of all rows and, by extrapolation, 70-80% of lookups [E]. Request shape: one batched request per Engram layer per step (24 rows x 264 B = 6.3 KB per token-layer, 0.8 MB/s at 60 tok/s, 32 MB/s at prefill 2.5K tok/s). Latency 50-200 us per request on a LAN, so it prefetches like NVMe. Not worth building unless the local spare RAM turns out to be needed for something else.
- **C1 CPU-side lookup+reduce, ship a 5120-dim vector: loses.** The consumer is `wkv` followed by a per-stream gate, so what the GPU needs is a 25,600-dim key/value (51 KB bf16) per token-layer, and the gate needs `h`. Shipping the 24 raw rows (6.3 KB FP8) and doing `wkv` on the GPU is 8x smaller and uses the resident FP8 matmul. CPU `wkv` for prefill (0.63 GFLOP per token, 2 layers) would need 1.6 TFLOP/s at 2.5K tok/s [E].

## 10. Accuracy plan

- Engram row requantization and dropping: gated by the engine's KL gate, not by estimates. Proposed gates [E, to be set in Phase 0 once the reference noise floor is known]: mean KL vs the EXL3 reference at most 2x the reference's own run-to-run KL and top-1 agreement within 0.5 points on the 64 x 128 golden set.
- Never drop Engram or zero-fill missed rows to hide IO latency: removing Engram output keeps only 29-44% of factual-knowledge scores [C paper 6.3].
- Reference greedy output is not bit-deterministic (vcruz notes it recommends comparing log-probs, not tokens) [C recipe]; use top-k log-prob goldens (`intarsia-golden-topk-v1` exists) with measured noise budgets, the same method as the TP self-golden budgets.

## 11. Phased plan (estimates are engineer-hours, mine, wide ranges)

| Phase | Work | Hours | Go / no-go gate | Main risks |
|---|---|---:|---|---|
| **0** Reference and goldens (no engine changes) | (a) Collect #2864's outcome (do not interfere); if it fails, rerun the vcruz recipe on one Spark under fleetq. (b) Produce top-k log-prob dumps for 64 prompts x 128 positions, twice, to measure the noise floor. (c) Dump per-layer activations on a few tokens for Phase 1 unit tests (needs the reference's tap points). | 8-12 (about 3 fleet-hours, Spark takedown of flight/guidance/capcom) | Two reference runs agree to a stated KL; recorded decode and quality numbers reproduce the card within 10% | Spark takedown windows; base image needs about 64 GiB to build; reference nondeterminism |
| **0b** Engram profile (CPU only) | Re-run `engram_skew.py` on a representative corpus (chat, web, code, multilingual), 50-200M tokens; record per-table counts and a frequency-sorted row order. | 4-6 | Hot-set curve known to 100M rows; decision on tier sizes | Corpus choice; the training distribution is unknown |
| **1** CPU / reference path in Rust | EXL3 tensor-group parser + `dequant_ref`, bit-exact vs `format.py`; Engram hash, bit-exact vs `engram.py` on 1M tokens; CPU reference ops for hc/Sinkhorn, router, sparse attention with sink/window/compressor/indexer/candidates, grouped O, swiglu clamp, YaRN, FP8 E8M0 matmul, EXL3 MoE; multi-stream residual and shared-slot concepts in the plan compiler; recipe `deepseek_v41`. The model cannot be held in RAM on any CPU box, so the CPU path streams layers from mmap. | 120-160 | Per-op parity vs numpy/torch reference; layers 0-2 hidden states match to 1e-4 relative; 8-position full-model log-probs within the Phase 0 noise budget (slow, tens of seconds per token) | Hidden detail in the reference (`gate.bias_vl`, `kernel.py` Sinkhorn, compressor); no ground truth for intermediate activations unless Phase 0(c) delivers it |
| **2** CUDA single-rank kernels (sm_121) | Port TensorFold EXL3 linear/MoE through the C ABI as a kernel pack; hd512 sparse attention with sink; hc/Sinkhorn fused ops; router; compressor/indexer; each admitted by the sweep against the CPU path. Engram tier as a library (`io_uring`, hot tier, prefetch) with its own tests on a spare NVMe. | 100-140 | Each new op passes admission on sm_121; EXL3 MoE at least 40% of DRAM peak (vcruz bar: 47.6%); Engram reader sustains 100K IOPS at p99 under 1 ms in isolation (needs a fleet slot, not before) | sm_121 quirks (build `sm_121`, not `sm_120`); instruction-issue-bound kernels |
| **3** CUDA TP=2, correct | EP-style expert sharding or TP sharding, row-sharded Engram per rank, graph split at layers 1 and 14, memguard integration, context 128K. | 80-120 | Self-golden KL within budget on the golden set; Engram stall under 2% of step; 1,000 requests without a memguard trip | Memory pressure on GB10 (4 GiB headroom seen in other sessions); spark2 disk at 93%; NCCL on RoCE |
| **4** Speed and features | DSpark / MTP speculative decode, batched prefill, long context (1M needs indexer work, 4.4 GB KV), hot tier tuning, prefix reuse. | 100-200 | At least 0.8x TensorFold TP2 on the same weights class (so about 50 tok/s decode) with accuracy gate green | Spec-decode machinery is design-only today; indexer cost at 1M |
| **5** Optional | Other backends (HIP cannot hold it; Metal only with expert streaming), vision tower | not scoped | n/a | |

**Total to correct TP=2 CUDA (phases 0-3): about 310-440 h; to competitive speed (phase 4): 420-640 h** [E]. The Metal backend scope doc used the same style (50-220 h blocks) as a calibration point.

### Decisions needed from you

1. **Go at all?** TensorFold already serves this model class on these Sparks at 61-67 tok/s. The case for building it in intarsiaml is modularity, the accuracy gate, and an owned engine. If that is not strategic, stop after Phase 0 and keep the goldens.
2. **Engram variant:** approve A2 (NVMe row shards). spark2 is 93% full (272 GB free) but `/var/tmp/tf1017-tp2/engram` there is a full 203 GB copy left by #2853 (`ls`/`du` confirmed 190 GiB, shards 47 and 48 of the LibertAI file); reuse and halve it. Adding a fresh shard instead would leave about 170 GB free (about 95% used), near the 97% where vcruz saw 509 ms/step, so also decide whether to clear about 550 GB to get under 85%. Or require RAM-only (forces requantized rows and a KL study, B2).
3. **EXL3 as a first-class quant format** in `intarsia-core` (reserved ABI id 5), importing TensorFold's MIT decoder as a pack (ADR-0002), rather than a home-made 2-3 bit format.
4. **Scope:** text only (drop the 0.97 GB vision tower), context 128K first, 1M later?
5. **Reference:** which engine defines the goldens: the vcruz vLLM recipe (one Spark, in flight as #2864), TensorFold-zig (TP2), or the official Python (does not fit any box).
6. **Fleet:** permission for a Phase 0 takedown (flight, guidance, capcom on spark1 and retro on spark2) if #2864 does not yield usable dumps; and a representative corpus for the Engram profile (Phase 0b).
7. **Priority versus paused work:** CUDA round 7 and HIP round 10 are parked until the quota resets; DSV4.1 phases 2-3 compete for the same CUDA attention.

---

## Appendix A: reproducibility

- Headers: `hf_headers.py vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw exl3.json` and `hf_headers.py deepseek-ai/DeepSeek-V4.1-Flash-1017 base.json` (HTTP range reads, 8 B then header JSON per shard; 190,552 and 96,085 tensors). `byte_budget.py exl3.json base.json` prints the table in section 0.1.
- Skew: `engram_skew.py 3000000 tokenizer.json /opt/code` (tokenizer at https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/tokenizer.json). It asserts compressed vocab = 99,092 and prints the row counts it derives; both matched the config. Corpus is shuffled with a fixed seed, split by file; results vary with corpus size.
- Hardware: `lsblk -d`, `free -g`, `nproc`, `/sys/class/nvme/nvme0/device/current_link_{speed,width}`, `/sys/block/nvme0n1/queue/logical_block_size`, `ip -br link`, `/sys/class/net/*/speed`, `ibv_devinfo` over ssh; `fleetq status|show|log`. No fio, no loads, no slot taken. spark1 figures were read while #2864 was running.

## Appendix B: sources

- Base model and official inference code: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 (files `inference/engram.py`, `inference/model.py`, `inference/config.json`, `tokenizer.json`)
- Quant under study: https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw (index and headers)
- vcruz recipe: https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe
- Engram paper: https://arxiv.org/abs/2601.07372 and https://arxiv.org/html/2601.07372v2 (sections 2.2, 2.5, 3.2, 6.3, 6.4, appendix C; the last 63K characters of the HTML were not read)
- Engram public demo (earlier variant): https://github.com/deepseek-ai/Engram
- EXL3: https://github.com/turboderp-org/exllamav3
- tonyd2wild lane tested by #2853: https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (as named in unanimis a07d3d17)
- GPUDirect Storage on DGX Spark: https://forums.developer.nvidia.com/t/gds-support-for-dgx-spark/350328
- PM9E1 4 TB listing (sibling SKU): https://esaitech.com/products/samsung-pm9e1-mzvlc4t0hbl1-00b07-4096gb-m-2-pci-express-gen-5-x4-solid-state-drive
- Internal: unanimis records a07d3d17-c405-4d82-9ab6-2175a55a7b9a, 8da5d89b-00f2-4415-b02b-1eb10ce3e0fc, 6a7efe40 (GB10 bandwidth); `docs/research/spark-tensorfold-tooling-lessons.md`; `docs/decisions/0002-*.md`, `0006-tensor-parallelism.md`; `/opt/code/TensorFold/src/tensorfold/cuda/exl3/`.
