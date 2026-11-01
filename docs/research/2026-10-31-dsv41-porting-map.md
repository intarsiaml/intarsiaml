# DeepSeek-V4.1 Flash: porting map from TensorFold (2026-10-31, lane 0 "reference")

Status: research. Written for the Phase 0 gate of `docs/plans/2026-10-31-dsv41-unpark.md`. Nothing here changes engine code.
Method: read-only reading of the patched TensorFold tree (below) and of DeepSeek's official inference code, two
sonnet explorer passes whose file:line claims were spot-checked (the ones marked "checked" were re-read by me; the rest
are +-2 lines). Line numbers are for the exact tree below.

## 1. What was read

| Item | Pin |
|---|---|
| Engine | `bertholomus/TensorFold` branch `deepseek-v41-zig`, **`fbbc97132cd004237d9887b8d3185896efaedc5c`** (branch tip of 2026-10-23) |
| Patches | MiaAI-Lab recipe `zig/patches/0001..0038` (34 files, applied in filename order by `zig/Dockerfile`), all apply cleanly |
| Kit inputs | `deepseek-v41-zig-tp4` @ `59402086…` (only `tf-dsv41-rank-cache` and three input scripts), recipe `zig/patches-inputs/0001,0002` |
| Official code | `deepseek-ai/DeepSeek-V4.1-Flash-1017` `inference/{model,kernel,engram}.py` (MIT), copy in `.worktrees/dsv41-official` |
| Weights | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw @45ee5d0f` |

**The pin the task named (`629c524a`) is NOT what the working build uses.** `scripts/config.sh` of the recipe says
`TF_REF=fbbc971…` ("the branch tip of 2026-10-23, was 629c524a"); `629c524a` is an ancestor, 24 commits older, and the
NOTICE file still names it. The patches apply to `fbbc971`. The patched tree is reproducible with:
`git clone https://github.com/bertholomus/TensorFold.git tf && cd tf && git checkout fbbc971… && for p in <recipe>/zig/patches/*.patch; do git apply $p; done`
(the copy used here: `.worktrees/dsv41-ref0/.scratch/tf`, not committed). Build base: `nvcr.io/nvidia/pytorch:26.07-py3`,
Zig 0.17.0, `-Dsm=121 -Doptimize=ReleaseFast`.

## 2. Headline findings

1. **Almost none of the model math is hand-written CUDA in the tree.** The DSv4.1 family launches
   (b1) **Triton AOT kernels** from `tri_*.zig` wrappers, (b2) **prebuilt CUDA cubins** (`linear_grouped.cubin`,
   `linear.cubin`, `experts.cubin`, `experts_cb.cubin`: the Python engine's EXL3 extensions) and (c) cuBLAS. The Triton
   and cubin binaries come from an external "kernel kit" (`bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10`, `kit/`,
   Python family branch `deepseek-v41-tp2`: `kernels.py`, `rounds.py`, `exl3/{linear,experts,prefill}.py`, `markov.py`)
   that is **not in the tree**; the Zig side only knows kernel names, grids and constants (recorded in
   `zig/src/families/dsv41/fixtures/launches.json`, `tools/dsv41-zig/kit/{kit_variants,aot_extra}.py`). The in-tree CUDA
   is small: `dsv41_torch.cu` (139 lines, torch-exact rms-norm / compress / rope), `dsv41_ops.cu` (513, top-k, block
   helpers, add/cast), `dsv41_shard.cu` (356, KV shard fetch), `dsv41_port.cu` (130, byte mover), `dsv41/rdma_gather.cu`
   (228, RDMA staging/collect), `torch_ops/*`.
2. `experts*.cu`, `qmm_*.cu`, `lane_gemv.cu`, `prefill_attention.cu` in `zig/kernels/cuda/` belong to the **Nemotron**
   family, not DSv4.1. `fmha.zig` is the vision tower's attention. `exact.zig` is torch-rounding helpers, not sampling.
3. So "port the TensorFold kernels" means: **keep the Zig host logic and the launch contracts, rewrite or re-source the
   kernels**. For the EXL3 kernels the open sources are ExLlamaV3 (MIT) and TensorFold's own MIT
   `src/tensorfold/cuda/exl3/{decode.cuh,linear.cu,experts.cu,experts_grouped.cuh,format.py}` (checked at
   `/opt/code/TensorFold`; EXL3 GEMV/GEMM already landed on main in #55/#56, only the grouped MoE is missing). For the
   Triton ones the Python family source (Apache-2.0, branch `deepseek-v41-tp2`) is the reference to fetch; its repo is
   not cloned here (UNVERIFIED that it is public).
4. **TensorFold is not bit-identical to the official model code**; a port should target the official math (fixtures in
   `golden/dsv41/`) and use TensorFold only for engineering structure and as the runtime reference for goldens. The
   differences found are listed in section 5.
5. TensorFold's server **refuses `logprobs`** (see `docs/research/2026-10-31-dsv41-golden-plan.md`).

## 3. Component map

Paths: `D` = `zig/src/families/dsv41/`, `K` = `zig/kernels/cuda/`. Official refs are `model.py` (`mp`), `kernel.py` (`kp`),
`engram.py` (`ep`) in `.worktrees/dsv41-official`. Verdict legend: **AS-IS** port the logic/code unchanged (licence
notice travels), **ADAPT** keep structure, change interfaces/numerics, **REWRITE** use as a spec only.

| # | Component | TensorFold host (Zig) | Kernel (type) | Official | Verdict |
|---|---|---|---|---|---|
| 1a | EXL3 linear, decode (1-128 rows) | `D/exl3_linear.zig:23-221` (plan/split-K `:23-41`, `glinear[k2][codebook][warps]` `:177-221`); callers `D/prompt.zig:602-675`, `D/round.zig:806-1076` | `rot_many_kernel` + `glinear_kernel<K2,cb,warps>` in `linear_grouped.cubin` (b2) | `mp:210-278` replaced; FP8/FP4 GEMMs `kp:208-562` | **ADAPT**: reuse `plan()`; kernel = our M2 GEMV (already on main) |
| 1b | EXL3 prefill (over 128 rows) | `D/exl3_prefill.zig:65-109`; `D/prompt.zig:874-900` | `rot_in`/`unpack<K2,cb>` from `linear.cubin` (b2) + Triton `_gemm` (`D/tri_markov.zig:152-246`), explicit bf16 +-1 Hadamard (`:110`) | same | **ADAPT**: our M2 prefill GEMM exists |
| 1c | Routed + shared experts, grouped MoE | `D/exl3_experts.zig:52-124` (kernel table), `:192-269` prompt path (8 launches/layer: `group_count, group_place, work_list, rot_in, grouped_mma, gateup_epilogue, grouped_rows, down_combine`, used from 64 rows), `:355-462` decode (3 launches: `decode_prep`, `grouped_cp` gate/up + epilogue, `grouped_cp` down + combine, PDL); `D/round.zig:1277-1305`; `D/prompt.zig:1350-1397`; weights pack `D/weights.zig:351-455` | `experts.cubin`, `experts_cb.cubin` (b2); K-range variants per layer `range(lo,hi)` (8,8) (2..10) (2..16) (`:28-32`) | `mp:832-906` | **ADAPT**: launch plan reusable, kernel = exllamav3/TensorFold `experts_grouped.cuh` (MIT). Shared expert is expert index `count-1` (last slot) |
| 2 | Router (Gate) | `D/tri_norm.zig:119-176` (`rowmmGate`), `:192-205` (`route`); `D/model.zig:415,1069` (bias_vl load); image spans `D/prompt.zig:1379-1397` | Triton `_rowmm_parts/_rowmm2`, `_route/_route_parts` (b1); cuBLAS fp32 Sgemm over 16 rows (`D/cublas.zig:78`, `D/prompt.zig:1363-1366`) | `mp:794-829` | **REWRITE** (tiny; fixtures pin it). Gate weight is **fp16** in the pack, bias fp32 |
| 3 | Hyper-connections + Sinkhorn | `D/tri_hc.zig:86-173`; `D/tri_basic.zig:19-33`; `D/round.zig:1537-1620` (`stretch`), `:1628` (`mix`); `D/prompt.zig:861-872,1341` | Triton `_hc_mix_part`, `_hc_mix_pf`, `_hc_post_only`, `_hc_sinkhorn`, `_hc_finish[_s/_u]`, `_hc_post`, `_collapse[_norm]` (b1) | `mp:950-968`, `kp:407-474` | **ADAPT**: fusion/gather structure is the value; Sinkhorn is 30 lines. TF uses split-K partial dots (`_hc_mix_part`), official one fp32 `F.linear` |
| 4 | Norms (eps 1e-20) | `D/tri_norm.zig:44-74` (`qKvNorm, rmsnormRot, kvNormRope`), `D/tri_basic.zig:34`; torch-order variant `D/exact.zig:57` | Triton `_q_kv_norm, _rmsnorm_rot, _kv_norm_rope, _rmsnorm` (b1); CUDA `K/dsv41_torch.cu:17-65` for compressor/indexer norms (a) | `mp:281-293` | **ADAPT**: fused q/kv-norm+hadamard+rope+window-fp8-quant kernels are the fast path; start unfused |
| 5 | MQA attention hd512, sink, window 128 | `D/tri_attn.zig:11-132`; `D/round.zig:806-1076`; `D/prompt.zig:913-1060` | Triton `_sparse_attn_part` (8-way key split at <=16 rows), `_sparse_attn_merge`, `_sparse_attn_merge_rot[2]` (merge folds sink, inverse RoPE, wo_a's input rotation); window = 128-slot ring, `WIN=128, BN=32, HB=16`, scale `hd^-0.5` | `mp:766-791`, `kp:310-403` | **REWRITE** against `kp:311-389` (online softmax, blocks of 64, p cast to q dtype, running max starts -1e30, sink added to the denominator); reuse the packed-fp4 gather idea |
| 6 | Compressor (ratios 0/1/2) + KV quant | `D/round.zig:906-945`; `D/prompt.zig:1149-1210`; positional stores `raw_kv/raw_score` instead of rolling state (`D/round.zig:918-924`; incomplete groups write a scratch row `:708`); prefill needs `start % r == 0` else `notPorted` (`D/prompt.zig:1149-1160`) | `K/dsv41_torch.cu:67-92` `compress2` (a, torch-order softmax+sum into bf16); Triton `_fp4_store` (`D/tri_attn.zig:135-137`; compressed KV block 16 E4M3 scale, floor `6*2^-9`); window fp8 inside `_kv_norm_rope` (`QUANT`) | `mp:429-485`, `:701-721`, `:740-764`, `kp:127-204` | **ADAPT**: positional stores are cleaner than the rolling state; our chunked fixtures define semantics for any `start` |
| 7 | Indexer + top-k + candidates | `D/round.zig:947-1010`; `D/prompt.zig:1222-1321`; `D/tri_index.zig:67-165`; keys fp4 block 32 E8M0 | Triton `_index_score[_rows/_cand]`, `_topk_sel`, `_topk_finish`, `_score_keys`, `_gather_tiles`; CUDA `K/dsv41_ops.cu` (`topk_i64`, `topk_indices` `:127`, `candFast` `:266`, `poolPick` `:278`, `blockMax` `:299`) (a); query runs `_fp4qd_p2` quant-dequant (`D/tri_attn.zig:140-143`, `D/round.zig:956`, `D/prompt.zig:1236`, **checked**) | `mp:488-611` | **REWRITE** score/top-k kernels; **ADAPT** two-level candidate logic (`candFast` needs `nb % cb == 0` else `notPorted`, `D/round.zig:995`). Indexer is replicated across ranks (no score all-reduce) |
| 7b | C7 indexer split (patch 0001) | `D/prompt.zig:905-908,1264-1322`; env `D/model.zig:173-175`, `TF_DS_IDX_SPLIT` | row-block split + all-gather, default on at >=131,072 tokens | n/a | **SKIP** at first (latency only, ordering effect only) |
| 8 | Grouped output projection | `D/plan.zig:166`, `D/weights.zig:479` (8 per-group EXL3 `wo_a.slice.N`); decode feeds the merge's rotated `xh` straight in (`D/round.zig:1011-1060`); prefill one `mm` per group (`D/prompt.zig:1010-1040`) | EXL3 linears + `_sparse_attn_merge_rot2` | `mp:784-791` | **ADAPT**. **No fp8 fake-quant of `o` before `wo_a`** anywhere in TF (grep + recorded constexprs); the official code has it (`mp:785-787`), the fixtures default to OFF to match |
| 9 | RoPE / YaRN | tables from `tools/dsv41-zig/kit/make_rope.py:1-40` (CPU, via the Python engine's `ops.rope_cs`, sha256-checked); `D/round.zig:799-802` (`ropeOf`: compressed iff ratio != 0); inverse RoPE on `o` at prefill `D/prompt.zig:1011`, folded into the merge at decode | Triton `_rope_heads`; torch-exact `K/dsv41_torch.cu:94-113` (FMA-contracted complex multiply) for compressed latents | `mp:369-405` | **REWRITE** from `mp:369-389` (theta 10000 plain, 160000 + YaRN x16/65536 on compressed layers); fixtures give cos/sin at far positions |
| 10 | SwiGLU clamp (limit 10), shared expert | inside the experts cubin epilogue, fp32 (`D/exl3_experts.zig:355-462`) | b2 | `mp:843-853` | ADAPT with 1c. Note TF applies the routing weight in fp32 after `down` (`down_combine`); the official code multiplies before `w2` and casts to bf16 |
| 11 | Embedding, final norm, head, sampling | `D/tri_basic.zig:39,49` (`_embed_init` copies into 4 hc streams with pre=(1,0,0,0)); `D/weights.zig:540-543` (embed bf16 replicated); head = EXL3 6-bit, vocab halves, fp32 logits all-gathered `D/prompt.zig:1729-1745`; sampling `D/sampling.zig:29-120` (argmax at T=0; else top-k + margin 8, top-p/min-p in fp64, Gumbel keyed by splitmix64(seed, position, token id)) | Triton + EXL3 | `mp:152-178, 999-1019, 1288-1295` | **AS-IS** for the keyed sampler (makes verify rows reproducible); head ADAPT |
| 12 | Layer loop / rounds | `D/round.zig` (decode round, `stretch` `:1540`, `attention` `:806`, `moe` `:1277`, `mix` `:1628`), `D/prompt.zig` (prefill chunks, `attention` `:913`, `kvSourceUpdate` `:1149`, `indexer` `:1222`, `moe` `:1350`), `D/model.zig` (one rank: weights, arena, ropes, pool), `D/lanes.zig` (multi-stream lane), `D/native.zig` (entry) | | `mp:1245-1275` | structure reference only |

### 3.1 Engram

| # | Piece | TensorFold | Official | Notes / verdict |
|---|---|---|---|---|
| E1 | Hash (token map, multipliers, primes, rolling XOR, rows) | `D/engram.zig:66-131` (`Hasher`, `hashes()` `:103-126`), primes `:48-63` (Miller-Rabin), offsets `:78-90`, row = `@mod(rolling, prime) + offset` wrapping i64 `:116-121`; **host side, both ranks hash everything**, prefill `D/prompt.zig:1522/1563`, decode `D/round.zig:1653-1678`; multipliers come from `<kit>/engram.json` made by `tools/dsv41-zig/rec/zrec_fixtures.py:105-120` (numpy `default_rng(10007*layer)`); token map from `dsv41_token_map.json` (`D/model.zig:383-384`); image tokens negative = DEAD (`:112`) | `ep:9-14, 17-61, 64-83, 160-184` | **ADAPT**: algorithm identical. **Verified here: the official `build_compressed_token_map` run on Mia's `tokenizer.json` is bit-identical to MiaAI's `dsv41_token_map.json`** (129,280 ids -> 99,092; fixture `token_map` in `layers_*.safetensors`). Draft tokens are hashed like any token in the verify window |
| E2 | Table layout and fetch | the **official safetensors shards 47/48 as downloaded** (`D/engram_io.zig:20-79, 96-108`): `layers.{1,14}.engram.embed.weight` F8_E4M3 `[384,006,168 / 384,016,682, 256]`, `embed.scale` F8_E8M0 `[rows, 8]`, separate regions (weights 98.3 GB then scales 3.07 GB per layer; 2 reads per row, 4 KiB-aligned O_DIRECT spans into 8 KiB bounce slots `D/engram_aio.zig:12-14`). **Not row-sharded**: both nodes hold the full ~203 GB; TP2 splits the 24 hash *columns* (rank0 0-11, rank1 12-23) (`D/prompt.zig:1446-1451`, `D/plan.zig:34-36,193`) | `mp:296-325` shards rows over ranks + all_reduce | Default path: 64 `pread` threads via page cache (`engram_io.zig:140-290`); `TF_DS_ENGRAM_AIO=1`: raw `io_setup/io_submit/io_getevents` on O_DIRECT, polled reap (0.4-0.8 ms for a 6-row window claimed) (`engram_aio.zig:1-130, 225-255`); rings: decode 1024 slots x ceil(rows/16) x 8 KiB, prompt 4096 slots (`D/model.zig:397-406`); prefetch: decode `engramAhead` at round start, `engramLand` before the layer (`D/round.zig:1506-1530, 1683-1706`); prompt: next chunk read while GPU runs earlier layers, optional lead reader per Engram layer (`D/engram_lead.zig`, `TF_DS_ENGRAM_LEAD=1`). **No hot tier**: relies on the OS page cache or none. Failure paths (patch 0035) end the pair with exit 75 (`D/prompt.zig:1533-1546`). The pre-packed files `spark1:~/dsv41-engram/engram-l{1,14}-r0of2.bin` (Oct 17, root-owned; spark2 holds only `engram-l14-r1of2.bin` there) are **not** what the Mia recipe reads (it reads the official shards; spark2's `~/.cache/dsv41-tensorfold/engram-1bae0e10` is a copy of shards 47/48, same sizes as spark1's HF cache). Their layout, probed read-only: 4096-byte header (magic `DSV4ENG1`, version 1, row count 192,001,740 as LE u32 at byte 24), then rows of 264 B = 256 B FP8 payload + 8 B E8M0 scales **interleaved**, row `r` at `4096 + 264*r` (size 50,688,463,456 = 4096 + 192,001,740*264 exactly); payload and scales are byte-identical to the official shard rows (checked rows 0, 1, 2, 5000 of layer 1). `r0of2` = rank 0's first 192,001,740 rows. This interleaved 264 B row is the better NVMe read unit (one read per row instead of two) and matches the scope doc's A2 layout; origin of the packer unknown (UNVERIFIED which revision/tool wrote it). |
| E3 | Merge (wkv, gate, add into 4 streams) | rows decoded on host: FP8 x 2^(scale-127) exact fp32 -> RNE bf16 (`D/engram.zig:135-170`, `e4m3 :185`, `bf16 :198`); `wkv` = EXL3 linear **split-K by hash column** (3072 in x 25,600 out per rank), fp32 partials summed in rank order and rounded to bf16 (`D/prompt.zig:135-138`, `D/ops.zig:106`, `K/dsv41_ops.cu:16`); gate = Triton `_engram_gate` (source absent, `BLOCK=1024, D=5120`, `D/tri_basic.zig:59-61`, called `D/prompt.zig:1616`, `D/round.zig:1573`), `engram_qk = q_weight*k_weight` precomputed fp32 (`D/weights.zig:83,528`) | `mp:328-365` (gate: fp32 rstd per hc copy, `copysign(sqrt(clamp_min(|dot|, 1e-6)), dot)`, sigmoid) | ADAPT; hardcodes 2 ranks (`notPorted` otherwise). Single-rank port can do the full matmul and skip the exchange. Gate kernel must be checked against `mp:350-365` (fixtures) |

### 3.2 DSpark / MTP draft

| Piece | TensorFold | Official | Verdict |
|---|---|---|---|
| Config: block 5, taps 37-39 (mean over hc after the attention mix) | `D/config.zig:262-269, 294-295`, `D/round.zig:1580-1584` | `mp:1103-1159, 1268-1269` | |
| Absorb (`forward_embed` + window seeding) | `D/draft.zig:362-414` | `mp:1131-1138, 1035-1054` | ADAPT |
| Draft pass (`forward_spec`) | `D/draft.zig:438-542`; up to 16 streams x 5 rows; 128 experts top-3; per stage fp32 gathers | `mp:1278-1285` | REWRITE (tied to TF's rings, Triton kits) |
| Markov head | `D/tri_markov.zig:43-141`; bias rows `[V,256]` (embed bf16, head fp16), 256 cached rows chosen from `markov_tokens.zig:6` (a token list from ExLlamaV3 frequencies); **draft is argmax**, official `sample()` is Gumbel | `mp:1080-1090, 1140-1159` | ADAPT |
| Confidence | fp32 GEMV over `[xc, markov_embed]` (5120+256) `D/draft.zig:530-541` | `mp:1092-1100` | AS-IS |
| Draft count | `D/draft.zig:642-685` (`[5,5,3,3]` by live streams; hard-coded `round_ms` table, `draft_ms` 3.5) | | policy only |
| Accept | `zig/src/core/lanes/accept.zig:40-60`, `engine.zig:406-420`: follow the first child equal to the target's sampled pick, no rejection sampling | | **AS-IS** (simple, exact w.r.t. the sampler) |
| Draft-token Engram | none separate: verify window hashes drafts like tokens; DSpark stages have no Engram (`D/plan.zig:192`) | same | |

### 3.3 TP=2 sharding and collectives (TensorFold)

Heads 64 -> 32/rank (`wq_b` columns, `wo_a` groups 8 -> 4, `wo_b` input rows, sinks); **experts are split by intermediate
width (2304 -> 1152 per rank), all 384 experts present on both ranks** (official: by expert id); head by vocab half;
Engram `wkv` by hash column; replicated: embed, norms, `wq_a`, `wkv`, compressor, the whole indexer, gate, hc params
(`D/plan.zig:1-202`). Collectives: per layer 2 fp32 all-gathers (attention partial `[R,5120]`, MoE partial `[R,5120]`,
`D/round.zig:1589,1599`); the sum happens in the next `hc_pre`/`hc_post` in rank order (no all-reduce); +1 per Engram
layer (`[R,25600]` fp32, `:1569`), +1 head gather (`[R,64640]` fp32, `:1618`): **83 collectives per target round**
(+6 for the draft pass). Transport: RDMA-write ring through pinned host staging with a proxy thread (`D/rdma.zig:1-8`,
`K/dsv41/rdma_gather.cu`) when `count*4 <= 4,456,448` B, else NCCL allGather (`D/comm.zig:113`); NCCL id over TCP
(`D/link.zig`, `TF_TP_MASTER`, port 29620). `D/rank_wire.zig` (2820 lines) is the spill/KV-shard/startup-fingerprint
protocol, not model data. Patches 0015/0017/0018/0022 (KV shard, row split, lanes, ring) and 0013/0014/0016/0019/0021
(spill tiers) are memory/latency optimisations: **skip for a first port**. Verdict: **REWRITE** the layout for our
own TP (our ADR-0006 NCCL path), keep the 83-collective count as the thing to beat.

### 3.4 Memory plan facts

Engine reserve `max(4 GiB, MemTotal/10)` (~12.2 GiB on a GB10, `zig/src/native/cuda_memory.zig:87-92`); window ring
`[128+16, 512]` bf16 per layer per sequence (`D/prompt.zig:46-47,149-151`); compressed KV fp4 codes `[cap/ratio+2, 256] B`
+ E8M0 scales `[rows, 32] B`, index K codes `[rows, 64]` + scales `[rows, 4]` (`D/prompt.zig:671-723`); compressor
positional store fp32 `raw_kv/raw_score` `[slots x raw_rows, 512]`. Only ratios 0, 1, 2 occur in this config.

## 4. Kernel triage

| Port as-is (logic or code) | Port with changes | Rewrite (spec only) |
|---|---|---|
| keyed sampler (`sampling.zig`); accept rule (`core/lanes/accept.zig`); engram hash (`engram.zig`) + FP8 row decode; Engram failure paths; reserve rule; confidence head | EXL3 dispatch plans (`exl3_linear.zig plan()`, `exl3_experts.zig`); hc fusion/gather structure; q/kv-norm+rope+quant fusion; compressor positional stores; Engram io (`engram_io/aio/lead`, add a hot tier and drop the 2-reads-per-row layout); draft absorb/markov | router; Sinkhorn; sparse attention (window + compressed, sink, merge); indexer score/top-k; YaRN tables; TP layout/collectives; Engram gate (source missing); draft pass |

## 5. Numerics: where TensorFold differs from the official code

1. **Expert routing weight**: official `x = weights * x` (fp32) then `.to(bf16)` before `w2` (`mp:850-853`); TF applies the
   weight after `down`, in fp32 (`down_combine`). Summation order: TF slot order with the shared expert last; official
   expert-id order then shared.
2. **hc mixes**: TF split-K partial dots (`_hc_mix_part`); official one fp32 `F.linear` (`mp:954-957`). Last-bit differences.
3. **Engram `wkv`**: TF sums two fp32 split-K partials and rounds to bf16; official: bf16 rows all-reduced, one matmul.
4. **Attention probabilities**: official kernel casts p to bf16 before P@V and starts the running max at -1e30 (`kp:339-386`);
   TF's Triton `_sparse_attn_*` behaviour is not visible (UNVERIFIED), it splits keys 8-way at <=16 rows.
5. **Fake quantisation sites** of the FP8 checkpoint: window KV fp8 (block 32, E8M0) - TF folds into `_kv_norm_rope`;
   compressed KV fp4 (block 16, E4M3 scale) - TF `_fp4_store`, lossless vs official; index key fp4 block 32 E8M0 - TF;
   **indexer query fp4 - TF runs `_fp4qd_p2`** (checked); **`o` before `wo_a` - absent in TF**.
6. **Compressor/indexer norms and RoPE**: TF uses torch-exact CUDA (reduce order, FMA contraction) for these to match the
   Python family bit for bit; the official GPU path is torch itself.
7. **Gate precision**: TF gate weights fp16 (as in the pack); the official code upcasts `weight.float()` (`mp:812`).
8. **Draft**: TF argmax; official `sample()` Gumbel.
9. EXL3 detail: the codebook is hard-coded to `mul1` in `exl3_linear.zig:12`: `.mul1` marker tensors are never read.
10. The pack is all-EXL3 for linears (attention, shared experts, compressor, indexer `wq_b/wk`, `wo_a` as 8 slices,
    `main_proj`, Engram `wkv`, head); plain tensors: embed, norms, gate `weight` (F16) / `bias` / `bias_vl` (F32),
    `indexer.weights_proj` (F16), `hc_*` (F32), `attn_sink` (F32), Engram `q_weight/k_weight` (F32). The "mixed pack"
    patch 0024 (BF16 indexer/compressor linears) is a no-op on Mia's EAQ v1 pack; 0025 widens `weights_proj` to fp32.
11. **`bias_vl`**: TF reads `kit/vision/gate_bias_vl.safetensors` (`D/model.zig:415,1069`) because its builder assumed the
    EXL3 pack lacks it; Mia's pack **does** contain `layers.N.ffn.gate.bias_vl` (40 + 3 MTP), read it from the shards.

## 6. Licence and attribution that must travel with ported code

The engine and the DSv4.1 family are Apache-2.0. Ported code carries, in file headers and `LICENSES/`:

* **TensorFold** - Copyright 2026 TensorFold contributors, Apache-2.0 (https://github.com/ashhart/TensorFold; 0.5.0 and
  earlier MIT, kept in `LICENSES/MIT.txt` upstream). NOTICE text: "TensorFold / Copyright 2026 TensorFold contributors /
  https://github.com/ashhart/TensorFold".
* **DSv4.1 family on TensorFold's Zig engine** - BertholomusAI (Albert Lee, `@bertholomus`), Copyright 2026, Apache-2.0
  (https://github.com/bertholomus/TensorFold, branch `deepseek-v41-zig`; `tools/dsv41-zig/ATTRIBUTION.md`: written
  clean-room from DeepSeek's MIT code, "no code copied" from DeepSeek).
* **DSv4.1 kernel kit** (if any Triton/cubin source is used) - BertholomusAI, Apache-2.0; parts keep their own licences:
  Triton MIT, PyTorch attention kernel BSD-3-Clause on NVIDIA CUTLASS BSD-3-Clause, ExLlamaV3 MIT
  (turboderp, Copyright (c) 2025).
* **MiaAI-Lab recipe patches** - Copyright 2026 MiaAI-Lab, Apache-2.0; authors per patch (recipe `NOTICE`):
  0005 Coinupbtc (PR #5), 0011-0022 Robert Wojciechowski (wojo, `@rwojo`; GLM-5.3-Flash recipe patches 0088/0110/0111/0118/0119,
  spill-v2 block layout after BadAd84), 0023 Robert Wojciechowski, 0024-0026 Mykhailo Tamarin (PR #57), 0001/0002/0004/0006
  written for the recipe; `scripts/` adapted from the GLM recipe (webzone, Alexbob0, john, Eduardo Florencio, Steve Darlow).
* **DeepSeek** - `model.py`, `engram.py`, `kernel.py`, weights: MIT, Copyright (c) 2023 DeepSeek. `chat_template.jinja`:
  Mia's AI Lab port, one change by hashd1ve. Mia EXL3 pack: MIT, Mia's AI Lab.
* **ExLlamaV3 / EXL3 format** - MIT, Copyright (c) 2025 Turboderp. TensorFold's numpy decoder `format.py` (used by
  `tools/dsv41/fixtures.py`) is under TensorFold's MIT-era notice.

Practical rule for the port: every file derived from a TensorFold/patch file starts with an SPDX line plus the
upstream copyright line(s) above; `LICENSES/` gets `TensorFold-Apache-2.0.txt`, `DeepSeek-MIT.txt`, `ExLlamaV3-MIT.txt`,
and a `NOTICE` section listing the patch authors.

## 7. Open points the fixtures answer or that remain

* Whether `Gate`, `hc`, `Engram` gate and the Triton attention reproduce the official numerics can only be settled by
  running TensorFold; the fixtures in `golden/dsv41/` + the goldens (token-level today) are the instruments.
* UNVERIFIED: that the Python family repo `deepseek-v41-tp2` (Triton sources) is public; not cloned here.
* UNVERIFIED: GPU behaviour of the CPU emulations in `tools/dsv41/fixtures.py` (`CpuKernels`).

## 8. Fixtures (tools/dsv41/fixtures.py, golden/dsv41/)

Generator: DeepSeek's official `model.py`/`engram.py` on CPU, `kernel.py` replaced by pure-torch `CpuKernels` (act_quant,
fp4_act_quant, sparse_attn, hc_split_sinkhorn; **not bit-checked against the GPU kernels**, none available), real Mia
EXL3 2.9 bpw weights decoded with TensorFold's numpy decoder (float64, rotate - W_q - rotate in 16-row blocks),
Engram rows read by `pread` from official shards 47/48. Run on spark1 in fleetq slots (`--load cpu --mem 30-34`);
generator inputs/outputs live in `spark1:~/dsv41-fixtures/{in,out}`; `out/manifest.json` (sha256, sizes) is copied to
`golden/dsv41/manifest.json`. Quantisation flavour `--quant tf` (default): the fp8 fake-quant of `o` before `wo_a` is OFF
(neither EXL3 nor TensorFold has it); `nofake` also drops the indexer-query fp4; `official` keeps all. Outputs in
`out/nofake/` use `nofake`.

| File (spark1 `~/dsv41-fixtures/out/`) | Size | Committed? |
|---|---|---|
| `layers_bf16.safetensors`, `layers_fp32.safetensors`: layers 0-2, prefill of 16 tokens + 2 single-token decode steps, taps at every sublayer boundary (hc mixes/pre/post/comb incl. Sinkhorn, norms, attention q/kv/latent/idxs/o_sparse/o_lora, indexer k/score, router weights+indices, routed/shared/combined MoE, Engram hash ids/rows/wkv/gate, `out.h`), plus `ids`, `token_map` (129,280), Engram primes/offsets/multipliers | 27.7 MB / 51.7 MB | subset `golden/dsv41/layers_{bf16,fp32}_ci.safetensors` (9.5 / 5.3 MB) |
| `router_gate_L2.safetensors`: gate weight/bias/bias_vl + input + image_mask (rows 4-10 use `bias_vl`) + outputs | 4.1 MB | no (spark1) |
| `attn4k_L2_bf16_full.safetensors` (+ `_report.json`): layer-2 attention on a 4,095-token prompt, whole vs chunked (1024/512/1000/777) vs official `Attention.forward`, caches | 128.6 MB | report only |
| `sparse_attn_4k_bf16.safetensors`: ten rows of sparse attention (q, gathered kv, idxs, sink, o) incl. window/compressed positions | 3.1 MB | no |
| `attn1k/*_fp32*`: same, fp32 flow, 1,023 tokens | 64.5 + 3.7 MB | report only |
| `nofake/*`: the `layers`/router fixtures with the indexer-query fp4 off | 83.6 MB | no |
| `golden/dsv41/ops.safetensors` (synthetic, any machine): fp8/fp4 fake-quant, Sinkhorn, hc_pre/post, candidate blocks, sparse_attn with -1 and an all-masked row, RoPE/YaRN cos/sin out to position 1,048,575, inverse RoPE | 1.6 MB | yes |

Sanity checks (all asserted by the generator or run on the outputs):

* Sublayer-tapped block == official `Block.forward`: layers 0/1/2 bf16 rel 1.0e-4 / 0 / 1.0e-3, fp32 rel 2.6e-8 / 3.9e-9 / 3.2e-8.
* Hidden-state norms after layers 0/1/2 (hc streams of 16 tokens, `out.h`): bf16 192.4 / 1323.5 / 1457.1, fp32 192.4 / 1321.6 / 1455.4,
  abs max 9.3 / 60.5 / 60.1; no NaN/Inf. bf16 vs fp32 relative difference after layer 0: 4.2e-3.
* Router (layer 2, 16 tokens): top-6 expert sets agree with an independent float64 recomputation in 16/16 rows including
  the `bias_vl` rows 4-10 (7 rows pick different experts than with `bias`), weights agree to 4e-7.
* Engram token map from the official function on Mia's `tokenizer.json` == MiaAI's `dsv41_token_map.json` (bit-identical).
* Attention (layer 2, ratio 2, 4,095 tokens, bf16): caches (window ring, compressed KV, index K, compressor state)
  bit-identical between the official code, my chunk-general code and chunk schedules {1024, 512, 1000, 777}; outputs
  bit-identical for all chunk sizes in bf16 (fp32: <= 3e-8 relative; BLAS shape noise). Official prefill(N) vs prefill(N-3) + 3 decode steps:
  caches bit-identical, last-3 outputs differ rel 6.9e-3 in bf16 (matmul batch shape).
* The official `topk(sorted=False)` differs from a stable sort in 1,013/4,095 rows, **all of them only equal-score swaps**
  (bf16 scores tie often): the port must define tie-breaking explicitly; the fixture's selection uses lowest-index-first.
* Official Compressor/`_window_kv`/Indexer/`select_candidate_blocks` only support prefill from position 0 and 1-token decode;
  `attn_chunk` generalises them. Candidate blocks (layer 20+) are covered by `ops.safetensors` only, not by a layer run
  (layers 0-2 do not reach them).

Not covered: the EXL3 kernels' own numerics (weights decoded in float64, so fixtures are the exact-math target, GPU EXL3
kernels will differ by fp16/bf16 rounding), layers >= 3, DSpark/MTP blocks, 16K+ positions (candidate blocks, top-512
engages past about 16K compressed positions; `ops` covers the selector only).
