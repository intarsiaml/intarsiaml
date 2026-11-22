# Inkling-2-Mini port spec (lane ink-spec, 2026-11-21)

Model: HF `thinkingmachines/Inkling-2-Mini` (snapshot `b85c3cfa`, weights sha256 `65e60f51…`), `model_type: inkling2_loop`, text model only.
Numerics authority: vLLM `0.1.dev25855+g7cb5dae22` from `vllm/vllm-openai:inkling-2-mini`, extracted to `.worktrees/ink-official/`
(paths below are relative to it; `V/` = `models/inkling2/nvidia/`). Oracle: `tools/inkling2_ref.py` (section i). Comments `aN` in the
script refer to the rows of section a.

Shape constants: D=2304 hidden, 24 physical layers, 48 layer visits, 24 q heads / 8 kv heads (G=3) / head_dim 128 on every layer, d_rel 16,
window 1024 = rel_extent 1024, conv K=4, 128 experts top-8, expert inter I=1152, vocab 201024 rows (200058 real).
**Full-attention layers: 5, 11, 17, 23** (`local_layer_ids` = all others; checked against the tensors: `k_input_norm`, `static_w`,
`(3,D)` `attn_queries` exist exactly there). Loop: core = layers 6..17, 3 passes; MXFP8 experts: layers 20-23 (coda only).

## a. Formulas (text forward)

Notation: `rms(x, w) = x * rsqrt(mean(x^2) + eps) * w` over the last dim. Two separate eps (both 1e-6 in this checkpoint, keep them separate op attrs): the per-source RMS inside the depth mix is a **fixed 1e-6** (`ops/attn_res.py:252`), every RMSNorm uses `rms_norm_eps`. `bf(x)` = round to bf16 (where vLLM
stores bf16; the oracle's `--rounding vllm`). Linear `y = x W^T`, W `[out, in]` as stored. "Visit" = one (layer, pass) execution.

| # | Step | Formula | vLLM source |
|---|---|---|---|
| a1 | Embed | `x0 = bf(rms(embed[id], embed_norm))` (fp32 math). This is block source 0. | `model.py` `embed_input_ids` |
| a2 | Depth mix `attn_res(S, last, Q, wn)` | Sources `x_n` = stored blocks `S` then `last` (= partial, or partial + addend). Queries `Q` `[nq, D]` = **raw checkpoint value × D^-0.5** (scaled at load). `r_n = rsqrt(mean(x_n^2)+1e-6)`; `logit_{q,n} = (Q_q·x_n)·r_n`; `p = softmax_n`; `mix_q = Σ_n p_{q,n} · x_n · r_n` (values normalized, `attn_res_normalize_values`); then if `wn` given `bf(rms(mix_q, wn[q]))` else fp32 `mix_q`. | `ops/attn_res.py:244-306`, kernel 85-167 |
| a2' | Which queries | `attn_queries.{i-1}` serves layer i (none for layer 0): `[1,D]` SWA, `[3,D]` full (rows q, k, v). `mlp_queries.{i}` `[1,D]`. `output_query` `[1,D]`. Loop: `middle_cycle_cross_pass_queries[24,D]` viewed `[12,2,D]` → (attn=q stream, mlp); `…_kv_queries` → (k, v). | `model.py:109-133`, `loop.py:167-176` |
| a3 | Attention inputs | Layer 0: `x_j = bf(rms(x0, norm_j))`. Else `m = attn_res(blocks, partial, attn_q[i])`, `x_j = bf(rms(m_j, norm_j))`. SWA: one input, `norm = attn_norm`, `x_q = x_k = x_v`. Full: three inputs with `q_input_norm`, `k_input_norm`, `v_input_norm`. Loop passes ≥1: see a12 (SWA layers then get three different inputs, all normed with `attn_norm`). | `model.py:318-325`, `loop.py:225-251` |
| a4 | Projections | From `x_q`: `q = wq_du` `[3072]`, `r = wr_du` `[384 = 24×16]`, `gate = gate_attn` `[3072]`, `sink = attn_sink_proj` `[24]`; from `x_k`: `k = wk_dv` `[1024]`; from `x_v`: `v = wv_dv` `[1024]`. `_du`/`_dv`/`_ud` only name the factor (d=model dim, u=q-head space, v=kv space); every tensor is a plain `[out, in]` linear. vLLM fuses them as rows `[q, r, gate, sink, k, v]` (`_shards`). All outputs bf16. | `attention.py:217-239, 330-348` |
| a5 | Head layout | q `[24,128]` head-major, r `[24,16]`, k/v `[8,128]`; q head h uses kv head `h // 3`. | |
| a6 | Short convs (K=4, causal, zero before position 0) | **Static** (`k_sconv`,`v_sconv` on SWA layers; `attn_sconv`, `mlp_sconv` on all): weight `[C,1,4]`, tap 3 = current token: `y_t = (Σ_{k=0..3} w[c,k]·x_{t-3+k} + x_t) · 5^-0.5`. **Dynamic** (k/v on full layers): `static_w [4,C]`, tap 0 = current; per-token taps `dyn = weight_proj(x_k)` (resp. `x_v`; the normed attention input, not k) `[T, 8×4]` laid out `[head][lag]`: `y_t = (Σ_{l=0..3} (static_w[l,c] + dyn[t, c//128, l])·x_{t-l} + x_t) · 9^-0.5`. Conv input `x` is the projection **rounded to the conv-state dtype (bf16) at the op boundary**, and every reader (the current tap, earlier rows of the same call, later calls) sees that rounded value (`ops/sconv.py:138-145, 254-255`); K/V likewise as stored in the cache. This is what makes an R-row call equal R 1-row calls bitwise (G0). Accumulate fp32; `y` rounded to bf16. The residual inside is `+ x_t`; the block residual is separate (a15). Hidden-stream inputs: `attn_sconv` convolves the `wo_ud` output, `mlp_sconv` the MoE output. | `ops/sconv.py:12-19, 96-183, 298, 951` |
| a7 | k / v / q / r post-processing | `k = bf(clip(bf(conv_k), 22.627))`; `v = bf(rms(bf(conv_v), no gamma))`; `q = bf(clip(q ⊙ q_l2_clip.scale, 22.627))`; `r = bf(clip(r ⊙ rel_l2_clip.scale, 8.0))`, per head row. `clip(x,θ) = x·min(θ/(‖x‖₂+1e-6), 1)`. θ_qk = `qk_l2_clip_threshold` = 16√2. | `ops/sconv.py:795-813`, `ops/norm.py:73-82` |
| a8 | Log scaling (full layers only) | `tau(pos) = 1 + 0.1·ln(max((pos+1)/8192, 1))`, fp32, absolute position. `q = bf(q_clipped · tau)`, `r = bf(r_clipped · tau)` (rounded after the clip, then scaled). Never k. tau = 1 for pos < 8192. | `ops/attention.py:341-365`, `ops/norm.py:80-81` |
| a9 | Relative bias | `rel[t,h,d] = bf(Σ_j r[t,h,j]·rel_logits_proj[j,d])` (fp32 accumulate), table `[16, E]`, E = window (SWA, `rel_extent = sliding_window_size`) or `rel_extent` (full). `d = pos_q − pos_k`. SWA: keys with `0 ≤ d < 1024` only (window = 1023 back + self). Full: all `d ≥ 0` visible; bias = `rel[d]` for `d < 1024`, **0** for `d ≥ 1024`. | `ops/sconv.py:466-489` (`shift_rel_proj`), `attention.py:207-209` |
| a10 | Attention + sink | `s_{t,j} = q_t·k_j / √128 + rel[t,h,d]` (bias **not** scaled). Softmax over visible keys plus one extra logit `sink[t,h]` with value 0: `o = Σ_j e^{s_j} v_j / (Σ_j e^{s_j} + e^{sink})`. vLLM computes `o_softmax · sigmoid(lse − sink)`, identical. The sink logit is per token, per head (projected from `x_q`), sign +. | `ops/attention.py:81-82, 272-314` |
| a11 | Output gate | `o' = bf(o · sigmoid(gate))` (gate per channel, `[3072]`), `attn_proj = bf(o' · wo_ud^T)`; `attn_out = bf(sconv_static(attn_proj; attn_sconv))`. | `ops/attention.py:297-299`, `model.py:358` |
| a12 | Loop (passes) | Visit order: layers 0-5 (prelude), 6-17 ×3 (passes 0,1,2), 18-23 (coda) = 48 visits; visit index v (0..47) is the cache-slot and MTP-tap index. At (6, p=0): save `prelude = (blocks, partial)`. At (6, p≥1): `bank = blocks[len(prelude.blocks)+1:] + [partial]` (= [block of layers 6-11, partial of 12-17] of the previous pass), then restore `prelude`. In a pass p ≥ 1 visit of layer i: `msg = attn_res(bank[:1], bank[1], MQ[i-6])` (exactly two bank entries, `loop.py:228-236`: the core must be two attn-res blocks; the loader refuses other shapes) (4 fp32 rows: q, k, v, mlp; no norm). Each sublayer input is blended before its norm: `dest` = the fp32 mix of a3 (SWA: broadcast to 3 rows), `blend(dest, msg) = (1 − tanh(f))·dest + tanh(g)·msg` with `h = bf(bf([rms(dest), rms(msg)]) · down^T)` (`down [64, 2D]`, dest half first), `g = h·gain^T`, `f = h·forget^T` (fp32 out). Then a3's norms (rows q,k,v; SWA: `attn_norm` ×3), and for the MLP input `blend(mix_mlp, msg_mlp)` then `mlp_norm`. Controllers: `controllers.{attn,k,v,mlp}` = streams q, k, v, MLP (llama.cpp calls them q/k/v/ffn). | `loop.py:37-63, 199-316, 318-398` |
| a13 | Router | `l = x·gate.weight^T` (fp32), `sel = sigmoid(l) + gate.bias` (bias fp32, selection only), top-8 by `sel` (ties → lower id), `γ_k = sigmoid(l_k) / Σ_sel sigmoid(l) · route_scale` (route_scale = 8 here, equal to top-k only by coincidence; computed as softmax of logsigmoid; no eps). | `ops/moe.py:32-64, 110-145` |
| a14 | Experts (SiTU) | `h = x·W13_e^T` `[2I]`, rows **interleaved**: `gate = h[0::2]`, `up = h[1::2]` (bf16 GEMM out). `a = bf(4·tanh(gate/4) · sigmoid(gate) · 25·tanh(up/25) · γ)` (sigmoid of the **uncapped** gate; γ applied **before** w2). `y = Σ_e bf(a·W2_e^T)` fp32 combine, rounded once to bf16. Activation quant (vLLM SM100/SM12x only): section c. | `ops/activation.py:61-78, 209-278`, `ops/moe_sm90.py:111-157`, `moe.py:261-312` |
| a15 | Block residual | Layer i with `i % 6 == 0` ("first"): `mlp_in = attn_res(blocks + [partial], attn_out, mlp_q[i], mlp_norm)`, then `blocks.push(partial)`, `partial' = bf(attn_out + mc)`. Else `mlp_in = attn_res(blocks, partial + attn_out, …)`, `partial' = bf((partial + attn_out) + mc)` (fp32 sum in that order). `mc = bf(sconv_static(MoE(mlp_in); mlp_sconv))`. Blocks: [embed, L0-5, L6-11, L12-17, L18-23 partial]; at most 4 stored + partial. | `model.py:332-381`, `ops/sconv.py:379-385` |
| a16 | Output | `h = attn_res(blocks, partial, output_query, norm)`; `logits = h·unembed^T` (bf16 weights, fp32 out); `logits = 30·tanh(logits/30)`; keep columns `[0, 200058)` (padded rows 200058..201023 dropped). | `model.py:426-431`, `configs.py`, `Inkling2LogitsProcessor` |

Rounding points vLLM uses (oracle `--rounding vllm`): the residual stack and partials, every linear output, every normed mix, conv outputs,
clipped q/k/r, rel logits, the **unnormalized attention probabilities before P·V** (row sums stay fp32, `ops/attention.py:807`), gated attention output, expert GEMM outputs, SiTU output, MoE output. Mixes fed to blends and the blends
themselves stay fp32. The default oracle (`--rounding none`) keeps everything in fp32 (bf16 weights).

## b. Persistent state per sequence

| State | Visits | Shape per token / slot | dtype | Bytes |
|---|---|---|---|---|
| KV, SWA | 40 (layers 0-4, 6-10, 12-16 ×3 passes for the core ones, 18-22) | ring of 1024 positions × K,V `[8,128]` | bf16 | 4 KiB/token, capped at 4 MiB per visit |
| KV, full | 8 (visits 5, 11, 17, 23, 29, 35, 41, 47 = layers 5, 11×3, 17×3, 23 … see note) | append-only K,V `[8,128]` | bf16 | 4 KiB/token per visit |
| Conv windows | 48 | last K−1 = 3 inputs of 4 streams: k 1024, v 1024, attn-out 2304, mlp-out 2304 (= 6656 ch) | bf16 (vLLM stores the bf16 conv input) | 39 KiB per visit, 1.9 MiB per sequence |
| Within-token only | – | block stack (≤4 + partial), bank (2), prelude copy, messages | f32 workspace | not persistent (but the MTP taps are exported from it, section f) |
| Prefix snapshot (reuse past the ring margin) | 40 SWA + 48 conv | full SWA rings + conv windows | bf16 | **168 MB SWA + 1.9 MB conv per stored prefix**; full-visit KV truncates by length |

Note: full visits are v5 (L5), v11/v23/v35 (L11 passes 0-2), v17/v29/v41 (L17 passes 0-2), v47 (L23).

| Context | full KV (8 visits) | SWA KV (40 visits) | conv | total / sequence | KV read per decode token |
|---|---|---|---|---|---|
| 4k | 134 MB | 168 MB | 1.9 MB | 0.30 GB | 302 MB |
| 32k | 1.07 GB | 168 MB | 1.9 MB | 1.24 GB | 1.24 GB |
| 128k | 4.30 GB | 168 MB | 1.9 MB | 4.47 GB | 4.46 GB |

## c. Weight map

All names carry the prefix `model.llm.`; `model.visual.*` (636 MB) is skipped for the text port. One file, 19.53 GB; LLM part 18.89 GB,
of which experts 15.67 GB. `N` = layer, `{F}` = full layers 5/11/17/23, `{S}` = SWA layers, `{C}` = core 6-17.

| Tensor | Shape | dtype | Count | Role |
|---|---|---|---|---|
| `embed.weight` | [201024, 2304] | bf16 | 1 | token embedding (rows ≥ 200058 unused) |
| `embed_norm.weight` | [2304] | bf16 | 1 | a1 |
| `unembed.weight` | [201024, 2304] | bf16 | 1 | a16; vLLM runs it bf16 → fp32 |
| `norm.weight` | [2304] | bf16 | 1 | final norm (a16) |
| `attn_res.attn_queries.{0..22}` | [2304] or [3,2304] (idx 4,10,16,22) | f32 | 23 | a2' (raw; ×D^-0.5) |
| `attn_res.mlp_queries.{0..23}`, `attn_res.output_query` | [2304] | f32 | 25 | a2' |
| `middle_cycle_cross_pass_queries`, `…_kv_queries` | [24, 2304] | f32 | 2 | a12 |
| `layers.N.attn.{wq_du,gate_attn}.weight` | [3072, 2304] | bf16 | 24 each | a4 |
| `layers.N.attn.wr_du.weight` | [384, 2304] | bf16 | 24 | a4 |
| `layers.N.attn.attn_sink_proj.weight` | [24, 2304] | bf16 | 24 | a4/a10 |
| `layers.N.attn.{wk_dv,wv_dv}.weight` | [1024, 2304] | bf16 | 24 each | a4 |
| `layers.N.attn.wo_ud.weight` | [2304, 3072] | bf16 | 24 | a11 |
| `layers.N.attn.q_l2_clip.scale` | [24, 128] | bf16 | 24 | a7 |
| `layers.N.attn.rel_l2_clip.scale` | [24, 16] | bf16 | 24 | a7 |
| `layers.N.attn.rel_logits_proj.proj` | [16, 1024] | bf16 | 24 | a9 |
| `layers.{S}.attn.{k,v}_sconv.weight` | [1024, 1, 4] | bf16 | 20 each | a6 static |
| `layers.{F}.attn.{k,v}_sconv.static_w` | [4, 1024] | bf16 | 4 each | a6 dynamic |
| `layers.{F}.attn.{k,v}_sconv.weight_proj` | [32, 2304] | bf16 | 4 each | a6 dynamic taps |
| `layers.{S}.attn_norm.weight` | [2304] | bf16 | 20 | a3 |
| `layers.{F}.{q,k,v}_input_norm.weight` | [2304] | bf16 | 4 each | a3 |
| `layers.N.{attn_sconv,mlp_sconv}.weight` | [2304, 1, 4] | bf16 | 24 each | a6/a11/a15 |
| `layers.N.mlp_norm.weight` | [2304] | bf16 | 24 | a15 |
| `layers.N.mlp.gate.weight` / `.bias` | [128, 2304] bf16 / [128] f32 | | 24 | a13 |
| `layers.{C}.controllers.{attn,k,v,mlp}.down_weight` | [64, 4608] | bf16 | 12×4 | a12 |
| `layers.{C}.controllers.{…}.{gain,forget}_weight` | [2304, 64] | bf16 | 12×4 each | a12 |
| `layers.0-19.mlp.experts.w13_weight` | [128, 2304, 1152] | u8 (2 e2m1 / byte) | 20 | NVFP4, out rows interleaved gate/up |
| `layers.0-19.mlp.experts.w13_weight_scale` | [128, 2304, 144] | f8 e4m3 | 20 | per 16 inputs |
| `layers.0-19.mlp.experts.w2_weight` / `_scale` | [128, 2304, 576] u8 / [128, 2304, 72] e4m3 | | 20 | NVFP4 |
| `layers.0-19.mlp.experts.{w13,w2}_weight_amax` | [128] | f32 | 20 each | per-expert amax |
| `layers.0-19.mlp.experts.{w13,w2}_weight_outer_scale_denominator` | [1] | f32 | 20 each | = 2688.0 in every layer |
| `layers.20-23.mlp.experts.w13_weight` / `_scale` | [128, 2304, 2304] e4m3 / [128, 2304, 72] u8 e8m0 | | 4 | MXFP8 |
| `layers.20-23.mlp.experts.w2_weight` / `_scale` | [128, 2304, 1152] e4m3 / [128, 2304, 36] u8 | | 4 | MXFP8 |

Quant formats (checked bit-exactly against `mx.dequantize` on endor for one expert of layer 0 and layer 20):

| Format | Dequant | Notes |
|---|---|---|
| NVFP4 (`amax_denominator`) | `W[n,k] = e2m1(code) · e4m3(scale[n, k/16]) · (amax[e] / denom)`; low nibble = even column; e2m1 = {0, .5, 1, 1.5, 2, 3, 4, 6} with bit 3 the sign | `denom = 2688 = 6·448`, so `amax/denom` is exactly ModelOpt's `weight_scale_2` (per expert; gate and up share the w13 value). Maps 1:1 onto intarsiaml's `MoeExperts` NVFP4 layout + `gscales [E,3] = (w13_amax, w13_amax, w2_amax)/2688` (`quant/moe.rs:76,511-520`), after de-interleaving w13 rows (and their scale rows) into gate-then-up. Not new. |
| MXFP8 | `W[n,k] = e4m3(w) · 2^(scale[n, k/32] − 127)` | `hf_quant_config` says `weight_block_shape [32,32]`, but the stored scales are `[N, K/32]` and differ row to row: it is 1×32. **New** format for intarsiaml (no MXFP8 anywhere; `MatmulFp8Block` is 2-D, cpu-only). |
| Activations, NVFP4 group (vLLM SM100 / SM12x; GB10 is SM121) | per token on the **bf16** row, fp32 math: `enc = min(1536/amax, FP32_MAX)`, `dec = 1/enc`, `outer = amax·(1/1536)`; per 16-block `s6 = e4m3(min(bmax·(enc·(1/6)), 448))`, `s4 = e4m3(min(bmax·(enc·0.25), 448))`; `q = e2m1_rne_satfinite(x · min(1/(s·dec), FP32_MAX))` (ties to the even code: 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2, 3.5→4, 5→4); `err = Σ|q·s·outer − x|` summed by adjacent pairs; **s4 only if err4 < err6**. GEMM: `Σ (q_a s_a)(q_w s_w)` fp32 (`dot_scaled` e2m1) × row scale `(x_amax·w_amax)·rcp(1536·denominator)`, bf16 out | `ops/nvfp4.py:28-130, 189-249` (`_e2m1`, `_block_scales`, `_quantize_block`, `_quantize_columns`, `_quantize_row`), used by `ops/moe.py:307-343` (gate/up input) and `ops/activation.py:62-78, 122-206` (down input = bf16 SiTU·γ). The SiTU there uses an `ex2.approx`/`rcp.approx` sigmoid and `tanh.approx` softcaps (`activation.py:20-58`). |
| Activations, MXFP8 group | per 32: `scale = amax/448` rounded **up** to a power of two (fp32 bit trick `(bits + 0x7FFFFF) & 0x7F800000`; exponent 0 if `≤ 2^-127`, max 254); `v = e4m3(clamp(x/scale, ±448))`. SM12x GEMM dequantizes both sides to bf16 (exact for e4m3·2^k) and runs a bf16 `tl.dot`, fp32 acc | `ops/mxfp8.py:15-30`, `ops/moe_sm120.py:77-160` |
| Which path the GB10 goldens ran | **W4A4**: SM121 is capability family 120 → `sm120 = True`; `humming_nvfp4` needs `--moe-backend humming` (default `auto`, `config/vllm.py:1152-1154`), absent from the launch line, so experts run `bmm_sm120` on the quantized rows (`moe.py:153-159, 243-292`). Even `humming` on SM12x consumes the same W4A4 quantized rows (`moe_humming.py:64-73`); only SM90 (Hopper) is W4A16. | |
| vLLM on SM90 (Hopper) | **W4A16**: Humming GEMMs on bf16 activations (`ops/moe_sm90.py:1-9`, `moe.py:204-224`) | So vLLM itself ships both numerics; oMLX is W4A16 too. |

## d. Three-way diff: vLLM vs llama.cpp vs oMLX

| Point | vLLM | llama.cpp (`inkling2*.cpp`, GGUF) | oMLX v0.8.1 | Matches vLLM |
|---|---|---|---|---|
| Expert numerics | W4A4 on SM100/12x, W4A16 on SM90 | requantized at conversion (Q8_0 / Q4 GGUF), ggml activation handling | W4A16, native codes via `gather_qmm`, NVFP4 global scale applied after the matmul (`model.py:248-251, 288-300`) | oMLX = vLLM-SM90 up to rounding |
| Static conv rounding | fp32 sum + x, round once (`ops/sconv.py:298`) | fp32 (`inkling2.cpp:215-236`) | rounds the conv sum to bf16 **before** `+ x` (`omlx model.py:380-383`) | llama.cpp |
| Block storage | raw bf16, normalized in fp32 at every mix | raw, `rms_norm` at every mix (`inkling2.cpp:178-191`) | unit-RMS fp32 blocks stored once (`model.py:608, 677`) | all equivalent (normalize_values); differs by bf16 rounding of the stored block |
| Controller | bf16 pair → down → bf16 hidden → fp32 gain/forget | all f32 (`inkling2.cpp:240-248`) | same as vLLM (`model.py:344-364`) | oMLX |
| tau on q/r | round clipped, ×tau, round | f32, no rounding (`:339-342`) | same as vLLM (`model.py:517-519`) | oMLX |
| Rel logits | fp32 accumulate, bf16 store | f32 table gathered by distance (`:357-365`) | bf16 matmul (`model.py:520`) | ≈ all |
| Sink | `sigmoid(lse − sink)` post-scale | subtracts the sink from the bias, zero sink logit (`:359-368`) | zero-value key with the sink logit (`kernels.py:417-446`) | equivalent |
| Beyond-extent bias (full layers) | 0 (`shift_rel_proj`) | 0 (padded table column) | 0 (`kernels.py:447`) | all |
| Router | softmax(logsigmoid) of the selected, no eps | sigmoid gating + bias, normalized weights × scale | sum + 1e-20 (`kernels.py:512-534`) | all (eps negligible) |
| w13 row order | interleaved gate/up | converter reorders into `ffn_gate_up_exps` (`:154`) | keeps interleave, reads `0::2`/`1::2` (`model.py:269-292`) | oMLX; llama.cpp's layout is its own |
| MXFP8 layer detection | from `hf_quant_config` | converter | from tensor dtypes; the fallback guess `idx >= n−8` (`model.py:215`) would wrongly flag 16-19, corrected by `sanitize` | – |
| Cache-slot numbering | visit order 0..47 (`configs.py layer_visit_ids`) | physical 0..23 for prelude/pass 0/coda, 24..47 for passes 1-2 (`inkling2-loop.cpp:99`) | visit order | oMLX |
| Vocab padding | slice to 200058 | view to unpadded (`:445`) | −inf bias on padded ids | equivalent |
| DSpark windows | local draft layers 1024+8 keys, layer 1 unbounded | n/a | all draft layers windowed to 1023 (`dspark.py:11-12`) | – (section h) |

## e. Tokenizer, chat template, reasoning

- vLLM loads `tokenizer.json` via `PreTrainedTokenizerFast` (`tokenizers/registry.py:61-64`); the `tiktoken/` model is the same vocabulary
  (200058 ranks; the 199998 BPE entries match byte for byte, 60 specials 199998..200057). Byte-level BPE, Split pre-tokenizer (o200k-style
  regex) then ByteLevel without prefix space; no normalizer; **no BOS** added by the post-processor.
- Specials: `<|endoftext|>` 199999 (pad), `<|message_user|>` 200000, `<|message_model|>` 200001, `<|message_system|>` 200002,
  `<|message_tool|>` 200003, `<|content_text|>` 200004, `<|content_image|>` 200005, **`<|content_model_end_sampling|>` 200006 = EOS**,
  `<|content_thinking|>` 200008, `<|end_message|>` 200010, `<|content_tool_error|>` 200022, `<|content_xml|>` 200024 (tool declarations),
  `<|begin_of_text|>` 200028 (never emitted), `<|content_invoke_tool_json|>` 200049, `<|content_invoke_tool_text|>` 200057,
  image placeholder `<|unused_200054|>`.
- Template: `<|message_system|><|content_text|>Thinking effort level: {e}<|end_message|>` (default 0.9; named none 0 / minimal 0.1 /
  low 0.2 / medium 0.7 / high 0.9 / xhigh, max 0.99) before the first non-system message, then
  `<|message_user|><|content_text|>{u}<|end_message|>`; assistant turns `<|message_model|>[<|content_thinking|>r<|end_message|><|message_model|>]<|content_text|>t<|end_message|><|content_model_end_sampling|>`.
  **No generation prompt**: the prompt ends on `<|end_message|>` and the model emits `<|message_model|>` itself.
- vLLM's renderer (`renderers/inkling.py:130-162` → `inkling_encoding.py:444-533`) gives the same token stream as `chat_template.jinja` for
  plain, system, multi-turn, effort and tool prompts. Differences: effort spelling for non-named floats, `content=None` assistant turns
  (jinja emits a lone end token), string tool-call arguments, `developer` role, `tool_choice`, `continue_final_message`, literal special
  strings inside user text (vLLM keeps them as text).
- Reasoning: the model opens `<|message_model|><|content_thinking|>` itself; the parser (`parser/inkling.py:236, 336-397`) starts in
  `MESSAGE_HEADER`, `<|content_thinking|>` enters reasoning, `<|end_message|>` leaves it; a text block marks the end of reasoning. Tool
  call: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`. Effort 0 does not fully disable thinking
  (observed in the oracle: a short thinking block, then the answer).

## f. Gap list against intarsiaml

Existing coverage per step (backend columns: c / m / cu / h = cpu / metal / cuda / hip; ✓ have, – missing):

| Step | Existing op | c | m | cu | h | Verdict |
|---|---|---|---|---|---|---|
| Embedding + RMSNorm (a1), final norm | `Embedding`, `RmsNorm` | ✓ | ✓ | ✓ | ✓ | reuse |
| bf16 projections (a4, a11, router, dyn taps) | `Matmul` / `MatmulPair` | ✓ | ✓ | ✓ | ✓ | reuse, with a stacked in_proj per layer |
| Sigmoid output gate (a11) | `SigmoidMul` | ✓ | ✓ | ✓ | ✓ | reuse (or fold into attention) |
| Unit-RMS v (a7) | `RmsNorm` with ones | ✓ | ✓ | ✓ | ✓ | fold into the KV prologue |
| Router (a13) | `MoeRoute` (sigmoid score, selection-only bias, scale, renorm: DSv4.1) | ✓ | – | ✓ | – | reuse; metal + hip kernels new |
| NVFP4 experts (a14) | `MoeExperts` (SiLU/GELU GLU) | ✓ | ✓ | ✓ | ✓ | new `situ` activation attr (gate/up caps, γ before w2); loader de-interleaves w13 |
| MXFP8 experts | – | – | – | – | – | **new** format + op |
| Logit softcap (a16) | host `softcap_logits` (Gemma) | ✓ | ✓ | ✓ | ✓ | reuse; read `final_logit_softcapping` for this family |
| Depth mix (a2), loop blend (a12) | – (DSv4.1 hyper-connections are a different mix) | – | – | – | – | **new** |
| Static/dynamic short conv with `(+x)·scale` (a6) | `ShortConv` is LFM2's `C·conv(B·x)`, f32 `Recurrent` state, cpu+metal | – | – | – | – | **new** (different math and state) |
| q/k/r L2 clip, tau, rel logits (a7-a9) | – | – | – | – | – | **new** |
| Attention with rel bias by distance + per-query sink + window, G=3 | `AttnSparse` has sinks (DSv4.1 MQA); Metal split-KV supports G ∈ {1,2,4,8} | – | – | – | – | **new** |
| W4A4 activation QDQ | `ActQdq` (DSv4.1 fake quant, other formats) | – | – | – | – | new mode of the expert op (switchable) |

Op set (revised after the review, finding 10; the skeleton lane owns the Rust contract and numbering, appended after `BridgeWrite` = 66).
Launch-granularity ops, each with an f64 cpu reference; the small reference ops (`AttnResMix`, `LoopBlend`, `SconvRes`) stay as the
cpu decomposition the fused ops are tested against:

| Op | Semantics (section a rows) |
|---|---|
| `InkInProj` | one GEMM over the stacked `[q \| r \| gate \| sink \| k \| v \| k_taps \| v_taps]` rows with a **row-range input select** (rows 0..6552 read `x_q`, k + k_taps read `x_k`, v + v_taps read `x_v`), so it is one launch on all 48 visits (28 visits have three distinct inputs). Opt-in e4m3 form (lane ink-fp8, coordinator go-ahead 2026-11-21): w `fp8_e4m3` plus a third input `scales [3]` f32, one W8A16 weight scale per pack (`pq = rA(scales[0] * x . e4m3(Wq)^T)`, ...); bf16 calls keep exactly two inputs; a backend without the e4m3 kernel runs the packs as three `MatmulFp8` |
| `InkKvPrologue` | a6 static or dynamic K/V conv (ring taps, slot-dtype rounding), k clip, v unit-RMS, KV write (ring or linear); q clip ⊙ scale, r clip ⊙ scale, tau, `rel = r·rel_proj`. Per-row positions |
| `RelSinkAttn` | a9-a11: GQA G=3, distance-indexed bias (zero past the extent on full layers), window or none, per-query sink, bf16 P, sigmoid gate epilogue. Decode/verify kernel reads the ring in place; prefill kernel reads `[ring tail \|\| chunk]`. Whether it fuses with `InkKvPrologue` for decode (one threadgroup per kv head) is the skeleton lane's call |
| `ConvMix` | `SconvRes` (a6 static, `+x`, 5^-0.5) + block residual (a15) + the **next** `AttnResMix` (a2, 1 or 3 queries, push-first-layer rule) + optional `LoopBlend` (a12, messages on looped visits) + norm → bf16 sublayer inputs. Two instances per visit: attention side (→ `mlp_in`) and MLP side (→ next visit's attention inputs); the MLP-side one may also run the router GEMV + top-k (a13) |
| `InkMsgMix` | a12 messages: `attn_res(bank[:1], bank[1], MQ)` → 4 fp32 rows; once per looped visit (may fold into `ConvMix`) |
| `MoeExperts` + `situ` attr, `MoeExpertsMxfp8` | a14 over NVFP4 (existing op, new activation attr) and MXFP8 weights (layers 20-23, new); `act_quant ∈ {w4a16, w4a4}` |
| reused | `Embedding`, `RmsNorm`, `MoeRoute` (if not folded into `ConvMix`), `Matmul` (wo, unembed) |

Contract points the op list must carry: per-row positions (tau and the ring row are per row, never per call); an MTP tap export
(the partial entering visits 5, 23, 41 and the final-norm output, `loop.py:372-373`) out of the otherwise non-persistent workspace;
greedy argmax on the **raw** logits on device (softcap is monotone), so the host softcap never blocks device token feedback.
Launch budget: about 6-7 launches per plain visit and 7-8 per looped visit, about 350 per token against about 700 with the
unfused list (review finding 10).

Block shape and the loop:
- `Block::Inkling2(InklingLayer)` per **physical** layer (like `Block::Gemma`): all weights of layer i (stacked, 16-byte-aligned in_proj
  `[q | r | gate | sink | k | v | k_taps | v_taps]` with row ranges per input), `kind ∈ {Swa, Full}`, `core: Option<controllers>`,
  expert format per layer.
- A model-level **visit table** `[(layer, pass)]` of 48 entries; slots are indexed by visit, weights by layer; `prelude`/`bank` restarts
  happen at `(core_start, p ≥ 1)`. The block stack, bank, prelude copy and messages are a per-token f32 workspace (≤ 7 × D × rows), never
  persistent state.
- `SlotSpec`s per visit: SWA → `Ring { rows: 1024 + margin, width: 1024, planes: 2 (K, V), dtype: bf16, kind: WindowKv }`;
  full → `AttnKv { n_kv_heads: 8, head_dim: 128, dtype: bf16 }`; conv → `Ring { rows: 3 + margin, width: 6656, planes: 1, dtype: bf16,
  kind: ConvWindow (new) }`.
- Ring semantics (vLLM addresses both caches by absolute position and never overwrites a row a call still needs, `ops/sconv.py:5-10,
  137-145, 804-813`):
  - `margin ≥ (max verify rows − 1) + rollback depth`; with `SPEC_K_MAX` 6, **margin ≥ 6** for both ring kinds. Position `p` lives in
    row `p % rows`.
  - Decode and verify kernels read the ring **in place** as at most two contiguous segments, recovering each key's position from its
    row index; never linearize into a `[W+T, D]` copy (that is 4 MiB per SWA visit, 168 MB per token).
  - Prefill chunks longer than the margin read `[ring tail ‖ chunk]` (keys of the chunk come from the call's own rows), then write
    only the last `rows` positions of the chunk. Conv: the at most 3 pre-chunk taps come from the ring, the rest from the chunk, and the
    last `rows` positions are written back. (The oracle's `VisitState.append` does the same.)
  - Writes happen before the attention read within a call (KV of the call's own rows is visible to its later rows).
  - Slot-dtype rule (a6): K, V and conv inputs are rounded to the slot dtype at the op boundary; same-call and later readers see only
    the rounded value. Required for G0's rows-exact property.
  - Prefix reuse past the margin needs a snapshot: 168 MB of SWA rings + 1.9 MB of conv per stored prefix (section b).
- Conv state as `Ring` vs `Recurrent` (Fable asks): `Recurrent` rolls back speculative rows through the per-row stash (`state.rs:204-206`),
  i.e. a copy of every recurrent slot per verify row (1.9 MB × rows per step, plus a launch per slot class); prefix reuse goes through
  snapshots. `Ring` rolls back by length within `margin` with no copies (`state.rs:20-26`), at the price of position-indexed tap reads in
  `SconvRes`/`InkKvPrologue`. Ring is cheaper and also matches the SWA KV (prefix reuse past the margin needs snapshots either way). But
  `Ring` exists today only on cpu and cuda (DSv4.1); metal and hip need the ring plumbing in both cases. **Use Ring.**
- Recipe keys (new family `inkling2`): `[model] family = "inkling2"`, `[config]` renames for `n_layers_per_attn_res_block`,
  `loop_config.*`, `local_layer_ids`, `sliding_window_size`, `rel_extent`, `d_rel`, `qk_l2_clip_threshold`, `rel_l2_clip_threshold`,
  `log_scaling_*`, `mlp_{gate,up}_softcap`, `route_scale`, `final_logit_softcapping`, `unpadded_vocab_size`; `[layer] prefix =
  "model.llm.layers.{i}"`; `[defaults] act_quant = "w4a16" | "w4a4"`; `[top] embed/unembed/final_norm/embed_norm`; vocab: logits buffer
  width 200058 (skip the 966 padded unembed rows). New keys need `docs/recipes.md` entries (`tests/recipes_doc.rs`).
- Loader: strip `model.llm.`, skip `model.visual.`; queries f32 ×D^-0.5 at load; message queries regrouped to (q, k, v, mlp); w13
  de-interleave (rows 2j → gate j, 2j+1 → up j, both values and scales); `gscales` from amax/denominator; MXFP8 tensors by
  `hf_quant_config` group (`compute_dtype: mxfp8`); refuse a checkpoint whose tensor pattern disagrees with `local_layer_ids` (the oracle
  asserts this).

Agreement with Fable's port requirements (§9): 1 visit table — agree; 2 stacked in_proj — agree; 3 macro-ops — agree with the
reuse noted above; 4 workspace not state — agree; storing blocks unit-RMS once is exact enough (normalization is idempotent up to the
1e-6 eps) but the partial must stay raw (it feeds the residual); 5 conv as Ring — agree; 6 rows-exact — agree; 7 activation quant as a
switchable mode — agree, **default W4A16** (see gate below); 8 drafter as a second model — agree. Fable's review withdrew "W4A4 is the
reference numerics" as a default; W4A4 stays the fast GB10 prefill mode, gated statistically (G3), not by bit match.

**Gate definitions ("exact")**: section i.

## g. Byte floor and FLOPs per decode token

From the real tensor sizes (`python3 -I tools/inkling2_floor.py`, same numbers as Fable's `raw/floor.py`). Per visit: attention projections 53.78 MB bf16;
one NVFP4 expert 4.479 MB, one MXFP8 expert 8.211 MB (×8 per visit, independent per pass); router 0.59 MB; controllers 4.72 MB on the
24 looped visits; unembed over the 200058 real rows. Unique LLM weights 18.89 GB (experts 15.67 GB).

| Layout | attn | experts | ctrl+router+small | unembed | **GB / token** | M5 Max 614 GB/s | GB10 273 GB/s |
|---|---|---|---|---|---|---|---|
| Native (bf16 attn + unembed, NVFP4 + MXFP8 experts) | 2.58 | 1.84 | 0.15 | 0.92 | **5.49** | 112 tok/s | 50 tok/s |
| fp8 attention | 1.30 | 1.84 | 0.15 | 0.92 | 4.21 | 146 | 65 |
| fp8 attention + fp8 unembed | 1.30 | 1.84 | 0.15 | 0.47 | 3.75 | 164 | 73 |
| NVFP4 attention | 0.73 | 1.84 | 0.15 | 0.92 | 3.64 | 169 | 75 |
| NVFP4 attention + NVFP4 unembed | 0.73 | 1.84 | 0.15 | 0.26 | 2.97 | 206 | 92 |
| Native, MXFP8 → NVFP4 (layers 20-23) | 2.58 | 1.72 | 0.15 | 0.92 | 5.37 | 114 | 51 |
| NVFP4 everything | 0.73 | 1.72 | 0.15 | 0.26 | 2.85 | 215 | 96 |

Plus KV reads (section b): +0.30 GB at 4k, +1.24 GB at 32k, +4.46 GB at 128k (native at 32k: 6.73 GB → 91 / 41 tok/s).
FLOPs per decode token: 9.8 GFLOP in weight GEMVs (2 × active params), attention 0.9 / 3.8 / 13.4 GFLOP at 4k / 32k / 128k. Decode is
bandwidth-bound everywhere (M5 Max at 112 tok/s × 10 GFLOP ≈ 1.1 TFLOP/s).

Reference points: endor oMLX oQ4e decode 41-51 tok/s (≈2.9 GB/token layout → 22-27 % of its 206 tok/s floor), prefill ~1300 tok/s;
endor llama.cpp Q8_0 33 / 122; spark2 vLLM NVFP4 37-43 tok/s = 74-86 % of the GB10 native floor (50), prefill ~2800; EXO's claimed
125 tok/s on an M5 Max is **above the native floor (112)**, so EXO cannot be running native bf16 attention; it fits fp8 attention +
fp8 unembed (164) or 4-bit dense. The native path's realistic target on endor is ~75-90 tok/s (Fable §2: launch cost per visit).

## h. MTP / DSpark (for later lanes)

`mtp.safetensors`: 54 bf16 tensors, 636 MB: `fuse_weight [4,4]`, `mask_embed [2304]`, `norm.weight`; 3 layers × {`input_layernorm`,
`post_attention_layernorm`, `dflare_norm`, `self_attn.{q_proj [3072,2304], k_proj, v_proj, k_proj_ctx, v_proj_ctx [1024,2304], o_proj
[2304,3072], q_norm, k_norm, k_norm_ctx [128]}`, `mlp.{gate,up}_proj [9216,2304]`, `down_proj [2304,9216]`}; `markov_head.{markov_w1
[201024,256], rnn_proj [768,5120] + bias, hidden_out [2304,256] + bias, dflare_norm}`. No embed / unembed (shared with the target).

- Taps: target visits 5, 23, 41 (the **input partial** entering that visit: L5 prelude, L11 pass 1, L17 pass 2) and 48 (final-norm
  output). Per draft layer i: `ctx_i = rms(Σ_t softmax(fuse_weight)[i,t] · tap_t; dflare_norm_i)`; row 3 feeds the markov head.
- Context K/V per draft layer: `K = RoPE(k_norm_ctx(k_proj_ctx(ctx)))`, `V = v_proj_ctx(ctx)` into the draft's own KV cache.
- Draft block of 8 rows: row 0 = `embed_norm(embed(anchor))`, rows 1-7 = `mask_embed`; bidirectional inside the block; q/k RMSNorm +
  neox RoPE θ=1e6 over 128 dims; local draft layers 0, 2 see 1024 + 8 keys, layer 1 unbounded; SiLU-GLU MLP (9216).
- Markov head (r=256): `raw = rnn_proj([state, w1[prev], h_i, f_rnn])`; `g = σ(raw[:r])`, `c = tanh(raw[r:2r])`, `state' = g·state +
  (1−g)·c`, `latent = tanh(raw[2r:])`; `bias = unembed(hidden_out(latent))` (folded to `[V, r]`). Logits `30·tanh(bf(unembed(h_i) +
  bias)/30)[:200058]`. 7 proposals per step (`dspark_block_size = 7`).
- Sources: `V/dspark.py:249-559`, `ops/dspark.py`, `speculator.py`, `configs.py:238-306`. oMLX windows every draft layer (differs).
- Fable's cost model says a short, confidence-truncated draft (1-3) beats vLLM's fixed 7 on this MoE (verify bytes grow with the expert
  union).

## i. Oracle (`tools/inkling2_ref.py`), measurements, gate

Run with `uv run --project tools python tools/inkling2_ref.py <mode>`; real-model runs on endor via `fleetq run --host endor --load cpu
--mem 30 …`. Peak RSS 18-21 GB for short prompts, 34 GB at 1100 tokens (dense weights fp32, experts read per expert from the mmapped file, 6 GB LRU of dequantized experts). Prefill
of 100 tokens ≈ 20 s, decode ≈ 1 s/token on endor's CPU.

| Mode | What |
|---|---|
| `run --prompt … [--chat] [--dump DIR] [--max-layers N]` | logits; `--dump` writes `promptN.safetensors` with `embed`, `vNN.{partial_in, blocks, bank, attn_in_q/k/v, q, k, v, rel, sink, attn_proj, attn_out, mlp_in, router_logits, router_sel, route_margin, expert_ids, expert_w, moe_out, partial_out}`, `final.{blocks, partial}`, `final_hidden`, `logits` (NN = visit 00..47); `--max-layers N` runs the first N **visits** in execution order, then a16 on the state reached |
| `gen`, `greedycmp --golden greedy.json` | greedy decode (stateful path), first divergence vs vLLM's greedy text |
| `cmp --golden X.safetensors [--out Y]` | top-1 and 21-way KL vs an `intarsia-golden-topk-v1` file; `--out` writes the oracle's own top-20 golden |
| `selftest [--chunks a,b,c] [--no-decode] [--prompt-file F --n-tokens N]` | whole-sequence prefill vs chunked prefill vs token-by-token decode |
| `mkcapture`, `replay --capture DIR [--visits ...]` | write a synthetic module-boundary capture / run `Ref.attention` and `Ref.moe` on a captured vLLM run (layout in the code above `make_capture`: `attn_vNN` with positions, x_q, x_k, x_v, out, optional tau; `moe_vNN` with x, out; optional `meta.json`) and print the relative error per visit. Tested on a synthetic tiny capture: 0 error at matching settings, 0.16 MoE error when replaying a W4A4 capture at W4A16. 12k-token fake capture, visits 0 and 5: 19 s, peak RSS 24 GB, so the full 48-visit replay is about 8 min and under 32 GB (`fleetq … --expect 20 --mem 40`) |
| `tiny` | writes `golden/inkling2-tiny/`: config, `hf_quant_config.json`, checkpoint in the real names and formats, `ref_logits.safetensors` (f64 / f32 / bf16-rounded logits, f64 top-2 margins of a 40-token prompt), `ref_dump_f64`, `ref_dump_bf16` (free-running) and `ref_dump_bf16_tf` (teacher-forced: every visit fed the f64 run's `partial_in`, `blocks`, `bank`). Config chosen so every branch bites: `qk_l2_clip_threshold` 3.0, `head_dim` 32 ≠ `d_rel` 16, softcaps 2.0 / 1.5, window 8 < extent 12 < 40 tokens, `n_floor` 4, full layers 3 (core) and 7, MXFP8 layers 6-7, 3 passes, top-4 of 16 with route_scale 8. The seed search accepts a seed only if q and k clip on 25-75 % of rows per layer kind (now 37-62 %), f32 routing = f64 routing at every visit, and then takes the widest top-k margin. 13-16 of 16 experts are used per visit |
| knobs | `--dtype float64`, `--rounding none|vllm`, `--act-quant w4a16|w4a4` |

Checks run:
- Dequant: NVFP4 (layer 0) and MXFP8 (layer 20) expert 3, w13, bit-identical to `mx.dequantize` (oMLX venv on endor).
- State: the stateful path keeps an explicit 1024-row ring for SWA KV and 3-tap conv windows. Tiny model (window 8, extent 12,
  n_floor 4 so tau ≠ 1, 40 tokens, full layers 3 (core) and 7, MXFP8 layers 6-7, 3 passes): prefill vs chunked (13/14/13) vs
  token-by-token agree to **~1e-12 relative in float64** (f32: top-1 identical, max abs 9.4e-4 vs f64). Real model (`selftest`, 17 tokens, f32): chunked and token-by-token vs prefill max rel 1.5e-6, top-1 identical; peak RSS 20.8 GB.
  The tiny random net is chaotic under bf16 activations when free-running (partial_out error 1e-2 at v00, above 1 from v08, routing
  differs on 1-13 of 40 rows from v01), so bf16 backends gate teacher-forced (G1(b)): every visit's tensors at bf16 rounding are
  within 9e-3 of f64 (non-routed tensors; routed ones on routable rows ≤ 1.2e-2).
- G1(b) mutation check (`tools/inkling2_mutants.py`, one config key changed, teacher-forced bf16 vs the fixture's budget): control 0
  failures; no q/k clip 155, θ_qk 3.3 instead of 3.0: 90, no rel clip 139, up softcap off 32, gate softcap off 32, final softcap off 1,
  tau off 40, route_scale = top-k 48, window 9 instead of 8: 98; eps 1e-5 instead of 1e-6 passes (below bf16 resolution).
- Ring on real weights: 1100 tokens of `docs/ARCHITECTURE.md` as chunks 700 / 399 / 1 (the second chunk crosses position 1024, so ring
  wrap-around and `[ring tail ‖ chunk]` reads run on all 40 SWA visits) vs one whole prefill: max rel 1.3e-5, top-1 identical; peak RSS 34 GB.
- Generation (W4A16, rounding none): raw `"The capital of France is"` → " Paris.\n\nA. Noun clause …"; `"def fibonacci(n):"` → a
  correct recursive body; chat (effort 0) → `<|message_model|><|content_thinking|>The user asks a simple factual question … <|end_message|>
  <|message_model|><|content_text|>The capital of France is Paris.<|end_message|><|content_model_end_sampling|>`.

Against the vLLM goldens (`golden/inkling2-mini-vllm-*`, GB10, W4A4, batch-1 sequential; vLLM vs itself concurrent: 97-98 % top-1, KL21
5e-3..1.2e-2):

| Oracle variant | short (569 pos) top-1 / KL21 | long (1569 pos) top-1 / KL21 | 4k |
|---|---|---|---|
| W4A16, rounding none (default) | **98.24 % / 6.35e-3** | **97.64 % / 1.24e-2** | **95.87 % / 9.84e-3** (13096 pos) |
| W4A16, rounding vllm (with bf16 P; before it: 97.72 % / 6.06e-3, 97.51 % / 1.23e-2) | 97.54 % / 6.58e-3 | 97.26 % / 1.20e-2 | – |
| W4A4 exact (`ops/nvfp4.py`, `ops/mxfp8.py`), rounding none | 97.01 % / 1.04e-2 | 96.94 % / 1.54e-2 | – |
| W4A4 exact, rounding vllm (closest to vLLM's numerics; before bf16 P) | 96.31 % / 1.32e-2 | 96.62 % / 1.81e-2 | 95.58 % / 1.19e-2 |
| W4A4 with my earlier inferred quantizer (superseded) | none 95.61 % / 1.19e-2, vllm 97.36 % / 9.62e-3 | – | – |

Oracle variants against each other (`tools/golden_cmp.py`, long set unless noted): W4A16 none vs vllm rounding 98.79 % / 2.7e-3
(short 98.42 % / 1.5e-3); W4A16 vs exact W4A4 (none) 98.09 % / 1.16e-2; **exact W4A4 none vs exact W4A4 vllm-rounding 96.94 % /
1.92e-2**. Reading: with the bit-faithful quantizer, two runs whose activations differ only by bf16 rounding land 1.9e-2 apart, more
than either is from vLLM. The per-16-block 4-vs-6 choice and the e2m1 codes are discontinuous in the input, so the W4A4 noise
pattern depends on the exact upstream bits; an engine that is not bitwise identical to vLLM up to each quantizer gets an
independent noise draw, and W4A4 then sits about one W4A4-noise further from the golden than W4A16 does. So W4A16 stays the
oracle default and the G2/G3 reference; W4A4 is a selectable mode for speed, not for accuracy against these goldens.
On the 4k set the oracle is about twice vLLM's own concurrent noise there (97.5-98.3 % / 4.6-5.8e-3), but flat by position:
per 1024-token bucket 94.4-98.0 % with no step at the SWA window (1024) or the rel extent, so the window / extent logic holds on real
weights within about 2e-3 KL (bucket standard error); `cmp` is one whole-prompt prefill, so the ring is not exercised there, and tau
(positions ≥ 8191) is not reached by any golden: both rest on the 1100-token selftest above and on the capture replay (section j); the gap tracks low-confidence text (golden top-1 p ≈ 0.6-0.7 on docs and chat). Agreement does not
degrade with position (long_prose_700 per-100-token top-1 91-100 %, lowest where the golden itself is least confident).

Greedy (first 48 tokens vs `golden/inkling2-mini-vllm-greedy.json`, tokens matched before the first divergence): count 48, story 13,
code 24, science 3, chat_short 17 / 27 (effort none / 0.9), chat_system 6 / 24, chat_multiturn 6 / 47, chat_reasoning 23 / 3; every
continuation stays coherent and on-topic. Divergences are at near-ties (`' ** 0.5'` vs `'**0.5'`,
`'heart of the village'` vs `'not just a machine'`), as expected two numerics families apart.

**Gate ("exact") for intarsiaml:**

| Gate | Compares | Threshold | Basis |
|---|---|---|---|
| G0 op | each new op's kernel vs its f64 cpu reference; R = 1..16 rows | op tolerance; **bitwise rows-exact** (row r of an R-row call = the 1-row call) | engine convention; needed for greedy-exact speculation |
| G1 tiny | (a) the f32-activation path (cpu reference ops) on `golden/inkling2-tiny` vs `logits_f64`; (b) every backend's bf16 path, **teacher-forced per visit** (fed `ref_dump_f64`'s `vNN.partial_in`, `vNN.blocks`, `vNN.bank`, `final.*` as one 40-token prefill), vs `ref_dump_f64` | (a) top-1 identical at all 40 positions, max abs logit error ≤ 1e-2 (f32 oracle: 9.2e-4). (b) per visit (all 16) and per dumped tensor: rel err ≤ max(2 × the oracle's own error in `ref_dump_bf16_tf`, 1e-3). Rows whose f64 top-k margin (`route_margin`) exceeds 5 × the oracle's max \|`router_sel`\| error at that visit are routable: identical expert sets required there, and `expert_w` (compared by expert id), `moe_out` (routable rows) and `partial_out` (rows whose last 4 rows are routable, for `mlp_sconv`) are compared on those rows only. Plus prefill = chunked = decode, bitwise for rows-exact kernels. Reference comparator: `tf_check` in the oracle | the fixture hits clips, window/extent edges, tau ≠ 1, 3 passes, both full-layer kinds, MXFP8; the mutation check shows each of those breaks the gate. Legitimate numerics variants (teacher-forced): fp32 activations pass; an extra bf16 round on the fp32 mixes fails 1 tensor (v10 `partial_out`, 2.6 × the oracle's error). At 3 × every mutant still fails (no q/k clip 109, θ 3.3: 44, others unchanged), so 3 × (`tf_check(factor=3)`) is the safer budget if a backend stores mixes in bf16 |
| G2 oracle | real model, W4A16, vs this oracle's own top-20 (`golden/oracle/inkling2-mini-oracle-w4a16-{short,long}-top20.safetensors`, rounding none) | KL21 mean ≤ 4e-3 (short), ≤ 6e-3 (long); **confident rows** (oracle `p1 − p2 ≥ 0.5`: 363 of 569 short, 1206 of 1569 long): top-1 misses ≤ 1 + rows / 1000 (short 1, long 2); raw top-1 ≥ 97 % as a tripwire only | a correct bf16 engine (proxy: the oracle at `--rounding vllm`, with bf16 P, KL21 from `cmp`) scores KL21 1.67e-3 (short) / 3.21e-3 (long), confident misses 0 / 1, raw top-1 98.9 % / 99.0 %: 2.4 × / 1.9 × KL headroom. Exact W4A4 against it: KL 1.2e-2, 2 confident misses on long, so a W4A4 mode does not have to pass G2 |
| G3 vLLM | real model vs the vLLM W4A4 goldens (short, long, 4k) | KL21 mean ≤ oracle + 4e-3: **1.05e-2 (short), 1.65e-2 (long), 1.4e-2 (4k)**; confident-row misses ≤ 1 + rows / 1000 (golden `p1 − p2 ≥ 0.5`: 367 / 1211 / 8229 rows → 1 / 2 / 9); raw top-1 is reported, not gated | oracle: 6.4e-3 / 1.24e-2 / 9.8e-3 KL, 0 / 0 / 1 confident misses. Correct bf16 engine (proxy, with bf16 P): 6.6e-3 / 1.20e-2 KL, 0 / 0 misses: 3.9e-3 / 4.5e-3 headroom (4k not run for the proxy). Exact W4A4 (rounding vllm, measured before bf16 P): 1.32e-2 / 1.81e-2 / 1.19e-2, 1 miss on long and 1 on 4k, so a W4A4 mode passes short and 4k, not long. G3 cannot resolve bugs below about 5e-3 KL: the resolving gates are G0, G1 and G2 |

All disagreements against vLLM are near-ties: on confident rows the oracle misses 0 of 367 (short), 0 of 1211 (long), 1 of 8229 (4k).
"Exact" therefore means: bitwise rows-exact against our own 1-row path (G0), and within G1-G3 against the references. W4A16 is the
default numerics. The noise model (why W4A4 is not the reference although vLLM ran W4A4): see the reading above the gate table.

## j. Changes since f38cd515a, open questions

Changes since f38cd515a (the full vLLM `ops/` tree is now in `.worktrees/ink-official/models/inkling2/nvidia/ops/`):

| # | Change | Where |
|---|---|---|
| 1 | Exact activation quantizers written out: NVFP4 `fouroversix_mae` (bf16 input, `enc/dec/outer`, e4m3 6- and 4-based block scales, e2m1 RNE-satfinite, pairwise error, strict `err4 < err6`); **MXFP8 scale = amax/448 rounded up to a power of two** (my earlier OCP `floor(log2 amax) − 8` guess was wrong for ~19 % of blocks) | section c |
| 2 | Settled: the GB10 goldens are **W4A4** (SM121 → family 120 → `bmm_sm120`; no `--moe-backend`, default `auto`; even `humming` on SM12x is W4A4). Only Hopper runs W4A16 | section c |
| 3 | Oracle `--act-quant w4a4` now implements (1) line for line (`fake_quant_nvfp4_act`, `fake_quant_mxfp8_act`); W4A16 stays the default | `tools/inkling2_ref.py` |
| 4 | Re-measured: exact W4A4 is **further** from the W4A4 goldens than W4A16 (long 96.6 % / 1.8e-2 vs 97.6 % / 1.2e-2), and two exact-W4A4 runs differing only in bf16 rounding are 1.9e-2 apart. W4A4 noise is a fresh draw per implementation, so G2/G3 keep the W4A16 oracle as reference; thresholds unchanged | section i |
| 5 | Controller numerics confirmed equal to the oracle: `pair_rms_norm` → bf16 `[rms(dest), rms(msg)]`, bf16 `hidden`, fp32 `gain_forget`, unfused fp32 blend (`enable_fp_fusion=False`), then RMSNorm·w → bf16 (`ops/loop.py:18-132`, `loop.py:112-130`). No oracle change | a12 |
| 6 | Summation order (informational, oracle unchanged): `linear_k_groups` / `k_group_partials` accumulate fp32 per 1024-wide K group and sum the groups in order (`ops/linear.py:202-290`); used for the router logits (`ops/moe.py:978`, K = 2304 → 3 groups), the dynamic conv taps (`sconv.py:106`) and the per-stream controller `down` (`loop.py:60`, K = 4608 → 5 groups). The batched controller path uses `torch.bmm` instead. Only near-tie top-k routing can notice; port kernels should be rows-exact, not order-matched | a6, a12, a13 |
| 7 | **Tiny fixture regenerated** (review BLOCKER 1): θ_qk 3.0 (q and k now clip on 37-62 % of rows per layer kind, asserted), head_dim 32 ≠ d_rel, softcaps 2.0 / 1.5 so they bite; new seed. Rust tests must use the new files | `golden/inkling2-tiny/`, section i |
| 8 | **G1(b) is teacher-forced per visit** (BLOCKER 2): new `ref_dump_bf16_tf`, `blocks` / `bank` / `router_sel` / `route_margin` in every dump, `tf_check` reference comparator, routable-row rule, mutation check `tools/inkling2_mutants.py` | section i |
| 9 | G2 / G3 rewritten as KL + confident-row top-1 with measured headroom; what a correct bf16 engine scores is in the table (MAJOR 4) | section i |
| 10 | Ring semantics (margin ≥ 6, in-place two-segment reads, prefill `[ring tail ‖ chunk]`, write-before-read) and the slot-dtype rounding rule; prefix snapshot sizes (findings 8, 9) | a6, b, f |
| 11 | Op list revised to launch granularity: `InkInProj` (row-range input select), `ConvMix` (SconvRes + residual + next AttnResMix + optional LoopBlend + norm, router may fold in), per-row positions, MTP tap export, device argmax on raw logits (finding 10). The skeleton lane owns the contract | f |
| 12 | Minors: `· route_scale` not `· 8` (a13); message mix uses exactly `bank[:1]`, `bank[1]`, loader refuses other core shapes (a12, oracle asserts); two eps attrs (a, notation); bf16 P before P·V added to `--rounding vllm`; ring duplicate-row write fixed; query-blocked attention (12k-token replay fits in memory) | a, oracle |
| 13 | New oracle modes `replay` / `mkcapture` for the vLLM module-boundary capture (finding 7); `selftest --chunks/--no-decode/--prompt-file/--n-tokens` | oracle, section i |

Still open:

| Question | Why it matters | Pointer |
|---|---|---|
| The W4A4 SiTU uses `ex2.approx`/`rcp.approx` sigmoid and `tanh.approx` softcaps; the oracle uses exact functions | only for a bitwise W4A4 mode, which (4) shows is not worth chasing | `ops/activation.py:20-58` |
| Batched controller (`torch.bmm`, cuBLAS order) vs per-stream `linear_k_groups`: which one the loop model runs at batch 1 | summation order only | `loop.py:60, 112-130` |
| llama.cpp's SITU op and GGUF gate/up order | three-way diff completeness only | `ggml` not in the extract |
| tau on real weights and W4A4 exactness | no golden reaches position 8191; W4A4 emulation exactness is unproven | `replay` on the spark2 capture (`golden/inkling2-capture/`, pending); expected attention rel err ≈ 3e-3 at `--rounding vllm` |

## k. vLLM 12k module-boundary replay (2026-11-21, coordinator)

Capture: `golden/inkling2-capture/` (ink/golden f038c46f6): one 12065-token prefill on GB10 vLLM, all 48 visits. Replay: `inkling2_ref.py replay --rounding vllm`, endor CPU, logs in `docs/results/raw/inkling2-replay/`.

| Module | act quant | Visits | Relative error vs vLLM |
|---|---|---|---|
| Attention (prologue + attn + gate + wo) | n/a | all 48 | 1.6e-3 to 2.9e-3 (bf16 level); full layers at positions up to 12064, tau max abs diff 6.1e-8 |
| MoE, NVFP4 layers | W4A4 exact | 44 | 3.3e-3 to 6.0e-3 |
| MoE, MXFP8 layers 20-23 | W4A4 exact | 4 | 7.0e-4 to 9.3e-4 |
| MoE, NVFP4 layers | W4A16 | 8 sampled | 3.8e-3 (v00, layer 0) up to 1.06e-1 (core); v47 3.3e-2 |

Conclusions: (1) window, extent, beyond-extent bias, sink, clips, dynamic taps and tau above 8192 match vLLM: attention is settled. (2) The exact W4A4 emulation reproduces vLLM's experts to bf16 precision, so the GB10 goldens are W4A4 and the noise model holds. (3) W4A16 differs from vLLM's W4A4 experts by up to 10 % per module but scores closer end to end (section i: long 1.24e-2 vs 1.54e-2): W4A4 is a noisy perturbation that compounds over 48 visits, and W4A16 sits near its mean. Default stays W4A16 (vLLM's own SM90 numerics); W4A4 stays an opt-in mode for FP4 tensor cores.
